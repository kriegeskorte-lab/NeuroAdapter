"""
Model Parameter Counter

Analyzes the Brain Adapter architecture to count:
- Total parameters in each component
- Frozen vs trainable parameters
- Percentage of diffusion model that's frozen
"""

import argparse
import torch
from types import SimpleNamespace

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from brain_adapter.ip_adapter.ip_adapter import ImageProjModel
from brain_adapter.ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from brain_adapter.ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor, 
        AttnProcessor2_0 as AttnProcessor
    )
else:
    from brain_adapter.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from brain_adapter.model import GuidanceGenerator


def format_params(num_params):
    """Format parameter count in millions."""
    return f"{num_params / 1e6:.2f}M"


def count_parameters(model, name="Model", show_details=False):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n{name}:")
    print(f"  Total parameters: {total_params:,} ({format_params(total_params)})")
    print(f"  Trainable parameters: {trainable_params:,} ({format_params(trainable_params)})")
    print(f"  Frozen parameters: {frozen_params:,} ({format_params(frozen_params)})")
    if total_params > 0:
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    if show_details and hasattr(model, 'named_parameters'):
        trainable_layers = [name for name, p in model.named_parameters() if p.requires_grad]
        if trainable_layers:
            print(f"  Trainable layers: {len(trainable_layers)}")
    
    return total_params, trainable_params, frozen_params


def setup_ip_adapter(unet, num_fmri_tokens, condition_dim):
    """Set up IP-Adapter components."""
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=condition_dim,
    )
    
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    # Count cross-attention layers
    num_cross_attn = 0
    num_self_attn = 0
    
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
            num_self_attn += 1
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_fmri_tokens
            )
            attn_procs[name].load_state_dict(weights)
            num_cross_attn += 1
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    return image_proj_model, adapter_modules, num_cross_attn, num_self_attn


def main(args):
    """Analyze model parameters."""
    print("="*80)
    print("BRAIN ADAPTER MODEL PARAMETER ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Subject ID: {args.subject_id}")
    print(f"  Top-k parcels: {args.topk}")
    print(f"  Decoder queries: {args.num_decoder_queries}")
    print(f"  Condition dim: {args.condition_dim}")
    print(f"  Sub-approach: {args.sub_approach}")
    
    model_path = args.pretrained_model_name_or_path
    
    # Load pre-trained models
    print("\nLoading pre-trained models...")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    
    # Freeze base models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Count base model parameters
    print("\n" + "="*80)
    print("1. BASE STABLE DIFFUSION COMPONENTS (All Frozen)")
    print("="*80)
    
    te_total, te_train, te_frozen = count_parameters(text_encoder, "CLIP Text Encoder")
    vae_total, vae_train, vae_frozen = count_parameters(vae, "VAE")
    unet_total, unet_train, unet_frozen = count_parameters(unet, "UNet (before IP-Adapter)")
    
    sd_total = te_total + vae_total + unet_total
    print(f"\nStable Diffusion Total: {sd_total:,} ({format_params(sd_total)})")
    
    # Set up IP-Adapter
    num_fmri_tokens = abs(args.topk) * 2
    print(f"\n\nSetting up IP-Adapter with {num_fmri_tokens} fMRI tokens...")
    image_proj_model, adapter_modules, num_cross_attn, num_self_attn = setup_ip_adapter(
        unet, num_fmri_tokens, args.condition_dim
    )
    
    # Count IP-Adapter parameters
    print("\n" + "="*80)
    print("2. IP-ADAPTER COMPONENTS (Trainable)")
    print("="*80)
    print(f"Architecture Info:")
    print(f"  Cross-attention layers with IP-Adapter: {num_cross_attn}")
    print(f"  Self-attention layers (no adapter): {num_self_attn}")
    print(f"  Total attention layers: {num_cross_attn + num_self_attn}")
    
    proj_total, proj_train, proj_frozen = count_parameters(image_proj_model, "Image Projection Model", show_details=True)
    adapter_total, adapter_train, adapter_frozen = count_parameters(adapter_modules, "Adapter Modules")
    
    # Set up Guidance Generator
    print(f"\n\nSetting up Guidance Generator ({args.sub_approach})...")
    
    # Create dummy dataset args to get parcel info
    dataset_args = SimpleNamespace(
        subj=args.subject_id,
        backbone_arch="dinov2_q",
        data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
        imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
        parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
        hemi=None,
        tokenizer=tokenizer,
        gen_size=512,
        num_decoder_queries=args.num_decoder_queries,
    )
    
    # Import dataset to get parcel dimensions
    from brain_adapter.dataset import nsd_topk_parcel_dataset
    temp_dataset = nsd_topk_parcel_dataset(dataset_args, split='train', transform=None, topk=args.topk)
    
    guidance_generator = GuidanceGenerator(
        num_parcels=temp_dataset.num_parcels,
        max_voxels=temp_dataset.max_voxels,
        num_decoder_queries=args.num_decoder_queries,
        output_dim=args.condition_dim,
        sub_approach=args.sub_approach,
    )
    
    print("\n" + "="*80)
    print("3. GUIDANCE GENERATOR (Trainable)")
    print("="*80)
    print(f"Architecture Info:")
    print(f"  Input shape: [{temp_dataset.num_parcels} parcels × {temp_dataset.max_voxels} voxels]")
    print(f"  Output shape: [{num_fmri_tokens} tokens × {args.condition_dim} dims]")
    print(f"  Approach: {args.sub_approach}")
    
    gg_total, gg_train, gg_frozen = count_parameters(guidance_generator, "Guidance Generator", show_details=True)
    
    # Detailed breakdown
    print("\n" + "="*80)
    print("4. NEUROADAPTER MODULE BREAKDOWN")
    print("="*80)
    neuroadapter_total = proj_total + adapter_total + gg_total
    neuroadapter_train = proj_train + adapter_train + gg_train
    
    print(f"\nNeuroAdapter Total: {neuroadapter_total:,} ({format_params(neuroadapter_total)})")
    print(f"\nComponent breakdown:")
    print(f"  1) Guidance Generator: {gg_train:,} ({format_params(gg_train)}) - {100 * gg_train / neuroadapter_train:.1f}%")
    print(f"  2) Image Projection Model: {proj_train:,} ({format_params(proj_train)}) - {100 * proj_train / neuroadapter_train:.1f}%")
    print(f"  3) Adapter Modules: {adapter_train:,} ({format_params(adapter_train)}) - {100 * adapter_train / neuroadapter_train:.1f}%")
    
    # Summary
    print("\n" + "="*80)
    print("5. OVERALL TRAINING SUMMARY")
    print("="*80)
    
    total_all = sd_total + proj_total + adapter_total + gg_total
    trainable_all = te_train + vae_train + unet_train + proj_train + adapter_train + gg_train
    frozen_all = te_frozen + vae_frozen + unet_frozen + proj_frozen + adapter_frozen + gg_frozen
    
    print(f"\n📊 TOTAL MODEL:")
    print(f"  Total parameters: {total_all:,} ({format_params(total_all)})")
    print(f"  Trainable parameters: {trainable_all:,} ({format_params(trainable_all)})")
    print(f"  Frozen parameters: {frozen_all:,} ({format_params(frozen_all)})")
    print(f"  Trainable percentage: {100 * trainable_all / total_all:.2f}%")
    
    print(f"\n🔒 STABLE DIFFUSION (Frozen):")
    print(f"  Total SD parameters: {sd_total:,} ({format_params(sd_total)})")
    print(f"  % of SD frozen: {100 * frozen_all / sd_total:.2f}%")
    print(f"  Text Encoder: {te_frozen:,} ({format_params(te_frozen)}) - {100 * te_frozen / sd_total:.1f}%")
    print(f"  VAE: {vae_frozen:,} ({format_params(vae_frozen)}) - {100 * vae_frozen / sd_total:.1f}%")
    print(f"  UNet (base): {unet_frozen:,} ({format_params(unet_frozen)}) - {100 * unet_frozen / sd_total:.1f}%")
    
    print(f"\n🎯 NEUROADAPTER (Trainable):")
    print(f"  Total trainable: {trainable_all:,} ({format_params(trainable_all)})")
    print(f"  % of total model: {100 * trainable_all / total_all:.2f}%")
    print(f"  Guidance Generator: {gg_train:,} ({format_params(gg_train)}) - {100 * gg_train / trainable_all:.1f}% of trainable")
    print(f"  Image Proj Model: {proj_train:,} ({format_params(proj_train)}) - {100 * proj_train / trainable_all:.1f}% of trainable")
    print(f"  Adapter Modules: {adapter_train:,} ({format_params(adapter_train)}) - {100 * adapter_train / trainable_all:.1f}% of trainable")
    
    print(f"\n📈 TRAINING EFFICIENCY:")
    print(f"  Training only {100 * trainable_all / total_all:.2f}% of total parameters")
    print(f"  Parameter ratio (trainable/frozen): 1:{frozen_all/trainable_all:.1f}")
    print(f"  Memory footprint reduction: ~{100 * (1 - trainable_all/total_all):.1f}% (gradients not needed for frozen)")
    
    # Memory estimation (rough)
    print(f"\n💾 ESTIMATED MEMORY (FP32):")
    trainable_memory = trainable_all * 4 / (1024**3)  # 4 bytes per param, convert to GB
    frozen_memory = frozen_all * 4 / (1024**3)
    total_memory = total_all * 4 / (1024**3)
    print(f"  Model weights: {total_memory:.2f} GB")
    print(f"  Trainable (with gradients): {trainable_memory * 2:.2f} GB (weights + gradients)")
    print(f"  Frozen (no gradients): {frozen_memory:.2f} GB (weights only)")
    print(f"  Total training memory: ~{trainable_memory * 2 + frozen_memory:.2f} GB (excluding activations & optimizer states)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters in Brain Adapter model")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained Stable Diffusion model"
    )
    parser.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Subject ID for fMRI data"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of top parcels to use"
    )
    parser.add_argument(
        "--num_decoder_queries",
        type=int,
        default=50,
        help="Number of decoder queries"
    )
    parser.add_argument(
        "--condition_dim",
        type=int,
        default=768,
        help="Dimension of conditioning tokens"
    )
    parser.add_argument(
        "--sub_approach",
        type=str,
        default="linear_projection",
        choices=["linear_projection", "transformer_decoder"],
        help="Sub-approach for Guidance Generator"
    )
    
    args = parser.parse_args()
    main(args)


"""
Usage Examples:

# Basic usage with linear projection:
python count_model_params.py

# With transformer decoder:
python count_model_params.py --sub_approach transformer_decoder

# With different configuration:
python count_model_params.py \
    --topk 200 \
    --num_decoder_queries 100 \
    --condition_dim 1280 \
    --sub_approach transformer_decoder
"""
