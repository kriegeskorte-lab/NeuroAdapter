"""
Brain Adapter Training Script

This script trains a brain adapter model that guides a diffusion model using fMRI tokens.
It adopts the IP-Adapter strategy to inject brain-derived conditioning tokens into 
the diffusion process, enabling fMRI-to-image generation.

The training pipeline:
1. Loads pre-trained Stable Diffusion components (UNet, VAE, Text Encoder)
2. Sets up IP-Adapter architecture with brain-specific modifications  
3. Creates guidance generator to convert fMRI data to conditioning tokens
4. Trains the adapter modules while keeping SD backbone frozen
"""

# Standard library imports
import os
import sys
import glob
import shutil
import argparse
import time
import itertools
from types import SimpleNamespace
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
from pathlib import Path
import wandb

# Accelerate imports
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, synchronize_rng_states

# Diffusion model imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import get_cosine_schedule_with_warmup

# Project-specific imports
from brain_adapter.ip_adapter.ip_adapter import ImageProjModel
from brain_adapter.ip_adapter.utils import is_torch2_available

# Conditional imports based on torch version
if is_torch2_available():
    from brain_adapter.ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor, 
        AttnProcessor2_0 as AttnProcessor
    )
else:
    from brain_adapter.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from brain_adapter.model import NeuroAdapter, GuidanceGenerator
from brain_adapter.utils import str2bool
from brain_adapter.dataset import nsd_topk_parcel_dataset, nsd_groupwise_topk_parcel_dataset
from brain_adapter.loss import min_snr_loss_weights, dispersive_loss

warnings.filterwarnings("ignore")

def setup_accelerator(args):
    """Initialize the Accelerator for distributed training."""
    set_seed(args.seed)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=False,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        device_placement=True,
        split_batches=False,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=None,
        log_with=None,
        project_dir=None,
        project_config=None,
    )
    accelerator.wait_for_everyone()
    return accelerator


def setup_wandb_logging(args, accelerator):
    """Initialize Weights & Biases logging if enabled."""
    if accelerator.is_main_process and args.wandb:
        # Note: Consider moving the API key to environment variable for security
        subj_info = args.subject_id if args.training_subjects is None else "_".join(map(str, args.training_subjects))
        wandb.login(key="fa2d96cf662daa2fc63a8242133501a23399a230")
        wandb.init(
            project="brain-decoding",
            entity='tonylovescode',
            name=f"fMRI_subj{subj_info}_{args.time}",
            config={
                "subject_id": subj_info,
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "epochs": args.num_train_epochs,
                "approach": "brain_adapter",
                "sub-approach": args.sub_approach,
                "topk": args.topk,
            }
        )


def load_pretrained_models(args):
    """Load pre-trained Stable Diffusion components."""
    model_path = args.pretrained_model_name_or_path
    
    # Load core diffusion components
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    # Freeze parameters to save memory (only adapter modules will be trained)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    return noise_scheduler, tokenizer, text_encoder, vae, unet


def setup_ip_adapter(unet, args):
    """Set up IP-Adapter components for brain conditioning."""
    topk = args.topk
    # Calculate number of fMRI tokens (multiply by 2 for left/right hemispheres)
    num_fmri_tokens = abs(topk) * 2  # if topk is negative, use parcels with lowest SNR
    
    # Create image projection model for fMRI tokens
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim, #768
        clip_embeddings_dim=args.condition_dim,
    )
    
    # Initialize attention processor modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    for name in unet.attn_processors.keys():
        # Determine if this is cross-attention or self-attention
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        # Get hidden size based on UNet block type
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        # Set up appropriate attention processor
        if cross_attention_dim is None:
            # Self-attention layers use standard processor
            attn_procs[name] = AttnProcessor()
        else:
            # Cross-attention layers use IP-Adapter processor
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
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    return image_proj_model, adapter_modules, num_fmri_tokens


def setup_dataset_and_dataloader(args, tokenizer):
    """Create dataset and dataloader for training."""
    dataset_args = SimpleNamespace(
        subj=args.subject_id,
        backbone_arch="dinov2_q",
        data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
        imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
        parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
        hemi=None,
        tokenizer=tokenizer,
        gen_size=args.gen_img_resolution,
        num_decoder_queries=args.num_decoder_queries,
    )

    if args.training_subjects is not None:
        train_dataset = nsd_groupwise_topk_parcel_dataset(
            dataset_args, 
            split='train', 
            transform=None, 
            topk=args.topk, 
            train_subj=args.training_subjects
        )
    else:
        train_dataset = nsd_topk_parcel_dataset(
            dataset_args,
            split='train',
            transform=None,
            topk=args.topk
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    return train_dataset, train_dataloader


def setup_models_and_optimizer(args, accelerator, train_dataset, image_proj_model, adapter_modules, unet):
    """Set up the neural adapter, guidance generator, and optimizer."""
    # Create NeuroAdapter
    neuro_adapter = NeuroAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    # Create GuidanceGenerator for converting fMRI data to conditioning tokens
    guidance_generator = GuidanceGenerator(
        num_parcels=train_dataset.num_parcels, 
        max_voxels=train_dataset.max_voxels, 
        num_decoder_queries=args.num_decoder_queries,  # dataset_args.num_decoder_queries
        output_dim=args.condition_dim,  # 768 for CLIP
        sub_approach=args.sub_approach,  # 'linear_projection' or 'transformer_decoder'
    )
    
    # Prepare models with accelerator
    neuro_adapter, guidance_generator = accelerator.prepare(neuro_adapter, guidance_generator)
    accelerator.register_for_checkpointing(guidance_generator)
    
    # Helper function to unwrap DDP models
    # access the original underlying PyTorch model 
    # when it may be wrapped by a parallelization/distributed training 
    # wrapper, like DistributedDataParallel (DDP) or DataParallel.
    def unwrap_ddp(model):
        return model.module if hasattr(model, "module") else model
    
    ip_adapter_unwrapped = unwrap_ddp(neuro_adapter)
    guidance_generator_unwrapped = unwrap_ddp(guidance_generator)
    
    # Set up optimizer for trainable parameters
    params_to_opt = itertools.chain(
        ip_adapter_unwrapped.image_proj_model.parameters(),
        ip_adapter_unwrapped.adapter_modules.parameters(),
        guidance_generator.parameters()
    )
    
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = accelerator.prepare(optimizer)
    
    return neuro_adapter, guidance_generator, optimizer, ip_adapter_unwrapped, guidance_generator_unwrapped


def setup_weight_dtype(accelerator):
    """Determine the appropriate weight dtype based on mixed precision setting."""
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def apply_fmri_token_dropout(condition_tokens):
    """Apply random dropout to fMRI tokens for regularization."""
    B, N, _ = condition_tokens.shape
    ratios = torch.rand(B, 1)
    roi_mask = (torch.rand(B, N) <= ratios).to(condition_tokens.device)
    roi_mask = roi_mask.view(B, N, 1)
    return condition_tokens * roi_mask

def apply_feature_dropout(condition_tokens, dropout_prob=0.5):
    """
    Apply random dropout to features inside each parcel token.
    
    Args:
        condition_tokens: (B, N, f) tensor
        dropout_prob: probability of dropping a feature

    Returns:
        masked_tokens: (B, N, f) tensor
    """
    if dropout_prob <= 0:
        return condition_tokens
    mask = (torch.rand_like(condition_tokens) > dropout_prob).float()
    return condition_tokens * mask

def apply_vertex_feature_dropout(vertex_data, dropout_prob=0.5):
    """
    Apply random dropout to vertices within each parcel.

    Args:
        vertex_data: (B, N, V) tensor
            B = batch size
            N = number of parcels
            V = number of vertices per parcel (padded to vmax)
        dropout_prob: probability of dropping a vertex

    Returns:
        masked_vertex_data: (B, N, V) tensor
    """
    if dropout_prob <= 0:
        return vertex_data
    mask = (torch.rand_like(vertex_data) > dropout_prob).float()
    return vertex_data * mask
 

def process_training_batch(batch, vae, noise_scheduler, text_encoder, guidance_generator, 
                          neuro_adapter, weight_dtype, accelerator, train_dataset):
    """Process a single training batch through the diffusion model."""
    with torch.no_grad():
        # Prepare input data
        img_ipadapter = batch["img_ipadapter"].to(accelerator.device, dtype=weight_dtype)
        text_input_ids = batch["text_input_ids"].to(accelerator.device)
        
        # Encode images to latent space
        latents = vae.encode(img_ipadapter).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Prepare diffusion training setup
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()
    
    # Forward diffusion process: add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Process brain data
    lh = batch["brain_lh_f"].to(accelerator.device, dtype=weight_dtype)  # [B, parcels, max_voxels]
    rh = batch["brain_rh_f"].to(accelerator.device, dtype=weight_dtype)  # [B, parcels, max_voxels]
    brain_data = torch.cat([lh, rh], dim=1)
    
    # Validate brain data dimensions
    assert brain_data.shape[1] == train_dataset.num_parcels, f"Expected {train_dataset.num_parcels} parcels, got {brain_data.shape[1]}"
    assert brain_data.shape[2] == train_dataset.max_voxels, f"Expected {train_dataset.max_voxels} voxels, got {brain_data.shape[2]}"
    
    # brain_data = apply_vertex_feature_dropout(brain_data, dropout_prob=0.25)
    # Generate conditioning tokens from brain data
    # Linear projection: [B, parcels, V] → [B, parcels, 768]
    condition_tokens, _ = guidance_generator(brain_data)
    
    # Apply dropout for regularization
    condition_tokens = apply_fmri_token_dropout(condition_tokens)

    # Get text embeddings
    # WARNING: we can set this to None, otherwise the computation in IPAttnProcessor would be unsafe.
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input_ids)[0]
    
    # Forward pass through adapted UNet
    noise_pred, cross_attn_maps = neuro_adapter(noisy_latents, timesteps, encoder_hidden_states, condition_tokens)
    # noise_pred = neuro_adapter(noisy_latents, timesteps, encoder_hidden_states, condition_tokens)
    
    # # Compute primary noise prediction loss
    # train_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
    
    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape))))  # Average over spatial dimensions
    snr_weights = min_snr_loss_weights(timesteps, noise_scheduler, gamma=5.0)
    loss = loss * snr_weights
    train_loss = loss.mean()
    
    return train_loss


def training_loop(args, accelerator, neuro_adapter, guidance_generator, train_dataloader, 
                 optimizer, lr_scheduler, vae, text_encoder, noise_scheduler, weight_dtype, train_dataset):
    """Main training loop."""
    global_step = 0
    
    for epoch in range(0, args.num_train_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        begin = time.perf_counter()
        
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            acc_modules = (neuro_adapter, guidance_generator)
            
            with accelerator.accumulate(*acc_modules):
                # Process batch and compute loss
                train_loss = process_training_batch(
                    batch, vae, noise_scheduler, text_encoder, guidance_generator,
                    neuro_adapter, weight_dtype, accelerator, train_dataset, 
                )

                # Gather losses across all processes for logging
                avg_train_loss = accelerator.gather(train_loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagation
                accelerator.backward(train_loss)

                # Gradient clipping if specified
                if args.clip_max_norm > 0:
                    params_to_opt = itertools.chain(
                        neuro_adapter.image_proj_model.parameters() if not hasattr(neuro_adapter, 'module') 
                        else neuro_adapter.module.image_proj_model.parameters(),
                        neuro_adapter.adapter_modules.parameters() if not hasattr(neuro_adapter, 'module')
                        else neuro_adapter.module.adapter_modules.parameters(),
                        guidance_generator.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_opt, args.clip_max_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler:
                    lr_scheduler.step()

                # Logging
                if accelerator.is_main_process and args.wandb and step % 100 == 0:
                    wandb.log({
                        'train_loss': avg_train_loss,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'step': global_step
                    })

                if accelerator.is_main_process and step == len(train_dataloader) - 1:
                    tqdm.write("Epoch {}, step {}, data_time: {:.3f}s, time: {:.3f}s, avg_train_loss: {:.6f}".format(
                            epoch, step, load_data_time, time.perf_counter() - begin, avg_train_loss))

            global_step += 1

            begin = time.perf_counter()
            progress_bar.update(1)
        
        # Save checkpoints every X epochs
        if (epoch + 1) % args.save_epochs == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
            accelerator.save_state(save_path, safe_serialization=False)
            if accelerator.is_main_process:
                tqdm.write(f"Saved checkpoint to {save_path}")
                cleanup_checkpoints(args.output_dir, keep_last_n=1)
            
            
def cleanup_checkpoints(output_dir, keep_last_n=3):
    """Clean up old checkpoints, keeping only the last N. Safe for distributed training."""
    # List all checkpoint directories
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split('-')[-1])
    )
    # Delete older checkpoints, keep only the last N
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            if os.path.exists(checkpoint):
                print(f"Deleting old checkpoint: {checkpoint}")
                shutil.rmtree(checkpoint)
        except (FileNotFoundError, OSError) as e:
            # Another process may have already deleted this checkpoint
            print(f"Checkpoint {checkpoint} already deleted or inaccessible: {e}")
            continue


def main(args):
    """Main training function."""
    # Set random seeds for reproducibility
    seed = args.seed if hasattr(args, 'seed') else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize training setup
    accelerator = setup_accelerator(args)
    accelerator.print(f'Number of GPUs: {accelerator.state.num_processes}')
    setup_wandb_logging(args, accelerator)
    
    # Load pre-trained models
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_pretrained_models(args)

    # Set up IP-Adapter components
    image_proj_model, adapter_modules, num_fmri_tokens = setup_ip_adapter(unet, args)
    accelerator.print(f"Number of fMRI tokens: {num_fmri_tokens}")
    
    # Set up dataset and dataloader
    train_dataset, train_dataloader = setup_dataset_and_dataloader(args, tokenizer)
    train_dataloader = accelerator.prepare(train_dataloader)
    accelerator.print(f"Training dataloader length: {len(train_dataloader)}")
    
    # Set up models and optimizer
    neuro_adapter, guidance_generator, optimizer, _, _ = setup_models_and_optimizer(
        args, accelerator, train_dataset, image_proj_model, adapter_modules, unet
    )
    
    # Determine weight dtype and move models to device
    weight_dtype = setup_weight_dtype(accelerator)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Set up learning rate scheduler
    steps_per_epoch = len(train_dataloader)
    # lr_scheduler = StepLR(optimizer, step_size=25*steps_per_epoch, gamma=0.5)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0.1 * args.num_train_epochs * steps_per_epoch,
    #     num_training_steps=args.num_train_epochs * steps_per_epoch
    # )
    lr_scheduler = None
    
    # Start training
    accelerator.print("Starting training...")
    training_loop(
        args, accelerator, neuro_adapter, guidance_generator, train_dataloader,
        optimizer, lr_scheduler, vae, text_encoder, noise_scheduler, 
        weight_dtype, train_dataset
    )
    
    accelerator.print("Training completed!")


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Brain Adapter model for fMRI-guided image generation."
    )
    
    # Model paths
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained IP adapter model. If not specified, weights are initialized randomly.",
    )
    
    # Dataset and output paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/engram/nklab/coco/images/train2017/",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/engram/nklab/pf2477/brain_decoding/brain_adapter/model_weights",
        help="Output directory for model predictions and checkpoints.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory. Defaults to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.",
    )
    
    # Training configuration
    parser.add_argument(
        "--training_subjects",
        nargs="+",         # one or more values, split on whitespace
        type=int,          # convert each to int
        default=None,
        help="List of subject IDs to include (e.g. 1 2 3 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-6, 
        help="Weight decay for regularization."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=8, 
        help="Batch size per device for training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses for data loading. 0 means data loaded in main process.",
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=10,
        help="Save checkpoint every X epochs",
    )
    parser.add_argument(
        '--clip_max_norm', 
        default=1.0, 
        type=float,
        help='Gradient clipping max norm'
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training. bf16 requires PyTorch >= 1.10 and Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    
    # Brain data
    parser.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Subject ID for fMRI data. Default is 1.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of top parcels to use for fMRI conditioning",
    )
    parser.add_argument(
        "--num_decoder_queries",
        type=int,
        default=50,
        help="Number of queries for the transformer decoder in the NeuroAdapter",
    )
    parser.add_argument(
        "--condition_dim",
        type=int,
        default=768,
        help="Dimension of the conditioning tokens (e.g., 768 for CLIP)",
    )
    
    # Image processing configuration
    parser.add_argument(
        "--encoder_resolution",
        type=int,
        default=224,
        help="Resolution for images input to image encoder",
    )
    parser.add_argument(
        "--make_square",
        type=int,
        default=224,
        help="Whether to put image in square before resizing",
    )
    parser.add_argument(
        "--gen_img_resolution",
        type=int,
        default=512,
        help="Resolution for images input to Stable Diffusion",
    )
    
    # Experiment tracking and misc
    parser.add_argument(
        "--sub_approach", 
        type=str, 
        default="linear_projection", 
        help="Sub-approach for training. Options: 'linear_projection', 'transformer_decoder'"
    )
    parser.add_argument(
        "-wb", "--wandb", 
        action='store_true',
        help="Whether to use W&B for progress tracking"
    )
    parser.add_argument(
        "--time", 
        type=str, 
        default=None,
        help="Timestamp for this training run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )
    
    return parser


if __name__ == "__main__":
    # Create argument parser and parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle distributed training local rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Create output directory if training (learning_rate > 0)
    if args.learning_rate > 0:
        # 1) If time wasn’t passed in, generate it now
        if args.time is None:
            args.time = datetime.now().strftime("%m_%d_%Y-%H_%M")

        # 2) Append the time to the base output_dir and make it
        args.output_dir = os.path.join(args.output_dir, args.time)
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Start training
    main(args)


"""
Usage Examples:

# Quick test run (no training):
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 0 \
    --num_train_epochs 1 \
    --train_batch_size 16 \
    --dataloader_num_workers=8 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 768 \
    --sub_approach linear_projection
    
accelerate launch --config_file acc_config.yaml --num_processes 4 train_brain_adapter.py \
    --learning_rate 0 \
    --num_train_epochs 1 \
    --train_batch_size 16 \
    --dataloader_num_workers=8 \
    --training_subjects 1 2 3 4 5 6 7 8 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach linear_projection \

# Full training run:
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 16 \
    --dataloader_num_workers=8 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach transformer_decoder \
    --wandb

# Training with existing IP-Adapter weights:
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --pretrained_ip_adapter_path brain_adapter/ip_adapter/checkpoints/ip-adapter_sd15.bin \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 16 \
    --dataloader_num_workers=8 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 1280 \
    --num_decoder_queries 50 \
    --sub_approach transformer_decoder \
    --wandb

# Training with transformer decoder:
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 16 \
    --dataloader_num_workers=8 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach transformer_decoder \
    --wandb

# Training with different subject:
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 16 \
    --dataloader_num_workers=8 \
    --subject_id 2 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach transformer_decoder \
    --wandb
"""