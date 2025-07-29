"""
Brain Adapter Decoding Script

This script uses a trained brain adapter model to decode/generate images from fMRI brain signals.
It loads the trained model weights and performs image generation using brain-derived conditioning tokens.

The decoding pipeline:
1. Loads trained brain adapter and guidance generator models
2. Processes fMRI brain data to generate conditioning tokens
3. Uses diffusion model with brain conditioning to generate multiple candidate images
4. Evaluates candidates using brain encoder correlation and selects best match
"""

# Standard library imports
import os
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace
import warnings

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torchvision import transforms

# Diffusion imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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

from brain_adapter.model import GuidanceGenerator
from brain_adapter.dataset import nsd_topk_parcel_dataset

# os.chdir("/engram/nklab/pf2477/brain_decoding/whole_brain_encoder")
from whole_brain_encoder.brain_encoder_wrapper import BrainEncoderWrapper

# os.chdir("/engram/nklab/pf2477/brain_decoding/")

warnings.filterwarnings("ignore")

class BrainIPAdapter(torch.nn.Module):
    """
    Brain-adapted IP-Adapter for fMRI-guided image generation.
    
    This class wraps the UNet with IP-Adapter components and provides methods
    for loading trained weights and performing classifier-free guidance.
    """

    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.ip_ckpt = ckpt_path

        if ckpt_path is not None:
            self.load_ip_adapter()

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, brain_embeds):
        """
        Forward pass with classifier-free guidance.
        
        Args:
            noisy_latents: Noisy image latents
            timesteps: Diffusion timesteps
            encoder_hidden_states: Text embeddings
            brain_embeds: Brain-derived conditioning tokens
            
        Returns:
            Tuple of (unconditional_pred, conditional_pred)
        """
        # Create conditional and unconditional brain embeddings
        brain_embeds_cond = brain_embeds
        brain_embeds_uncond = torch.zeros_like(brain_embeds)
        
        # Project brain embeddings to IP tokens
        ip_tokens_cond = self.image_proj_model(brain_embeds_cond)
        ip_tokens_uncond = self.image_proj_model(brain_embeds_uncond)
        
        # Concatenate with text embeddings
        encoder_hidden_states_cond = torch.cat([encoder_hidden_states, ip_tokens_cond], dim=1)
        encoder_hidden_states_uncond = torch.cat([encoder_hidden_states, ip_tokens_uncond], dim=1)
        
        # Batch conditional and unconditional for efficient inference
        combined_encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)
        combined_noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        
        # Handle timestep broadcasting for the combined batch
        if isinstance(timesteps, torch.Tensor):
            combined_timesteps = torch.cat([timesteps, timesteps], dim=0) if timesteps.dim() > 0 else timesteps
        else:
            combined_timesteps = timesteps
        
        # Single forward pass through U-Net
        combined_noise_pred = self.unet(combined_noisy_latents, combined_timesteps, combined_encoder_hidden_states).sample
        
        # Split predictions
        noise_pred_uncond, noise_pred_cond = combined_noise_pred.chunk(2)
        return noise_pred_uncond, noise_pred_cond

    def load_from_checkpoint(self, ckpt_path: str):
        """Load model weights from checkpoint with verification."""
        # Calculate original checksums for verification
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Load state dictionary
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load weights
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Verify weights changed
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        assert orig_ip_proj_sum != new_ip_proj_sum, "Image projection model weights did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Adapter module weights did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def load_ip_adapter(self):
        """Load IP-Adapter weights from checkpoint file."""
        try:
            if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
                from safetensors import safe_open
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(self.ip_ckpt, map_location="cpu")
                
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
            ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])
            print(f"Successfully loaded IP-Adapter weights from {self.ip_ckpt}")
        except Exception as e:
            print(f"Warning: Could not load IP-Adapter weights: {e}")

    def set_scale(self, scale):
        """
        Set IP-Adapter attention scales per transformer block.

        Args:
            scale: Can be:
                - float: Global scale for all blocks
                - dict: Granular control per block type and layer
        """
        if not isinstance(scale, (float, dict)):
            raise ValueError("Scale must be float or dict")
            
        for name, attn_processor in self.unet.attn_processors.items():
            if isinstance(attn_processor, IPAttnProcessor):
                if isinstance(scale, float):
                    attn_processor.scale = scale
                else:
                    # Handle dictionary-based scaling
                    for block_type, blocks in scale.items():
                        if name.startswith(block_type):
                            for block_name, block_scale in blocks.items():
                                if block_name in name:
                                    attn_processor.scale = block_scale


def load_diffusion_models(model_path="runwayml/stable-diffusion-v1-5"):
    """Load pre-trained Stable Diffusion components."""
    print("Loading Stable Diffusion components...")
    
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    
    return noise_scheduler, tokenizer, text_encoder, vae, unet


def setup_ip_adapter_modules(args, unet, num_tokens=200):
    """Initialize IP-Adapter attention processor modules."""
    print("Setting up IP-Adapter modules...")
    
    # Create image projection model
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=args.conditioning_dim,
        clip_extra_context_tokens=num_tokens,
    )

    # Initialize attention processors
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        # Determine hidden size based on block type
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
                num_tokens=num_tokens,
            )
            attn_procs[name].load_state_dict(weights)
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    return image_proj_model, adapter_modules


def create_test_dataset(
    args,
    tokenizer, 
    topk=100, 
    eval_full_dataset=False,
    start_idx=0, 
    end_idx=8, 
    max_samples=None,
    batch_size=1, 
    num_workers=0
):
    """
    Create test dataset and dataloader with flexible evaluation options.
    
    Args:
        tokenizer: CLIP tokenizer for text processing
        topk: Number of top parcels to use per hemisphere
        eval_full_dataset: If True, use entire test dataset
        start_idx: Starting index (only used if eval_full_dataset=False)
        end_idx: Ending index (only used if eval_full_dataset=False)
        max_samples: Maximum number of samples to evaluate (None = no limit)
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        
    Returns:
        Tuple of (test_dataset, test_dataloader, actual_indices)
    """
    dataset_args = SimpleNamespace(
        subj=args.subject_id,
        backbone_arch="dinov2_q",
        data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
        imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
        parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
        hemi=None,
        tokenizer=tokenizer,
        gen_size=512,
    )

    test_dataset = nsd_topk_parcel_dataset(dataset_args, split="test", transform=None, topk=topk)
    total_samples = len(test_dataset)
    
    print(f"Test dataset created with {total_samples} total samples, using top {topk} parcels per hemisphere.")
    
    # Determine evaluation indices
    if eval_full_dataset:
        print("Evaluating on FULL test dataset")
        indices = list(range(total_samples))
        if max_samples is not None and max_samples < total_samples:
            indices = indices[:max_samples]
            print(f"Limited to first {max_samples} samples")
        actual_start, actual_end = 0, len(indices)
    else:
        print(f"Evaluating on SUBSET: indices {start_idx} to {end_idx}")
        # Validate indices
        if end_idx > total_samples:
            print(f"Warning: end_idx ({end_idx}) exceeds dataset size ({total_samples}). Adjusting to {total_samples}")
            end_idx = total_samples
        if start_idx >= total_samples:
            raise ValueError(f"start_idx ({start_idx}) exceeds dataset size ({total_samples})")
        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")
            
        indices = list(range(start_idx, end_idx))
        if max_samples is not None and len(indices) > max_samples:
            indices = indices[:max_samples]
            print(f"Limited to first {max_samples} samples from specified range")
        actual_start, actual_end = indices[0], indices[-1] + 1

    print(f"Final evaluation: {len(indices)} samples (indices {actual_start} to {actual_end-1})")

    # Create subset and dataloader
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    test_dataloader = torch.utils.data.DataLoader(
        test_subset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return test_dataset, test_dataloader, (actual_start, actual_end)


def setup_device():
    """Initialize device for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def validate_arguments(args):
    """Validate and process command line arguments."""
    # Handle evaluation mode logic
    if args.eval_full_dataset:
        print("Mode: Full dataset evaluation")
        if args.max_samples:
            print(f"  - Limited to first {args.max_samples} samples")
    else:
        print(f"Mode: Index-based evaluation ({args.start_idx} to {args.end_idx})")
        
    # Validate numerical arguments
    if args.num_predictions <= 0:
        raise ValueError("num_predictions must be positive")
    if args.noise_factor < 0:
        raise ValueError("noise_factor must be non-negative")
    if args.topk <= 0:
        raise ValueError("topk must be positive")
        
    return args


def estimate_memory_usage(num_samples, num_predictions, image_size=512):
    """Estimate memory usage for the evaluation."""
    # Rough estimates in GB
    base_model_memory = 8.0  # Base models (UNet, VAE, etc.)
    brain_encoder_memory = 4.0  # Brain encoder
    
    # Per sample memory (images + features)
    bytes_per_pixel = 4  # float32
    image_memory_per_sample = (num_predictions * image_size * image_size * 3 * bytes_per_pixel) / (1024**3)
    
    total_memory = base_model_memory + brain_encoder_memory + (num_samples * image_memory_per_sample)
    
    print(f"Estimated memory usage: {total_memory:.1f} GB")
    if total_memory > 16:
        print("Warning: High memory usage expected. Consider reducing batch_size or num_predictions")
    
    return total_memory


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_individual_results(
    original_images, 
    decoded_images, 
    candidate_images_list,
    correlation_scores_list,
    save_dir, 
    args, 
    evaluation_indices,
    dataset_indices,
    processing_time=None
):
    """
    Save results individually for each sample with organized folder structure.
    
    Args:
        original_images: Ground truth images
        decoded_images: Best decoded images
        candidate_images_list: List of all candidate images per sample (for subset mode)
        correlation_scores_list: List of correlation scores per sample (for subset mode)
        save_dir: Base save directory
        args: Command line arguments
        evaluation_indices: Tuple of (start, end) indices
        dataset_indices: Original dataset indices for each sample
        processing_time: Total processing time
    """
    # Create epoch-specific directory
    epoch_dir = os.path.join(save_dir, f"epoch_{args.saved_epochs}")
    Path(epoch_dir).mkdir(parents=True, exist_ok=True)
    
    # Create evaluation mode specific directory
    if args.eval_full_dataset:
        eval_dir = os.path.join(epoch_dir, "full")
        mode_name = "full_dataset"
    else:
        eval_dir = os.path.join(epoch_dir, "subset")
        mode_name = f"subset_{evaluation_indices[0]}_{evaluation_indices[1]-1}"
    
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to: {eval_dir}")
    
    # Create metadata for the entire evaluation
    global_metadata = {
        'model_weights_dir': args.model_weights_dir,
        'saved_epochs': args.saved_epochs,
        'num_predictions': args.num_predictions,
        'noise_factor': args.noise_factor,
        'topk': args.topk,
        'conditioning_dim': args.conditioning_dim,
        'num_decoder_queries': args.num_decoder_queries,
        'sub_approach': args.sub_approach,
        'subject_id': args.subject_id,
        'evaluation_mode': 'full_dataset' if args.eval_full_dataset else 'indices',
        'start_idx': evaluation_indices[0],
        'end_idx': evaluation_indices[1],
        'num_samples': len(original_images),
        'timestamp': np.datetime64('now').astype(str),
    }
    
    if processing_time:
        global_metadata['processing_time_seconds'] = processing_time
        global_metadata['time_per_sample'] = processing_time / len(original_images)
    
    # Save global metadata
    metadata_path = os.path.join(eval_dir, "evaluation_metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    # Save individual samples
    for i, (orig_img, decoded_img, dataset_idx) in enumerate(zip(original_images, decoded_images, dataset_indices)):
        # Use dataset index for filename to maintain consistency
        sample_filename = f"sample_{dataset_idx:06d}.npz"
        sample_path = os.path.join(eval_dir, sample_filename)
        
        # Prepare data to save
        save_data = {
            'groundtruth_image': orig_img.cpu().numpy(),
            'predicted_image': decoded_img.cpu().numpy(),
            'dataset_index': dataset_idx,
            'evaluation_index': i,
        }
        
        # For subset evaluation, optionally save all candidates
        if not args.eval_full_dataset and args.save_all_candidates:
            if i < len(candidate_images_list) and candidate_images_list[i] is not None:
                save_data['candidate_images'] = candidate_images_list[i]
                save_data['correlation_scores'] = correlation_scores_list[i]
                save_data['best_candidate_idx'] = np.argmax(correlation_scores_list[i])
        
        # Save individual sample
        np.savez_compressed(sample_path, **save_data)
        
        if (i + 1) % 50 == 0 or i == len(original_images) - 1:
            print(f"Saved {i + 1}/{len(original_images)} samples")
    
    # Create summary file with sample index mapping
    summary_data = {
        'evaluation_indices': evaluation_indices,
        'dataset_indices': dataset_indices,
        'sample_filenames': [f"sample_{idx:06d}.npz" for idx in dataset_indices],
        'num_samples': len(original_images),
        'mode': mode_name
    }
    
    summary_path = os.path.join(eval_dir, "sample_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved successfully!")
    print(f"Directory: {eval_dir}")
    print(f"Samples: {len(original_images)}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")
    
    return eval_dir


def load_evaluation_results(results_dir):
    """
    Utility function to load and examine saved evaluation results.
    
    Args:
        results_dir: Path to evaluation results directory (epoch_X/full or epoch_X/subset)
        
    Returns:
        Dictionary with loaded results and metadata
    """
    import json
    
    # Load metadata
    metadata_path = os.path.join(results_dir, "evaluation_metadata.json")
    summary_path = os.path.join(results_dir, "sample_summary.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Evaluation Results Summary:")
    print(f"  Mode: {metadata['evaluation_mode']}")
    print(f"  Samples: {metadata['num_samples']}")
    print(f"  Epoch: {metadata['saved_epochs']}")
    print(f"  Processing time: {metadata.get('processing_time_seconds', 'N/A')} seconds")
    
    # Load first sample as example
    first_sample_file = summary['sample_filenames'][0]
    first_sample_path = os.path.join(results_dir, first_sample_file)
    
    if os.path.exists(first_sample_path):
        sample_data = np.load(first_sample_path, allow_pickle=True)
        print(f"  Sample data keys: {list(sample_data.keys())}")
        
        if 'candidate_images' in sample_data:
            print(f"  Includes candidate images: {sample_data['candidate_images'].shape}")
        
        sample_data.close()
    
    return {
        'metadata': metadata,
        'summary': summary,
        'results_dir': results_dir
    }


def freeze_models(*models):
    """Freeze all parameters in the given models for inference."""
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

def run_diffusion(brain_embeds, img_ip, models_dict, num_predictions=1, noise_factor=4.0):
    """
    Run diffusion process with brain conditioning to generate images.
    
    Args:
        brain_embeds: Brain-derived conditioning embeddings
        img_ip: Initial image for latent space (set to zeros for unconditional)
        models_dict: Dictionary containing all necessary models
        num_predictions: Number of candidate images to generate
        noise_factor: Classifier-free guidance scale
        
    Returns:
        Generated images as numpy array
    """
    # Extract models from dictionary
    tokenizer = models_dict['tokenizer']
    text_encoder = models_dict['text_encoder']
    vae = models_dict['vae']
    noise_scheduler = models_dict['noise_scheduler']
    brain_adapter = models_dict['brain_adapter']
    device = models_dict['device']
    weight_dtype = models_dict['weight_dtype']
    
    # Process empty text prompt
    text = ""
    text_input_ids = tokenizer(
        text,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    
    encoder_hidden_states = text_encoder(text_input_ids)[0].to(device, dtype=weight_dtype)
    encoder_hidden_states = encoder_hidden_states.repeat(num_predictions, 1, 1)

    # Setup diffusion timesteps
    num_inference_steps = 50
    noise_scheduler.set_timesteps(num_inference_steps)
    init_timestep = min(int(num_inference_steps * 1.0), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start:]

    # Generate initial latents from input image
    init_latents = vae.encode(img_ip).latent_dist.sample() * vae.config.scaling_factor

    # Generate different noise for each prediction
    latents = []
    for i in range(num_predictions):
        noise = torch.randn(init_latents.shape).to(device, dtype=weight_dtype)
        noised_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
        latents.append(noised_latents)
    latents = torch.cat(latents, dim=0)

    # Diffusion denoising process
    for i, t in enumerate(tqdm(timesteps, desc="Diffusion steps", disable=True)):
        noise_pred_uncond, noise_pred_cond = brain_adapter(
            latents, t, encoder_hidden_states, brain_embeds.repeat(num_predictions, 1, 1)
        )
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + noise_factor * (noise_pred_cond - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to images
    latents = 1 / vae.config.scaling_factor * latents
    with torch.autocast(device_type=device.type):
        pred_image = vae.decode(latents).sample

    # Post-process images
    pred_image = (pred_image / 2 + 0.5).clamp(0, 1)
    pred_image = pred_image.cpu().permute(0, 2, 3, 1).numpy()
    pred_image = (pred_image * 255).round().astype("uint8")
    
    # Clean up intermediate tensors
    del latents, init_latents, encoder_hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_image


def compute_brain_correlation(ground_truth, predictions):
    """
    Compute Pearson correlation between ground truth and predicted brain activity.
    
    Args:
        ground_truth: Ground truth brain activity [1, voxels]
        predictions: Predicted brain activity [N, voxels]
        
    Returns:
        Correlation scores for each prediction [N]
    """
    # Standardize both ground truth and predictions with epsilon for numerical stability
    eps = 1e-8
    gt_std = ground_truth.std(dim=1, keepdim=True) + eps
    pred_std = predictions.std(dim=1, keepdim=True) + eps
    
    gt_centered = (ground_truth - ground_truth.mean(dim=1, keepdim=True)) / gt_std
    pred_centered = (predictions - predictions.mean(dim=1, keepdim=True)) / pred_std

    # Compute Pearson correlation
    correlations = torch.matmul(pred_centered, gt_centered.T).squeeze() / ground_truth.size(1)
    return correlations


def preprocess_image(image, target_size, device="cpu"):
    """Preprocess image to target size."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(image).to(device)


def process_brain_data_for_correlation(batch, test_dataset, brain_encoder_predictions, device):
    """
    Process brain data and predictions to compute correlation scores.
    
    Args:
        batch: Data batch containing brain activity
        test_dataset: Test dataset with parcel information
        brain_encoder_predictions: Predicted brain activity from encoder
        device: Device to use for computation
        
    Returns:
        Tuple of (ground_truth_concat, predictions_concat) ready for correlation
    """
    target_device = device
    
    # Concatenate brain data across all parcels and hemispheres
    ground_truth_concat = []
    predictions_concat = []
    
    for hemisphere in ["lh", "rh"]:
        for parcel_idx, parcel_id in enumerate(test_dataset.selected_parcel_idx[hemisphere]):
            voxel_indices = test_dataset.parcels[hemisphere][parcel_id]
            num_voxels = len(voxel_indices)
            
            # Ground truth brain activity for this parcel
            ground_truth_concat.append(
                batch[f"brain_{hemisphere}_f"][:, parcel_idx, :num_voxels].to(target_device)
            )
            
            # Predicted brain activity for this parcel
            predictions_concat.append(
                brain_encoder_predictions[hemisphere][:, voxel_indices].to(target_device)
            )
    
    # Concatenate all parcels
    ground_truth = torch.cat(ground_truth_concat, dim=1)
    predictions = torch.cat(predictions_concat, dim=1)
    
    assert ground_truth.shape[1] == predictions.shape[1], \
        f"Dimension mismatch: GT {ground_truth.shape[1]} vs Pred {predictions.shape[1]}"
    
    return ground_truth, predictions


def decode_images_from_brain(dataloader, models_dict, brain_encoder, test_dataset, num_predictions=8, noise_factor=4.0, save_all_candidates=False):
    """
    Main function to decode images from brain signals.
    
    Args:
        dataloader: DataLoader containing brain data
        models_dict: Dictionary with all necessary models
        brain_encoder: Brain encoder for validation
        test_dataset: Dataset instance for parcel information
        num_predictions: Number of candidate images per brain sample
        noise_factor: Guidance scale for diffusion
        save_all_candidates: Whether to save all candidate images (for subset evaluation)
        
    Returns:
        Tuple of (original_images, decoded_images, candidate_images_list, correlation_scores_list, dataset_indices)
    """
    original_images, decoded_images = [], []
    candidate_images_list, correlation_scores_list = [], []
    dataset_indices = []
    
    device = models_dict['device']
    weight_dtype = models_dict['weight_dtype']
    guidance_generator = models_dict['guidance_generator']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Decoding brain signals")):
            # Get original image
            img = batch["img_encoder"].squeeze()  # Remove batch dimension
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            original_images.append(img)
            
            # Get dataset index for this sample
            if hasattr(dataloader.dataset, 'indices'):
                # This is a Subset, get the original dataset index
                dataset_idx = dataloader.dataset.indices[batch_idx]
            else:
                dataset_idx = batch_idx
            dataset_indices.append(dataset_idx)
            
            # Process brain data
            lh = batch["brain_lh_f"].to(device, dtype=weight_dtype)
            rh = batch["brain_rh_f"].to(device, dtype=weight_dtype)
            brain_data = torch.cat([lh, rh], dim=1)
            
            # Validate brain data dimensions
            assert brain_data.shape[1] == test_dataset.num_parcels, \
                f"Expected {test_dataset.num_parcels} parcels, got {brain_data.shape[1]}"
            assert brain_data.shape[2] == test_dataset.max_voxels, \
                f"Expected {test_dataset.max_voxels} voxels, got {brain_data.shape[2]}"

            # Generate brain conditioning tokens
            brain_embeds, _ = guidance_generator(brain_data)

            # Use zero-filled image as initialization (no image information)
            img_init = batch["img_ipadapter"].to(device, dtype=weight_dtype)
            img_init = torch.zeros_like(img_init)

            # Generate candidate images using diffusion
            candidate_images = run_diffusion(
                brain_embeds, img_init, models_dict, 
                num_predictions=num_predictions, noise_factor=noise_factor
            )

            # Get brain encoder predictions for all candidates
            brain_predictions = brain_encoder.forward(candidate_images)

            # Compute correlations with ground truth brain activity
            ground_truth, predictions = process_brain_data_for_correlation(
                batch, test_dataset, brain_predictions, device
            )
            correlations = compute_brain_correlation(ground_truth, predictions)

            # Select best candidate based on highest correlation
            best_idx = torch.argmax(correlations).item()
            best_image = candidate_images[best_idx]
            
            # Preprocess to match original image size and move to CPU for consistency
            best_image = preprocess_image(best_image, img.shape[-1], "cpu")
            decoded_images.append(best_image)
            
            # Save candidate data if requested (for subset evaluation)
            if save_all_candidates:
                # Convert candidate images back to numpy for storage
                candidates_numpy = candidate_images
                correlations_numpy = correlations.cpu().numpy()
                
                candidate_images_list.append(candidates_numpy)
                correlation_scores_list.append(correlations_numpy)
            else:
                candidate_images_list.append(None)
                correlation_scores_list.append(None)
            
            # Clean up GPU memory after each sample to prevent accumulation
            del brain_predictions, ground_truth, predictions, correlations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Stack all images
    original_images = torch.stack(original_images, dim=0)
    decoded_images = torch.stack(decoded_images, dim=0)
    
    return original_images, decoded_images, candidate_images_list, correlation_scores_list, dataset_indices


def main():
    """Main execution function with enhanced evaluation options."""
    # Setup device
    device = setup_device()
    
    import time
    start_time = time.time()
    
    # Parse and validate arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    args = validate_arguments(args)
    
    print("=== Brain Adapter Decoding Configuration ===")
    print(f"Model weights: {args.model_weights_dir}")
    print(f"Checkpoint: {args.saved_epochs}")
    print(f"Evaluation mode: {'Full dataset' if args.eval_full_dataset else f'Indices {args.start_idx}-{args.end_idx}'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'No limit'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Noise factor: {args.noise_factor}")
    print(f"Predictions per sample: {args.num_predictions}")
    print(f"Top-k parcels: {args.topk}")
    print("=" * 50)

    
    # Load pre-trained models
    print("Loading diffusion models...")
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models()
    
    # Setup IP-Adapter components
    print("Setting up IP-Adapter...")
    image_proj_model, adapter_modules = setup_ip_adapter_modules(args, unet, num_tokens=args.topk*2)
    
    # Create test dataset with flexible evaluation
    print("Creating test dataset...")
    test_dataset, test_dataloader, evaluation_indices = create_test_dataset(
        args,
        tokenizer, 
        topk=args.topk, 
        eval_full_dataset=args.eval_full_dataset,
        start_idx=args.start_idx, 
        end_idx=args.end_idx,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Estimate memory usage
    num_samples = evaluation_indices[1] - evaluation_indices[0]
    estimate_memory_usage(num_samples, args.num_predictions)
    
    # Create brain adapter and guidance generator
    print("Initializing brain models...")
    brain_adapter = BrainIPAdapter(unet, image_proj_model, adapter_modules, ckpt_path=None)
    guidance_generator = GuidanceGenerator(
        num_parcels=test_dataset.num_parcels,
        max_voxels=test_dataset.max_voxels,
        num_decoder_queries=args.num_decoder_queries,
        output_dim=args.conditioning_dim,
        sub_approach=args.sub_approach,
    )

    # Load trained model weights
    print(f"Loading trained weights from checkpoint-{args.saved_epochs}...")
    try:
        brain_adapter.load_state_dict(torch.load(
            f"{args.model_weights_dir}/checkpoint-{args.saved_epochs}/pytorch_model.bin",
            map_location="cpu"
        ))
        guidance_generator.load_state_dict(torch.load(
            f"{args.model_weights_dir}/checkpoint-{args.saved_epochs}/pytorch_model_1.bin", 
            map_location="cpu"
        ))
        print("Model weights loaded successfully")
    except FileNotFoundError as e:
        print(f"Error: Could not load model weights: {e}")
        return 1

    # Set up weight dtype and move models to device
    weight_dtype = torch.float32
    
    # Move models to device
    print("Moving models to device...")
    brain_adapter = brain_adapter.to(device)
    guidance_generator = guidance_generator.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    image_proj_model = image_proj_model.to(device)

    print(f"Models moved to device: {device}")
    print(f"Target weight dtype: {weight_dtype}")
    print("Models prepared successfully")

    # Load brain encoder for evaluation
    print("Loading brain encoder...")
    original_cwd = os.getcwd()
    try:
        os.chdir("/engram/nklab/pf2477/brain_decoding/whole_brain_encoder")
        brain_encoder = BrainEncoderWrapper(num_gpus=args.num_gpus, subj=args.subject_id)
        print("Brain encoder loaded successfully")
    except Exception as e:
        print(f"Error loading brain encoder: {e}")
        return 1
    finally:
        os.chdir(original_cwd)

    # Freeze all models for inference
    print("Freezing models for inference...")
    freeze_models(unet, vae, text_encoder, image_proj_model, adapter_modules, brain_adapter, guidance_generator)

    # Create models dictionary for easy passing
    models_dict = {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'noise_scheduler': noise_scheduler,
        'brain_adapter': brain_adapter,
        'guidance_generator': guidance_generator,
        'device': device,
        'weight_dtype': weight_dtype
    }

    # Run decoding
    print(f"\nStarting image decoding from brain signals...")
    print(f"Processing {num_samples} samples with {args.num_predictions} predictions each...")
    
    # Determine whether to save all candidates
    save_candidates = args.save_all_candidates and not args.eval_full_dataset
    if args.save_all_candidates and args.eval_full_dataset:
        print("Warning: --save_all_candidates ignored for full dataset evaluation due to storage constraints")

    try:
        original_images, decoded_images, candidate_images_list, correlation_scores_list, dataset_indices = decode_images_from_brain(
            test_dataloader, models_dict, brain_encoder, test_dataset,
            num_predictions=args.num_predictions, noise_factor=args.noise_factor,
            save_all_candidates=save_candidates
        )
        print("Decoding completed successfully")
    except Exception as e:
        print(f"Error during decoding: {e}")
        return 1
    finally:
        cleanup_gpu_memory()

    # Calculate processing time
    processing_time = time.time() - start_time
    print(f"Total processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
    print(f"Time per sample: {processing_time/num_samples:.1f} seconds")

    # Save results with new organized structure
    brain_decoding_dir = "/engram/nklab/pf2477/brain_decoding/"
    if not os.path.exists(brain_decoding_dir):
        brain_decoding_dir = os.getcwd()
    
    os.chdir(brain_decoding_dir)
    base_save_dir = os.path.join(args.decoded_stimuli_dir, args.model_weights_dir.split("/")[-1])
    
    # Save individual results with organized folder structure
    results_dir = save_individual_results(
        original_images, decoded_images, candidate_images_list, correlation_scores_list,
        base_save_dir, args, evaluation_indices, dataset_indices, processing_time
    )

    print(f"\nDecoding completed successfully!")
    print(f"Results directory: {results_dir}")
    print(f"Processed {len(original_images)} samples in {processing_time/60:.1f} minutes")
    print(f"Individual files saved for each sample")

    return 0
    
    
def create_argument_parser():
    """Create and configure the argument parser for decoding."""
    parser = argparse.ArgumentParser(
        description="Decode images from brain signals using trained Brain Adapter model"
    )
    
    parser.add_argument(
        "--model_weights_dir", 
        type=str, 
        default="brain_adapter/model_weights/06_27_2025-21_15",
        help="Directory containing trained model weights"
    )
    parser.add_argument(
        "--decoded_stimuli_dir", 
        type=str, 
        default="./brain_adapter/decoded_stimuli",
        help="Directory to save decoded images"
    )
    parser.add_argument(
        "--saved_epochs", 
        type=str, 
        default="100",
        help="Training step checkpoint to load"
    )
    parser.add_argument(
        "--num_predictions", 
        type=int, 
        default=8,
        help="Number of candidate images to generate per brain sample"
    )
    
    # Evaluation mode selection
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval_full_dataset",
        action="store_true",
        help="Evaluate on the entire test dataset (ignores start_idx/end_idx)"
    )
    eval_group.add_argument(
        "--eval_indices",
        action="store_true", 
        default=True,
        help="Evaluate on specific indices (default behavior)"
    )
    
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting index for test samples (only used with --eval_indices)"
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=8,
        help="Ending index for test samples (only used with --eval_indices)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (useful for full dataset evaluation)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = no limit)"
    )
    parser.add_argument(
        "--noise_factor", 
        type=float, 
        default=4.0,
        help="Classifier-free guidance scale for diffusion"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for decoding"
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
        help="Number of top parcels to use per hemisphere"
    )
    parser.add_argument(
        "--num_decoder_queries",
        type=int,
        default=50,
        help="Number of decoder queries for guidance generator"
    )
    parser.add_argument(
        "--conditioning_dim",
        type=int,
        default=768,
        help="Conditioning dimension for guidance generator"
    )
    parser.add_argument(
        "--sub_approach", 
        type=str, 
        default="linear_projection", 
        help="Sub-approach for training. Options: 'linear_projection', 'masking + transformer_decoder'"
    )
    parser.add_argument(
        "--save_all_candidates",
        action="store_true",
        default=False,
        help="Save all candidate images for subset evaluation (not recommended for full dataset due to storage)"
    )

    return parser


if __name__ == "__main__":
    """
    Usage Examples:
    
    # 1. Evaluate specific indices (saves to subset/ folder):
    python decode_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/07_26_2025-22_29 \
        --saved_epochs 100 \
        --start_idx 0 \
        --end_idx 8 \
        --num_predictions 8 \
        --subject_id 1 \
        --topk 100 \
        --num_decoder_queries 50 \
        --conditioning_dim 768 \
        --sub_approach transformer_decoder
    
    # 2. Evaluate subset with all candidates saved:
    python decode_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/07_28_2025-01_26 \
        --saved_epochs 70 \
        --start_idx 0 \
        --end_idx 16 \
        --num_predictions 4 \
        --save_all_candidates \
        --subject_id 1 \
        --topk 50 \
        --num_decoder_queries 50 \
        --conditioning_dim 192 \
        --sub_approach linear_projection
    
    # 3. Evaluate entire test dataset (saves to full/ folder):
    python decode_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/07_26_2025-22_29 \
        --saved_epochs 100 \
        --eval_full_dataset \
        --batch_size 1 \
        --num_predictions 2 \
        --subject_id 1 \
        --topk 100 \
        --num_decoder_queries 50 \
        --conditioning_dim 768 \
        --sub_approach transformer_decoder
    
    # Output Structure:
    # decoded_stimuli/
    # └── model_name/
    #     └── epoch_100/
    #         ├── full/           # Full dataset evaluation
    #         │   ├── sample_000001.npz
    #         │   ├── sample_000002.npz
    #         │   ├── ...
    #         │   ├── evaluation_metadata.json
    #         │   └── sample_summary.json
    #         └── subset/         # Subset evaluation
    #             ├── sample_000001.npz  (may include candidate_images if --save_all_candidates)
    #             ├── sample_000002.npz
    #             ├── ...
    #             ├── evaluation_metadata.json
    #             └── sample_summary.json
    """
    import sys
    exit_code = main()
    sys.exit(exit_code)