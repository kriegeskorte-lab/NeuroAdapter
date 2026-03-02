"""
Brain Adapter Attention Extraction Script

This script extracts attention maps from a trained brain adapter model during diffusion inference.
It loads the trained model weights and saves attention weights and intermediate images for each timestep.

The extraction pipeline:
1. Loads trained brain adapter and guidance generator models
2. Processes fMRI brain data to generate conditioning tokens
3. Runs diffusion with attention extraction enabled
4. Saves attention maps and intermediate images for analysis
"""

# Standard library imports
import os
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
from brain_adapter.dataset import nsd_topk_parcel_dataset, nsd_groupwise_topk_parcel_dataset

warnings.filterwarnings("ignore")

class NeuroAdapter(torch.nn.Module):
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
            
        self.cross_attn_maps = {}  # Store cross-attention outputs by timestep
        self.current_timestep = None
        self.extract_attention = False  # Flag to control attention extraction
        self._register_block_hook()
        
    def _register_block_hook(self):
        """Register hooks to capture cross-attention maps from all IP-Adapter layers."""
        def hook_fn(name):
            def forward_hook(module, input, output):
                if self.extract_attention and hasattr(module.processor, 'ip_attn_map'):
                    # Store IP-Adapter attention weights with minimal memory footprint
                    if module.processor.ip_attn_map is not None:
                        # Store only the essential attention map data
                        attn_map = module.processor.ip_attn_map.detach().cpu()
                        if attn_map.shape[0] == 2:
                            attn_map = attn_map[1] # Select the conditional attention map
                        self.cross_attn_maps[name] = attn_map
                        # Clear the processor's attention map to save memory
                        module.processor.ip_attn_map = None
            return forward_hook

        # Register hooks for all cross-attention layers with IP-Adapter processors
        self.layer_names = []
        for name, module in self.unet.named_modules():
            # Check if module has processor attribute and if it's an IP-Adapter processor
            if hasattr(module, 'processor') and isinstance(module.processor, IPAttnProcessor):
                handle = module.register_forward_hook(hook_fn(name))
                self.layer_names.append(name)
                print(f"Registered attention hook for: {name}")
        
        print(f"Total IP-Adapter layers registered: {len(self.layer_names)}")
    
    def clear_attention_maps(self):
        """Clear stored attention maps to free memory."""
        self.cross_attn_maps.clear()

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
        # Clear attention maps before each forward pass to save memory
        if self.extract_attention:
            self.clear_attention_maps()
        
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
        clip_embeddings_dim=args.condition_dim,
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


def create_dataset(args, tokenizer):
    """
    Create dataset for attention extraction.
    
    Args:
        args: Command line arguments
        tokenizer: CLIP tokenizer for text processing
        
    Returns:
        Tuple of (dataset, dataloader, indices)
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

    if args.multi_subject_training:
        dataset = nsd_groupwise_topk_parcel_dataset(
            dataset_args, split="test", transform=None, topk=args.topk, test_subj=args.subject_id)
    else:
        dataset = nsd_topk_parcel_dataset(dataset_args, split="test", transform=None, topk=args.topk)
    
    total_samples = len(dataset)
    print(f"Dataset created with {total_samples} total samples, using top {args.topk} parcels per hemisphere.")
    
    # Determine evaluation indices based on selected_idx or start_idx/end_idx
    if args.selected_idx is not None:
        # Use specific indices provided by user
        indices = args.selected_idx
        # Validate that all selected indices are within dataset bounds
        invalid_indices = [idx for idx in indices if idx >= total_samples or idx < 0]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}: must be between 0 and {total_samples-1}")
        print(f"Using selected indices: {indices}")
    else:
        # Use range-based selection (original behavior)
        end_idx = min(args.end_idx, total_samples)
        if args.start_idx >= total_samples:
            raise ValueError(f"start_idx ({args.start_idx}) exceeds dataset size ({total_samples})")
        if args.start_idx >= end_idx:
            raise ValueError(f"start_idx ({args.start_idx}) must be less than end_idx ({end_idx})")
            
        indices = list(range(args.start_idx, end_idx))
        print(f"Using range-based selection: indices {args.start_idx} to {end_idx-1}")
    
    # Apply max_samples limit if specified
    if args.max_samples is not None and len(indices) > args.max_samples:
        indices = indices[:args.max_samples]
        print(f"Limited to first {args.max_samples} samples")
    
    print(f"Processing {len(indices)} samples: {indices}")

    # Create subset and dataloader (batch_size=1 for attention extraction)
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, shuffle=False, batch_size=1, num_workers=0)
    
    return dataset, dataloader, indices


def setup_device():
    """Initialize device for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def validate_arguments(args):
    """Validate and process command line arguments."""
    # Print extraction mode
    if args.selected_idx is not None:
        print(f"Mode: Selected indices attention extraction {args.selected_idx}")
    else:
        print(f"Mode: Range-based attention extraction ({args.start_idx} to {args.end_idx})")
    if args.max_samples:
        print(f"  - Limited to first {args.max_samples} samples")
        
    # Validate numerical arguments
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


def freeze_models(*models):
    """Freeze all parameters in the given models for inference."""
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


def save_attention_maps(attention_maps, timestep_data, save_dir, sample_idx, timestep, layer_names, save_images=False):
    """
    Save attention maps and optionally timestep images for a specific sample and timestep.
    This function is optimized for memory efficiency and storage compression.
    
    Args:
        attention_maps: Dictionary of attention maps by layer name (current timestep only)
        timestep_data: Dictionary containing images for current timestep (optional)
        save_dir: Directory to save the attention data
        sample_idx: Sample index for filename
        timestep: Current timestep value
        layer_names: List of all layer names for reference
        save_images: Whether to save decoded images (memory intensive)
    """
    # Create sample-specific directory
    sample_dir = os.path.join(save_dir, f"sample_{sample_idx:06d}")
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data for this specific timestep with memory optimization
    timestep_save_data = {
        'timestep': np.int32(timestep),  # Use smaller int type
        'attention_maps': {},
    }
    
    # Process attention maps for this timestep with compression
    total_attn_size = 0
    for layer_name, attn_map in attention_maps.items():
        if attn_map is not None:
            attn_array = attn_map.numpy().mean(axis=0, keepdims=True)

            # Convert to float16 for storage efficiency (halves storage size)
            attn_array = attn_array.astype(np.float16)
            
            timestep_save_data['attention_maps'][layer_name] = attn_array
            total_attn_size += attn_array.nbytes
        else:
            timestep_save_data['attention_maps'][layer_name] = None

    # Save estimated images
    timestep_save_data['clean_predictions'] = timestep_data['clean_predictions']
    
    # Save to individual timestep file with maximum compression
    timestep_file = os.path.join(sample_dir, f"time_step_{int(timestep):03d}.npz")
    np.savez_compressed(timestep_file, **timestep_save_data)
    
    return timestep_file

def decode_latents_to_images(latents, vae, device):
    """Helper function to decode latents to images."""
    temp_latents = 1 / vae.config.scaling_factor * latents
    with torch.autocast(device_type=device.type):
        temp_images = vae.decode(temp_latents).sample
    # Post-process and store images
    temp_images = (temp_images / 2 + 0.5).clamp(0, 1)
    temp_images = temp_images.cpu().permute(0, 2, 3, 1).numpy()
    temp_images = (temp_images * 255).round().astype("uint8")
    return temp_images


def run_diffusion(brain_embeds, img_ip, models_dict, num_predictions=1, noise_factor=4.0, save_attention=False, attention_save_dir=None, sample_idx=None, save_freq=5):
    """
    Run diffusion process with brain conditioning to generate images.
    
    Args:
        brain_embeds: Brain-derived conditioning embeddings
        img_ip: Initial image for latent space (set to zeros for unconditional)
        models_dict: Dictionary containing all necessary models
        num_predictions: Number of candidate images to generate
        noise_factor: Classifier-free guidance scale
        save_attention: Whether to save attention maps
        attention_save_dir: Directory to save attention maps
        sample_idx: Sample index for naming attention files
        save_freq: Save attention maps and images every N timesteps
        
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
    
    # Enable attention extraction if saving is requested
    if save_attention:
        brain_adapter.extract_attention = True
        brain_adapter.clear_attention_maps()
    
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
    
    # Clean up text input ids immediately
    del text_input_ids
    torch.cuda.empty_cache()

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
        # Clean up noise tensor immediately
        del noise
    latents = torch.cat(latents, dim=0)

    # Diffusion denoising process
    diffusion_pbar = tqdm(timesteps, desc=f"Diffusion steps (sample {sample_idx})", leave=False, position=1)
    for i, t in enumerate(diffusion_pbar):
        # Set current timestep for attention hooks
        if save_attention:
            brain_adapter.current_timestep = t.item()
        
        noise_pred_uncond, noise_pred_cond = brain_adapter(
            latents, t, encoder_hidden_states, brain_embeds.repeat(num_predictions, 1, 1)
        )
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + noise_factor * (noise_pred_cond - noise_pred_uncond)
        
        # Clean up intermediate predictions immediately
        del noise_pred_uncond, noise_pred_cond
        
        # Store and save timestep data immediately if saving attention
        if save_attention and attention_save_dir is not None:
            # Save attention maps based on frequency + ensure we capture start/end
            save_this_timestep = (i % save_freq == 0) or (i < 5) or (i > len(timesteps) - 6)
            
            if save_this_timestep:
                # Get clean prediction using pred_original_sample (more memory efficient)
                step_output = noise_scheduler.step(noise_pred, t, latents, return_dict=True)
                
                # Save both attention maps and images at the same frequency
                pred_original_sample = step_output.pred_original_sample
                clean_images_decoded = decode_latents_to_images(pred_original_sample.clone(), vae, device)
                current_timestep_data = {'clean_predictions': clean_images_decoded}
                del pred_original_sample, clean_images_decoded
                
                # Save attention maps and images together
                save_attention_maps(
                    brain_adapter.cross_attn_maps,  # Current timestep attention maps only
                    current_timestep_data,
                    attention_save_dir,
                    sample_idx,
                    t.item(),
                    brain_adapter.layer_names,
                    save_images=True  # Always save images when saving attention
                )
                
                latents = step_output.prev_sample
                del step_output, current_timestep_data
            else:
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        else:
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Clean up noise prediction and clear attention maps for next timestep
        del noise_pred
        if save_attention:
            brain_adapter.clear_attention_maps()
        
        # More frequent memory cleanup during diffusion
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Close the diffusion progress bar
    diffusion_pbar.close()

    # Disable attention extraction after diffusion
    if save_attention:
        brain_adapter.extract_attention = False
        brain_adapter.clear_attention_maps()

    # Decode final latents to images
    latents = 1 / vae.config.scaling_factor * latents
    with torch.autocast(device_type=device.type):
        pred_image = vae.decode(latents).sample

    # Post-process images and move to CPU immediately
    pred_image = (pred_image / 2 + 0.5).clamp(0, 1)
    pred_image = pred_image.cpu().permute(0, 2, 3, 1).numpy()
    pred_image = (pred_image * 255).round().astype("uint8")
    
    # Clean up all intermediate tensors
    del latents, init_latents, encoder_hidden_states
    torch.cuda.empty_cache()

    return pred_image


def freeze_models(*models):
    """Freeze all parameters in the given models for inference."""
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


def save_layer_info(layer_names, save_dir, sample_idx):
    """
    Save comprehensive layer information for reference.
    
    Args:
        layer_names: List of all IP-Adapter layer names
        save_dir: Directory to save the layer info
        sample_idx: Sample index (used for first sample only)
    """
    if sample_idx == 0:  # Only save once for the first sample
        layer_info = {
            'total_layers': len(layer_names),
            'layer_names': layer_names,
            'layer_hierarchy': {},
            'block_counts': {}
        }
        
        # Analyze layer hierarchy
        for i, name in enumerate(layer_names):
            parts = name.split('.')
            block_type = parts[0] if len(parts) > 0 else 'unknown'
            
            if block_type not in layer_info['block_counts']:
                layer_info['block_counts'][block_type] = 0
            layer_info['block_counts'][block_type] += 1
            
            layer_info['layer_hierarchy'][i] = {
                'name': name,
                'block_type': block_type,
                'parts': parts
            }
        
        # Save layer info
        layer_info_file = os.path.join(save_dir, "layer_info.npz")
        np.savez_compressed(layer_info_file, **layer_info)
        
        print(f"Layer info saved: {len(layer_names)} total IP-Adapter layers")
        for block_type, count in layer_info['block_counts'].items():
            print(f"  {block_type}: {count} layers")


def extract_attention_maps(args, dataloader, models_dict, dataset, save_attention=True, attention_save_dir=None, save_freq=5):
    """
    Extract attention maps from brain signals during diffusion process.
    Optimized for memory efficiency and comprehensive attention capture.
    
    Args:
        dataloader: DataLoader containing brain data
        models_dict: Dictionary with all necessary models
        dataset: Dataset instance for parcel information
        save_attention: Whether to save attention maps during diffusion
        attention_save_dir: Directory to save attention maps
        save_freq: Save attention maps and images every N timesteps
        
    Returns:
        List of dataset indices processed
    """
    dataset_indices = []
    
    device = models_dict['device']
    weight_dtype = models_dict['weight_dtype']
    guidance_generator = models_dict['guidance_generator']
    brain_adapter = models_dict['brain_adapter']
    
    # Print attention extraction setup
    if save_attention:
        print(f"Attention extraction enabled:")
        print(f"  - Tracking {len(brain_adapter.layer_names)} IP-Adapter layers")
        print(f"  - Saving attention maps and images every {save_freq} timesteps + first/last few timesteps")
        print(f"  - Using float16 compression for attention maps")
    
    with torch.no_grad():
        # Create progress bar for the main extraction loop
        extraction_pbar = tqdm(dataloader, desc="Extracting attention maps", position=0)
        
        for batch_idx, batch in enumerate(extraction_pbar):
            # Get dataset index for this sample
            if hasattr(dataloader.dataset, 'indices'):
                dataset_idx = dataloader.dataset.indices[batch_idx]
            else:
                dataset_idx = batch_idx
            dataset_indices.append(dataset_idx)
            
            tqdm.write(f"\nProcessing sample {dataset_idx} ({batch_idx+1}/{len(dataloader)})")
            
            # Process brain data
            lh = batch["brain_lh_f"].to(device, dtype=weight_dtype)
            rh = batch["brain_rh_f"].to(device, dtype=weight_dtype)
            brain_data = torch.cat([lh, rh], dim=1)
            
            # Clean up individual hemisphere data
            del lh, rh
            
            # Validate brain data dimensions
            assert brain_data.shape[1] == dataset.num_parcels, \
                f"Expected {dataset.num_parcels} parcels, got {brain_data.shape[1]}"
            assert brain_data.shape[2] == dataset.max_voxels, \
                f"Expected {dataset.max_voxels} voxels, got {brain_data.shape[2]}"
            
            # Generate brain conditioning
            brain_conditioning, _ = guidance_generator(brain_data)
            
            # Create zero-filled image as initialization (no image information)
            device = models_dict['device']
            weight_dtype = models_dict['weight_dtype']
            # dummy_img = torch.zeros((1, 3, 512, 512), device=device, dtype=weight_dtype)
            dummy_img = batch["img_ipadapter"].to(device, dtype=weight_dtype)
            
            # Save layer information (only for first sample)
            if save_attention and attention_save_dir:
                save_layer_info(brain_adapter.layer_names, attention_save_dir, dataset_idx)
            
            # Run single diffusion pass to extract attention
            try:
                run_diffusion(
                    brain_conditioning, dummy_img, models_dict,
                    num_predictions=1,  # Only need one pass for attention
                    noise_factor=args.noise_factor,
                    save_attention=save_attention,
                    attention_save_dir=attention_save_dir,
                    sample_idx=dataset_idx,
                    save_freq=save_freq
                )
            except Exception as e:
                print(f"Error processing sample {dataset_idx}: {e}")
                continue
            
            # Clean up
            del brain_data, brain_conditioning, dummy_img
            torch.cuda.empty_cache()
        
        # Close the extraction progress bar
        extraction_pbar.close()
    
    return dataset_indices


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
    
    print("=== Brain Adapter Attention Extraction Configuration ===")
    print(f"Model weights: {args.model_weights_dir}")
    print(f"Checkpoint: {args.saved_epochs}")
    if args.selected_idx is not None:
        print(f"Selected indices: {args.selected_idx}")
    else:
        print(f"Sample range: Indices {args.start_idx}-{args.end_idx}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'No limit'}")
    print(f"Subject ID: {args.subject_id}")
    print(f"Multi-subject training: {args.multi_subject_training}")
    print(f"Noise factor: {args.noise_factor}")
    print(f"Top-k parcels: {args.topk}")
    print("=" * 60)

    # Force optimal settings for attention extraction
    args.batch_size = 1
    args.num_predictions = 1
    print("Attention extraction mode: using batch_size=1 and num_predictions=1")
    
    # Load pre-trained models
    print("Loading diffusion models...")
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models()
    
    # Setup IP-Adapter components
    print("Setting up IP-Adapter...")
    image_proj_model, adapter_modules = setup_ip_adapter_modules(args, unet, num_tokens=args.topk*2)
    
    # Create dataset for attention extraction
    print("Creating dataset...")
    dataset, dataloader, indices = create_dataset(args, tokenizer)
    
    # Estimate memory usage
    num_samples = len(indices)
    estimate_memory_usage(num_samples, args.num_predictions)
    
    # Create brain adapter and guidance generator
    print("Initializing brain models...")
    brain_adapter = NeuroAdapter(unet, image_proj_model, adapter_modules, ckpt_path=None)
    guidance_generator = GuidanceGenerator(
        num_parcels=dataset.num_parcels,
        max_voxels=dataset.max_voxels,
        num_decoder_queries=args.num_decoder_queries,
        output_dim=args.condition_dim,
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

    # Clear any cached memory after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        print(f"Initial GPU memory usage: {initial_memory:.2f}GB")

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

    # Set up attention extraction directory
    model_name = args.model_weights_dir.split("/")[-1]
    
    # For multi-subject training, add subject-specific subdirectory
    if args.multi_subject_training:
        attention_save_dir = os.path.join(
            args.output_dir, 
            model_name, 
            f"epoch_{args.saved_epochs}",
            "subset",
            str(args.subject_id)
        )
    else:
        attention_save_dir = os.path.join(
            args.output_dir, 
            model_name, 
            f"epoch_{args.saved_epochs}",
            "subset"
        )
    
    Path(attention_save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Attention maps will be saved to: {attention_save_dir}")

    # Run attention extraction
    print(f"\nStarting attention extraction from brain signals...")
    print(f"Processing {num_samples} samples")
    print("Attention extraction enabled - saving attention maps for each timestep")
    
    # Extract attention maps
    try:
        dataset_indices = extract_attention_maps(
            args,
            dataloader, models_dict, dataset,
            save_attention=True,
            attention_save_dir=attention_save_dir,
            save_freq=args.save_frequency
        )
        print("Attention extraction completed successfully")
    except Exception as e:
        print(f"Error during attention extraction: {e}")
        return 1
    finally:
        cleanup_gpu_memory()

    # Calculate processing time
    processing_time = time.time() - start_time
    print(f"Total processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
    print(f"Time per sample: {processing_time/num_samples:.1f} seconds")

    print(f"\nAttention extraction completed successfully!")
    print(f"Attention maps saved to: {attention_save_dir}")
    print(f"Processed {len(dataset_indices)} samples in {processing_time/60:.1f} minutes")

    return 0
    
    
def create_argument_parser():
    """Create and configure the argument parser for attention extraction."""
    parser = argparse.ArgumentParser(
        description="Extract attention maps from brain signals using trained Brain Adapter model"
    )
    
    # Model and checkpoint arguments
    parser.add_argument(
        "--model_weights_dir", 
        type=str, 
        required=True,
        help="Directory containing trained model weights"
    )
    parser.add_argument(
        "--saved_epochs", 
        type=str, 
        required=True,
        help="Training checkpoint epoch to load"
    )
    
    # Data and subject arguments
    parser.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Subject ID for fMRI data (1-8)"
    )
    parser.add_argument(
        "--multi_subject_training",
        action="store_true",
        help="Use multi-subject trained model"
    )
    
    # Sample selection arguments
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting sample index (ignored if --selected_idx is used)"
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=5,
        help="Ending sample index (ignored if --selected_idx is used)"
    )
    parser.add_argument(
        "--selected_idx",
        type=int,
        nargs='+',
        default=None,
        help="Specific sample indices to process (e.g., --selected_idx 1 3 5 6). Overrides start_idx/end_idx"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = no limit)"
    )
    
    # Model architecture arguments
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
        "--condition_dim",
        type=int,
        default=768,
        help="Conditioning dimension for guidance generator"
    )
    parser.add_argument(
        "--sub_approach", 
        type=str, 
        default="linear_projection", 
        help="Sub-approach for model. Options: 'linear_projection', 'transformer_decoder'"
    )
    
    # Diffusion parameters
    parser.add_argument(
        "--noise_factor", 
        type=float, 
        default=2.0,
        help="Classifier-free guidance scale for diffusion"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion denoising steps"
    )
    
    # Output and storage options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="brain_adapter/attn_maps",
        help="Base directory to save attention maps"
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=5,
        help="Save attention maps and images every N timesteps (lower = more storage, higher = less detail)"
    )
    parser.add_argument(
        "--compress_attention",
        action="store_true",
        default=True,
        help="Use float16 compression for attention maps (halves storage)"
    )

    return parser


if __name__ == "__main__":
    main()
    """
    Usage Examples:
    
    # 1. Extract attention maps for specific selected indices:
    python extract_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/08_16_2025-18_03 \
        --saved_epochs 200 \
        --selected_idx 105 6 484 432 \
        --subject_id 1 \
        --noise_factor 4.0 \
        --topk 100 \
        --condition_dim 768 \
        --sub_approach linear_projection \
        --save_frequency 1
    
    # 2. Extract attention maps for range-based selection (original behavior):
    python extract_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/08_16_2025-18_03 \
        --saved_epochs 200 \
        --start_idx 6 \
        --end_idx 7 \
        --subject_id 1 \
        --noise_factor 4.0 \
        --topk 100 \
        --num_decoder_queries 50 \
        --condition_dim 768 \
        --sub_approach linear_projection \
        --save_frequency 5
    
    # 3. Extract attention maps for larger sample range:
    python extract_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/08_10_2025-15_50 \
        --saved_epochs 100 \
        --start_idx 0 \
        --end_idx 20 \
        --max_samples 10 \
        --subject_id 1 \
        --noise_factor 2.0 \
        --topk 100 \
        --num_decoder_queries 50 \
        --condition_dim 768 \
        --sub_approach transformer_decoder
    
    # 4. Multi-subject training model attention extraction with selected indices:
    python extract_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/08_10_2025-15_50 \
        --saved_epochs 100 \
        --multi_subject_training \
        --selected_idx 0 2 4 6 \
        --subject_id 1 \
        --noise_factor 2.0 \
        --topk 100 \
        --num_decoder_queries 50 \
        --condition_dim 768 \
        --sub_approach linear_projection
    
    # 5. Different noise factor for attention extraction:
    python extract_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/08_03_2025-17_39 \
        --saved_epochs 200 \
        --start_idx 0 \
        --end_idx 8 \
        --noise_factor 3.0 \
        --subject_id 2 \
        --topk 100 \
        --num_decoder_queries 50 \
        --condition_dim 768 \
        --sub_approach linear_projection
    
    # Output Structure:
    # Attention extraction output:
    # brain_adapter/attn_maps/  (or custom output directory)
    # └── model_name/
    #     └── epoch_100/
    #         └── subset/
    #             ├── sample_000000/
    #             │   ├── time_step_999.npz  # Each timestep saved separately
    #             │   ├── time_step_979.npz
    #             │   ├── time_step_959.npz
    #             │   └── ... (50 timestep files total)
    #             ├── sample_000001/
    #             │   ├── time_step_999.npz
    #             │   └── ... (50 timestep files)
    #             └── ...
    #
    # Multi-subject training models:
    # brain_adapter/attn_maps/
    # └── model_name/
    #     └── epoch_100/
    #         └── subset/
    #             └── 1/              # Subject-specific subdirectory
    #                 ├── sample_000000/
    #                 │   ├── time_step_999.npz
    #                 │   ├── time_step_979.npz
    #                 │   └── ... (50 files)
    #                 ├── sample_000001/
    #                 │   ├── time_step_999.npz
    #                 │   └── ... (50 files)
    #                 └── ...
    #
    # Each time_step_X.npz file contains:
    # - 'timestep': Single timestep value (int)
    # - 'attention_maps': Dictionary with layer_name -> {'ip_attn_map'}
    # - 'noisy_images': Noisy image array during denoising for this timestep
    # - 'clean_predictions': Clean prediction array for this timestep
    # - 'layer_names': List of all attention layer names captured
    # - 'sample_index': Original dataset sample index for reference
    #
    # Memory Efficiency: Each timestep is saved immediately during diffusion
    # to minimize memory usage. Attention maps are cleared after each step.
    
    
python -c "
import numpy as np
import glob
import os

# Set your sample directory path here
sample_dir = 'brain_adapter/attn_maps/08_10_2025-15_50/epoch_100/subset/1/sample_000000'

# Get all timestep files, sorted by timestep value (high to low)
files = glob.glob(os.path.join(sample_dir, 'time_step_*.npz'))
files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

print(f'Sample: {os.path.basename(sample_dir)}')
print(f'Found {len(files)} timestep files')

if not files:
    print('No files found! Check the path.')
    exit()

# Show available timesteps
timesteps = [int(f.split('_')[-1].split('.')[0]) for f in files]
print(f'Timesteps: {max(timesteps)} → {min(timesteps)}')

# Quick analysis function
def analyze(file_path):
    data = np.load(file_path, allow_pickle=True)
    
    # Basic info
    timestep = data['timestep']
    file_size = os.path.getsize(file_path) / (1024**2)
    print(f'\\n--- Timestep {timestep} ({file_size:.1f}MB) ---')
    
    # Attention maps
    attn = data['attention_maps'].item()
    print(f'Layers: {list(attn.keys())}')
    
    if attn:
        for i in range(len(attn)):
            a = list(attn.values())[i]['ip_attn_map']
            if a is not None:
                print(f'Layer: {list(attn.keys())[i]}')
                print(f'Attention shape: {a.shape}')
                print(f'Attention range: [{a.min():.3f}, {a.max():.3f}]')
    # Images
    noisy = data['noisy_images']
    clean = data['clean_predictions']
    if noisy is not None:
        print(f'Images: {noisy.shape}, range [{noisy.min()}-{noisy.max()}]')
    if clean is not None:
        print(f'Images: {clean.shape}, range [{clean.min()}-{clean.max()}]')

    data.close()

# Analyze key timesteps
print('\\n' + '='*50)
analyze(files[0])      # First (highest noise)
analyze(files[-1])     # Last (lowest noise)
if len(files) > 10:
    analyze(files[len(files)//2])  # Middle

# Sample summary
total_size = sum(os.path.getsize(f) for f in files) / (1024**2)
print(f'\\nTotal: {total_size:.1f}MB, Avg: {total_size/len(files):.1f}MB per file')

"""