#!/usr/bin/env python3
"""
Brain Adapter Cross-Attention Timeline Visualization

This script creates comprehensive visualizations of the denoising process across timesteps,
showing synchronized views of:
1. Ground truth image (constant reference)
2. Noisy images at each timestep
3. Clean predictions at each timestep  
4. Brain attention patterns mapped onto cortical surface using Pycortex

The script processes attention maps saved during diffusion inference and creates
animated GIFs showing the denoising progression with brain guidance.

Usage:
    python plot_cross_attention.py --sample_id 0 --model_dir 08_14_2025-00_21 \
                                   --saved_epochs 200 --max_timesteps 20

Author: Brain Adapter Team
Date: August 2025
"""

import os
import argparse
import glob
import re
from pathlib import Path
from types import SimpleNamespace
import warnings
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cortex
import imageio.v2 as imageio

# Suppress warnings and output for cleaner execution
@contextmanager
def suppress_print():
    """Silence stdout/stderr and warnings inside the with-block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, e = io.StringIO(), io.StringIO()
        with redirect_stdout(f), redirect_stderr(e):
            yield

# Set working directory
os.chdir("/engram/nklab/pf2477/brain_decoding/")

# Import dataset utilities
from brain_adapter.dataset import nsd_topk_parcel_dataset, get_dominant_roi_per_parcel


def setup_cortex_inkscape():
    """Setup robust Inkscape version detection for Pycortex."""
    import shutil, subprocess as sp
    import re
    
    def robust_inkscape_version():
        env = os.environ.copy()
        env.setdefault('LANG','C.UTF-8')
        env.setdefault('LC_ALL','C.UTF-8')
        p = sp.run(['inkscape','--version'], stdout=sp.PIPE, stderr=sp.STDOUT, text=True, env=env)
        m = re.search(r'Inkscape\s+([0-9]+(?:\.[0-9]+)*)', p.stdout or "")
        return m.group(1) if m else '0'   # fallback minimal

    import cortex.svgoverlay as svo
    svo.INKSCAPE_VERSION = robust_inkscape_version()
    print(f"Patched INKSCAPE_VERSION = {svo.INKSCAPE_VERSION}")


def setup_brain_dataset(subject_id=1, topk=100):
    """
    Setup brain dataset and metadata for the given subject.
    
    Args:
        subject_id: Subject ID (1-8)
        topk: Top-k parcels per hemisphere
        
    Returns:
        Tuple of (train_dataset, metadata, dominant_rois, all_roi_names)
    """
    print(f"Setting up brain dataset for subject {subject_id}")
    
    # Create dataset arguments
    args = SimpleNamespace(
        subj=subject_id,
        backbone_arch="dinov2_q",
        data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
        imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
        parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
        hemi=None,
        tokenizer=None,
        gen_size=512,
        topk=topk
    )
    
    # Create dataset
    test_dataset = nsd_topk_parcel_dataset(args, split='test', transform=None, topk=args.topk)
    
    # Load metadata
    neural_data_path = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )
    metadata = np.load(
        neural_data_path / f"metadata_sub-{subject_id:02}.npy", allow_pickle=True
    ).item()
    
    # Get ROI names
    all_roi_names = list(metadata['lh_rois'].keys())[:24] + list(metadata['rh_rois'].keys())[:24]
    
    # Calculate dominant ROIs per parcel
    min_overlap_threshold = 0.5
    dominant_rois = get_dominant_roi_per_parcel(
        test_dataset, metadata, all_roi_names, min_overlap_threshold=min_overlap_threshold
    )
    
    print(f"Dataset setup complete:")
    print(f"  Total samples: {len(test_dataset)}")
    print(f"  LH parcels: {len(test_dataset.selected_parcel_idx['lh'])}")
    print(f"  RH parcels: {len(test_dataset.selected_parcel_idx['rh'])}")
    print(f"  ROI names: {len(all_roi_names)}")
    
    return test_dataset, metadata, dominant_rois, all_roi_names


def create_cortex_attention_frame(
    attn_map, test_dataset, metadata, dominant_rois,
    spatial_idx=None, head_idx=None, subject="fsaverage",
    vmin=None, vmax=None, normalize_roi_weights=False
):
    """
    Create a single cortex attention frame for timeline visualization.
    
    Args:
        attn_map: Attention tensor [num_heads, spatial_tokens, condition_tokens]
        test_dataset: Dataset with parcel information
        metadata: Brain metadata with ROI mappings
        dominant_rois: ROI assignments per parcel
        spatial_idx: Which spatial token to visualize (None = average all)
        head_idx: Which attention head to visualize (None = average all)
        subject: Pycortex subject name
        vmin, vmax: Global min/max for consistent scaling
        normalize_roi_weights: Whether to renormalize weights for only valid ROI parcels
        
    Returns:
        Dictionary with frame data and statistics
    """
    
    # Handle attention map dimensions
    num_heads, spatial_tokens, condition_tokens = attn_map.shape
    
    # Select which spatial token and head to visualize
    if spatial_idx is None:
        attn_weights = attn_map.mean(axis=1)  # [heads, tokens]
    else:
        attn_weights = attn_map[:, spatial_idx, :]  # [heads, tokens]
    
    if head_idx is None:
        attn_weights = attn_weights.mean(axis=0)  # [tokens]
    else:
        attn_weights = attn_weights[head_idx, :]  # [tokens]
    
    # Get hemisphere sizes
    lh_hemi_size = len(next(iter(metadata['lh_rois'].values())))
    rh_hemi_size = len(next(iter(metadata['rh_rois'].values())))
    
    # Initialize cortical surface with zeros
    lh_values = np.zeros(lh_hemi_size, dtype=np.float32)
    rh_values = np.zeros(rh_hemi_size, dtype=np.float32)
    
    # Get selected voxel indices for each parcel
    sel_vox = test_dataset.get_selected_voxel_indices()
    
    # Collect weights for valid ROI parcels
    valid_roi_weights = []
    roi_token_indices = []
    
    for hemi in ["lh", "rh"]:
        for parcel_pos, (parcel_idx, roi_name, overlap, nvox) in enumerate(dominant_rois[hemi]):
            if roi_name is not None:
                if hemi == "lh":
                    token_idx = parcel_pos
                else:
                    token_idx = parcel_pos + 100
                
                if token_idx < len(attn_weights):
                    valid_roi_weights.append(attn_weights[token_idx])
                    roi_token_indices.append(token_idx)
    
    valid_roi_weights = np.array(valid_roi_weights)
    
    # Normalize weights for valid ROI parcels only (optional)
    if normalize_roi_weights and len(valid_roi_weights) > 0:
        weight_sum = valid_roi_weights.sum()
        if weight_sum > 0:
            valid_roi_weights = valid_roi_weights / weight_sum
    
    # Create mapping from token index to (optionally normalized) weight
    token_to_weight = dict(zip(roi_token_indices, valid_roi_weights))
    
    # Paint attention weights onto cortical surface
    for hemi, values in [("lh", lh_values), ("rh", rh_values)]:
        for parcel_pos, (parcel_idx, roi_name, overlap, nvox) in enumerate(dominant_rois[hemi]):
            if roi_name is None:
                continue
            
            vox_indices = sel_vox[hemi][parcel_pos]
            vox_indices = vox_indices.cpu().numpy() if isinstance(vox_indices, torch.Tensor) else np.asarray(vox_indices)
            
            if vox_indices.size == 0:
                continue
            
            if hemi == "lh":
                token_idx = parcel_pos
            else:
                token_idx = parcel_pos + 100
            
            if token_idx in token_to_weight:
                weight = token_to_weight[token_idx]
                values[vox_indices] = weight
    
    # Combine hemispheres for Pycortex
    cortex_data = np.concatenate([lh_values, rh_values])
    
    # Create custom colormap: grey for unlabeled, viridis for labeled
    viridis = plt.cm.viridis
    n_colors = 256
    
    colors = np.zeros((n_colors + 1, 4))
    colors[0] = [0.5, 0.5, 0.5, 1.0]  # Grey for unlabeled
    colors[1:] = viridis(np.linspace(0, 1, n_colors))
    
    custom_cmap = ListedColormap(colors)
    
    # Adjust data for custom colormap
    cortex_data_adjusted = cortex_data.copy()
    non_zero_mask = cortex_data > 0
    
    if np.any(non_zero_mask):
        if vmin is not None and vmax is not None and vmax > vmin:
            # Use global scaling
            cortex_data_adjusted[non_zero_mask] = 1 + (n_colors - 1) * np.clip(
                (cortex_data[non_zero_mask] - vmin) / (vmax - vmin), 0, 1
            )
        else:
            # Local scaling
            min_val = cortex_data[non_zero_mask].min()
            max_val = cortex_data[non_zero_mask].max()
            if max_val > min_val:
                cortex_data_adjusted[non_zero_mask] = 1 + (n_colors - 1) * (cortex_data[non_zero_mask] - min_val) / (max_val - min_val)
            else:
                cortex_data_adjusted[non_zero_mask] = n_colors // 2
    
    # Create Pycortex vertex and render
    vertex_data = cortex.Vertex(
        cortex_data_adjusted, 
        subject=subject, 
        cmap=custom_cmap,
        vmin=0,
        vmax=n_colors
    )
    
    # Render to frame
    with suppress_print():
        fig = cortex.quickflat.make_figure(vertex_data, with_curvature=True, with_colorbar=False)
    
    fig.canvas.draw()
    # Use buffer_rgba instead of deprecated tostring_rgb
    try:
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, :3]  # Remove alpha channel
    except AttributeError:
        # Fallback for older matplotlib versions
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Calculate statistics
    stats = {
        'num_valid_rois': len(valid_roi_weights),
        'total_tokens': len(attn_weights),
        'attention_range': (cortex_data.min(), cortex_data.max()),
        'mean_attention': cortex_data[cortex_data > 0].mean() if np.any(cortex_data > 0) else 0.0
    }
    
    return {'frame': frame, 'stats': stats}


def resize_image_array(img, target_size):
    """Simple image resizing using PIL or numpy fallback."""
    try:
        from PIL import Image
        # Convert numpy array to PIL Image
        if len(img.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(img, mode='L')
        else:  # RGB
            pil_img = Image.fromarray(img, mode='RGB')
        
        # Resize using PIL
        resized_pil = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
        
        # Convert back to numpy array
        return np.array(resized_pil)
    
    except ImportError:
        # Fallback to simple numpy-based resizing
        print("PIL not available, using simple numpy resizing")
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Simple nearest neighbor resizing
        y_ratio = h / target_h
        x_ratio = w / target_w
        
        if len(img.shape) == 2:  # Grayscale
            resized = np.zeros((target_h, target_w), dtype=img.dtype)
            for i in range(target_h):
                for j in range(target_w):
                    resized[i, j] = img[int(i * y_ratio), int(j * x_ratio)]
        else:  # RGB
            resized = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            for i in range(target_h):
                for j in range(target_w):
                    resized[i, j] = img[int(i * y_ratio), int(j * x_ratio)]
        
        return resized


def create_composite_timeline_frames(timeline_data):
    """
    Create composite frames combining ground truth, noisy images, clean predictions, and cortex visualizations.
    Layout: Ground Truth | Noisy | Clean | Brain Attention (2 columns)
    
    Args:
        timeline_data: Dictionary with timestep data
        
    Returns:
        List of composite frame arrays
    """
    
    composite_frames = []
    n_timesteps = len(timeline_data['timesteps'])
    ground_truth_img = timeline_data['ground_truth_img']

    for i in tqdm(range(n_timesteps), desc="Creating composite frames", unit="frame"):
        # Get components for this timestep
        timestep = timeline_data['timesteps'][i]
        noisy_img = timeline_data['noisy_images'][i]
        clean_img = timeline_data['clean_images'][i]
        cortex_frame = timeline_data['cortex_frames'][i]
        
        # Skip if any component is missing (except ground truth which is constant)
        if noisy_img is None or clean_img is None or cortex_frame is None:
            continue
        
        # Ensure images are uint8
        if ground_truth_img is not None and ground_truth_img.dtype != np.uint8:
            ground_truth_img = (np.clip(ground_truth_img, 0, 1) * 255).astype(np.uint8)
        if noisy_img.dtype != np.uint8:
            noisy_img = (np.clip(noisy_img, 0, 1) * 255).astype(np.uint8)
        if clean_img.dtype != np.uint8:
            clean_img = (np.clip(clean_img, 0, 1) * 255).astype(np.uint8)
        
        # Resize images to consistent size (e.g., 512x512)
        target_size = (512, 512)
        if ground_truth_img is not None:
            gt_resized = resize_image_array(ground_truth_img, target_size)
        else:
            # Create placeholder if ground truth is missing
            gt_resized = np.ones(target_size + (3,), dtype=np.uint8) * 128
        
        noisy_resized = resize_image_array(noisy_img, target_size)
        clean_resized = resize_image_array(clean_img, target_size)
        
        # Resize cortex frame to be larger (2x width)
        cortex_target_size = (target_size[0], target_size[1] * 2)
        cortex_resized = resize_image_array(cortex_frame, cortex_target_size)
        
        # Create composite frame with 5 columns (brain attention uses 2 columns)
        gap = 10  # Gap between images
        composite_width = target_size[1] * 3 + cortex_target_size[1] + gap * 4  # 3 regular + 1 double-width + 4 gaps
        composite_height = target_size[0] + 60  # Extra space for title
        
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255
        
        # Place images: Ground Truth | Noisy | Clean | Brain Attention (2 columns)
        x_offset = 0
        composite[30:30+target_size[0], x_offset:x_offset+target_size[1]] = gt_resized
        x_offset += target_size[1] + gap
        
        composite[30:30+target_size[0], x_offset:x_offset+target_size[1]] = noisy_resized
        x_offset += target_size[1] + gap
        
        composite[30:30+target_size[0], x_offset:x_offset+target_size[1]] = clean_resized
        x_offset += target_size[1] + gap
        
        composite[30:30+target_size[0], x_offset:x_offset+cortex_target_size[1]] = cortex_resized
        
        # Add title (simple text rendering)
        title = f"Timestep {timestep:3d}"
        
        composite_frames.append(composite)
    
    return composite_frames


def create_denoising_timeline_visualization(
    sample_id, 
    model_dir,
    saved_epochs,
    test_dataset=None, 
    metadata=None, 
    dominant_rois=None,
    subject="fsaverage",
    layer_name=None,  # If None, will use the first available layer
    spatial_idx=None,  # If None, average across spatial tokens
    head_idx=None,     # If None, average across heads
    save_gif=True,
    gif_filename=None,
    fps=6,
    max_timesteps=None,  # Limit number of timesteps for faster processing
    normalize_roi_weights=False  # Whether to renormalize weights for only valid ROI parcels
):
    """
    Create a comprehensive timeline visualization of the denoising process showing:
    1. Ground truth image (constant across all timesteps)
    2. Noisy images across timesteps
    3. Clean predictions across timesteps  
    4. Brain surface attention mappings across timesteps
    
    Args:
        sample_id: Sample identifier as integer (e.g., 0, 1, 2)
        model_dir: Model directory name (e.g., "08_14_2025-00_21")
        saved_epochs: Epoch number (e.g., "200")
        test_dataset: Dataset with parcel information
        metadata: Brain metadata with ROI mappings
        dominant_rois: ROI assignments per parcel
        subject: Pycortex subject name
        layer_name: Which attention layer to visualize (None = use first available)
        spatial_idx: Which spatial token to visualize (None = average all)
        head_idx: Which attention head to visualize (None = average all)
        save_gif: Whether to save as animated GIF
        gif_filename: Custom filename for GIF (None = auto-generate)
        fps: Frames per second for GIF
        max_timesteps: Maximum number of timesteps to process (None = all)
        normalize_roi_weights: Whether to renormalize weights for only valid ROI parcels
    
    Returns:
        Dictionary with timestep data and file paths
    """
    
    # Ensure subject exists for pycortex
    if subject not in cortex.db.subjects:
        cortex.download_subject(subject)
    
    # Construct sample directory path using new structure
    base_dir = f"brain_adapter/attn_maps/{model_dir}/epoch_{saved_epochs}/subset"
    sample_dir = Path(base_dir) / f"sample_{sample_id:06d}"
    
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    
    # Get ground truth image from dataset
    try:
        ground_truth_data = test_dataset[sample_id]
        ground_truth_img = ground_truth_data['img_encoder']
        ground_truth_img = (ground_truth_img - ground_truth_img.min()) / (ground_truth_img.max() - ground_truth_img.min())
        # Convert to numpy array if it's a tensor
        if hasattr(ground_truth_img, 'numpy'):
            ground_truth_img = ground_truth_img.numpy()
        # Ensure it's in the right format [H, W, C] and uint8
        if ground_truth_img.dtype != np.uint8:
            if ground_truth_img.max() <= 1.0:
                ground_truth_img = (ground_truth_img * 255).astype(np.uint8)
            else:
                ground_truth_img = ground_truth_img.astype(np.uint8)
        # Handle channel dimension if needed
        if ground_truth_img.ndim == 3 and ground_truth_img.shape[0] == 3:
            ground_truth_img = ground_truth_img.transpose(1, 2, 0)
    except Exception as e:
        print(f"Warning: Could not load ground truth image for sample {sample_id}: {e}")
        ground_truth_img = None
    
    # Get all timestep files and sort them numerically
    timestep_files = list(sample_dir.glob("time_step_*.npz"))
    
    def extract_timestep(filename):
        match = re.search(r'time_step_(\d+)\.npz', str(filename))
        return int(match.group(1)) if match else 0
    
    timestep_files.sort(key=extract_timestep, reverse=True)
    
    if max_timesteps is not None:
        timestep_files = timestep_files[:max_timesteps]
    
    print(f"Processing {len(timestep_files)} timesteps for sample {sample_id}")
    
    # Storage for all timeline data
    timeline_data = {
        'timesteps': [],
        'ground_truth_img': ground_truth_img,  # Store ground truth for all frames
        'noisy_images': [],
        'clean_images': [],
        'cortex_frames': [],
        'attention_stats': []
    }
    
    # Process each timestep
    vmin_global, vmax_global = None, None  # For consistent cortex scaling
    
    # First pass: collect attention weight ranges for consistent scaling
    print("Computing global attention weight range...")
    all_attention_weights = []
    
    for timestep_file in timestep_files[:5]:  # Sample first 5 for range estimation
        try:
            npz_data = np.load(timestep_file, allow_pickle=True)
            attn_maps = npz_data['attention_maps'].item()
            
            # Use first available layer if none specified
            if layer_name is None:
                layer_name = list(attn_maps.keys())[0]
            
            if layer_name in attn_maps:
                attn_map = attn_maps[layer_name]['ip_attn_map']
                
                # Handle batch dimension
                if attn_map.shape[0] == 2:
                    attn_map = attn_map[1]  # Use conditional attention
                
                # Process attention weights
                if spatial_idx is None:
                    attn_weights = attn_map.mean(axis=1)  # Average across spatial
                else:
                    attn_weights = attn_map[:, spatial_idx, :]
                
                if head_idx is None:
                    attn_weights = attn_weights.mean(axis=0)  # Average across heads
                else:
                    attn_weights = attn_weights[head_idx, :]
                
                all_attention_weights.extend(attn_weights.flatten())
        except Exception as e:
            print(f"Warning: Could not process {timestep_file} for range estimation: {e}")
            continue
    
    if all_attention_weights:
        vmin_global = np.min(all_attention_weights)
        vmax_global = np.max(all_attention_weights)
        print(f"Global attention range: [{vmin_global:.4f}, {vmax_global:.4f}]")
    else:
        vmin_global, vmax_global = 0.0, 1.0
        print("Warning: Could not determine attention range, using [0, 1]")
    
    # Second pass: create visualizations
    print("Creating timeline visualizations...")

    for i, timestep_file in tqdm(enumerate(timestep_files), 
                                  desc="Processing timesteps", 
                                  total=len(timestep_files), 
                                  unit="timestep"):
        try:
            # Load timestep data
            npz_data = np.load(timestep_file, allow_pickle=True)
            timestep = extract_timestep(timestep_file)
            
            # Extract images
            noisy_images = npz_data.get('noisy_images', None)
            clean_predictions = npz_data.get('clean_predictions', None)
            attn_maps = npz_data['attention_maps'].item()
            
            # Store basic data
            timeline_data['timesteps'].append(timestep)
            
            if noisy_images is not None:
                timeline_data['noisy_images'].append(noisy_images[0])  # Remove batch dim
            else:
                timeline_data['noisy_images'].append(None)
                
            if clean_predictions is not None:
                timeline_data['clean_images'].append(clean_predictions[0])  # Remove batch dim
            else:
                timeline_data['clean_images'].append(None)
            
            # Process attention for cortex visualization
            if layer_name in attn_maps:
                attn_map = attn_maps[layer_name]['ip_attn_map']
                
                # Handle batch dimension
                if attn_map.shape[0] == 2:
                    attn_map = attn_map[1]  # Use conditional attention
                
                # Create cortex visualization
                cortex_data = create_cortex_attention_frame(
                    attn_map, test_dataset, metadata, dominant_rois,
                    spatial_idx=spatial_idx, head_idx=head_idx,
                    subject=subject, vmin=vmin_global, vmax=vmax_global,
                    normalize_roi_weights=normalize_roi_weights
                )
                
                timeline_data['cortex_frames'].append(cortex_data['frame'])
                timeline_data['attention_stats'].append(cortex_data['stats'])
                
            else:
                print(f"Warning: Layer {layer_name} not found in timestep {timestep}")
                timeline_data['cortex_frames'].append(None)
                timeline_data['attention_stats'].append(None)
                
        except Exception as e:
            print(f"Error processing timestep {timestep_file}: {e}")
            # Add None placeholders to maintain alignment
            timeline_data['noisy_images'].append(None)
            timeline_data['clean_images'].append(None)
            timeline_data['cortex_frames'].append(None)
            timeline_data['attention_stats'].append(None)
            continue
    
    # Create composite visualization
    print("Creating composite visualization...")
    composite_frames = create_composite_timeline_frames(timeline_data)
    
    # Save as GIF if requested
    if save_gif and composite_frames:
        if gif_filename is None:
            # Save to denoising_vis directory instead of attn_maps
            output_dir = f"brain_adapter/denoising_vis/{model_dir}/epoch_{saved_epochs}"
            gif_filename = f"{output_dir}/sample_{sample_id:06d}.gif"
        
        # Ensure output directory exists
        Path(gif_filename).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving animated GIF: {gif_filename}")
        imageio.mimsave(gif_filename, composite_frames, fps=fps)
        print(f"GIF saved with {len(composite_frames)} frames at {fps} FPS")
    
    # Display final summary
    print(f"\nTimeline Summary for Sample {sample_id}:")
    print(f"  Total timesteps processed: {len(timeline_data['timesteps'])}")
    if timeline_data['timesteps']:
        print(f"  Timestep range: {min(timeline_data['timesteps'])} - {max(timeline_data['timesteps'])}")
    print(f"  Layer used: {layer_name}")
    if spatial_idx is not None:
        print(f"  Spatial token: {spatial_idx}")
    else:
        print(f"  Spatial tokens: averaged across all")
    if head_idx is not None:
        print(f"  Attention head: {head_idx}")
    else:
        print(f"  Attention heads: averaged across all")
    
    return {
        'timeline_data': timeline_data,
        'composite_frames': composite_frames,
        'gif_filename': gif_filename if save_gif else None,
        'sample_id': sample_id,
        'layer_name': layer_name,
        'model_dir': model_dir,
        'saved_epochs': saved_epochs
    }


def main():
    """Main function to run denoising timeline visualization."""
    parser = argparse.ArgumentParser(
        description="Create denoising timeline visualization with brain attention mapping"
    )
    
    # Required arguments
    parser.add_argument("--sample_id", type=int, required=True,
                       help="Sample ID to visualize (e.g., 0, 1, 2)")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Model directory name (e.g., '08_14_2025-00_21')")
    parser.add_argument("--saved_epochs", type=str, required=True,
                       help="Epoch number (e.g., '200')")
    
    # Optional arguments
    parser.add_argument("--subject_id", type=int, default=1,
                       help="Subject ID for brain data (1-8)")
    parser.add_argument("--max_timesteps", type=int, default=None,
                       help="Maximum number of timesteps to process")
    parser.add_argument("--layer_name", type=str, default='mid_block.attentions.0.transformer_blocks.0.attn2',
                       choices=['down_blocks.0.attentions.0.transformer_blocks.0.attn2',
                                'mid_block.attentions.0.transformer_blocks.0.attn2',
                                'up_blocks.3.attentions.2.transformer_blocks.0.attn2'],
                       help="Specific attention layer to visualize")
    parser.add_argument("--spatial_idx", type=int, default=None,
                       help="Specific spatial token index (None = average all)")
    parser.add_argument("--head_idx", type=int, default=None,
                       help="Specific attention head index (None = average all)")
    parser.add_argument("--fps", type=int, default=6,
                       help="Frames per second for GIF output")
    parser.add_argument("--topk", type=int, default=100,
                       help="Top-k parcels per hemisphere")
    parser.add_argument("--normalize_roi_weights", action="store_true",
                       help="Whether to renormalize weights for only valid ROI parcels")
    parser.add_argument("--no_gif", action="store_true",
                       help="Skip GIF creation")
    
    args = parser.parse_args()
    
    print("Brain Adapter Denoising Timeline Visualization")
    print("=" * 60)
    print(f"Sample ID: {args.sample_id}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Saved Epochs: {args.saved_epochs}")
    print(f"Subject ID: {args.subject_id}")
    print(f"Max Timesteps: {args.max_timesteps or 'All'}")
    print("=" * 60)
    
    # Setup Pycortex
    setup_cortex_inkscape()
    
    # Setup brain dataset
    test_dataset, metadata, dominant_rois, all_roi_names = setup_brain_dataset(
        subject_id=args.subject_id, topk=args.topk
    )
    
    # Create visualization
    try:
        result = create_denoising_timeline_visualization(
            sample_id=args.sample_id,
            model_dir=args.model_dir,
            saved_epochs=args.saved_epochs,
            test_dataset=test_dataset,
            metadata=metadata,
            dominant_rois=dominant_rois,
            subject="fsaverage",
            layer_name=args.layer_name,
            spatial_idx=args.spatial_idx,
            head_idx=args.head_idx,
            save_gif=not args.no_gif,
            fps=args.fps,
            max_timesteps=args.max_timesteps,
            normalize_roi_weights=args.normalize_roi_weights
        )
        
        print(f"\nVisualization completed successfully!")
        if result['gif_filename']:
            print(f"GIF saved: {result['gif_filename']}")
        print(f"Total frames: {len(result['composite_frames'])}")

    except Exception as e:
        print(f"\nError creating visualization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

'''
python plot_cross_attention.py \
    --model_dir 08_14_2025-00_21 \
    --sample_id 0 \
    --layer_name "mid_block.attentions.0.transformer_blocks.0.attn2" \
    --saved_epochs 200 \
    --normalize_roi_weights \
    --fps 8
'''