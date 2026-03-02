#!/usr/bin/env python3
"""
Brain Adapter Attention Dynamics Visualization

Two visualization modes:
1. Parcel Contribution: Pycortex brain surface with ranking-based color mapping
2. ROI Attention Map: 1x3 layout (target, reconstruction, attention overlay)

Usage:
    # Parcel contribution with 5 levels, every 5th timestep
    python plot_attn_dynamics.py --sample_id 0 --model_dir 08_16_2025-18_03 \
                                 --saved_epochs 200 --parcel_contribution --levels 10 --interval 5 \
                                 --roi_attention Body 

Author: Brain Adapter Team
Date: August 2025
"""

import os
import argparse
import glob
import gc
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
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

# Suppress warnings
@contextmanager
def suppress_print():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, e = io.StringIO(), io.StringIO()
        with redirect_stdout(f), redirect_stderr(e):
            yield

# Set working directory
os.chdir("/engram/nklab/pf2477/brain_decoding/")

# Import dataset utilities
from brain_adapter.dataset import nsd_topk_parcel_dataset, get_dominant_roi_per_parcel

# ROI groups definition
ROI_GROUPS = {
    "V1": ["V1v", "V1d"],
    "V2": ["V2v", "V2d"],
    "V3": ["V3v", "V3d"],
    "V4": ["hV4"],
    "Body": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
    "Face": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
    "Scene": ["OPA", "PPA", "RSC"],
    "Word": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
}


def setup_cortex_inkscape():
    """Setup robust Inkscape version detection for Pycortex."""
    import subprocess as sp
    import re
    
    def robust_inkscape_version():
        env = os.environ.copy()
        env.setdefault('LANG','C.UTF-8')
        env.setdefault('LC_ALL','C.UTF-8')
        p = sp.run(['inkscape','--version'], stdout=sp.PIPE, stderr=sp.STDOUT, text=True, env=env)
        m = re.search(r'Inkscape\s+([0-9]+(?:\.[0-9]+)*)', p.stdout or "")
        return m.group(1) if m else '0'

    import cortex.svgoverlay as svo
    svo.INKSCAPE_VERSION = robust_inkscape_version()


def setup_brain_dataset(subject_id=1, topk=100):
    """Setup brain dataset and metadata."""
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
    
    test_dataset = nsd_topk_parcel_dataset(args, split='test', transform=None, topk=args.topk)
    
    neural_data_path = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )
    metadata = np.load(
        neural_data_path / f"metadata_sub-{subject_id:02}.npy", allow_pickle=True
    ).item()
    
    return test_dataset, metadata


def setup_dominant_rois(dataset, metadata, min_overlap_threshold=0.75):
    """Setup dominant ROI mappings."""
    all_roi_names = list(metadata['lh_rois'].keys())[:24] + list(metadata['rh_rois'].keys())[:24]
    return get_dominant_roi_per_parcel(dataset, metadata, all_roi_names, min_overlap_threshold=min_overlap_threshold)


def load_attention_data(sample_folder, interval=1, max_timesteps=None):
    """Load attention data with query-weighted integration."""
    sample_path = Path(sample_folder)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample folder not found: {sample_folder}")
    
    timestep_files = sorted(glob.glob(str(sample_path / "time_step_*.npz")), reverse=True)
    
    # Apply interval filtering: select every Nth timestep
    if interval > 1:
        timestep_files = timestep_files[::interval]
    if max_timesteps:
        timestep_files = timestep_files[:max_timesteps]
    
    attention_data = {}
    timesteps = []
    
    for file_path in tqdm(timestep_files, desc="Loading attention data"):
        timestep = int(Path(file_path).stem.split("_")[-1])
        timesteps.append(timestep)
        
        npz_data = np.load(file_path, allow_pickle=True)
        attn_maps = npz_data['attention_maps'].item()
        clean_predictions = npz_data.get('clean_predictions', None)
        
        # Query-weighted integration across layers
        integrated_attention = integrate_attention_layers(attn_maps)
        
        attention_data[timestep] = {
            'attention_maps': attn_maps,
            'integrated_attention': integrated_attention,
            'clean_predictions': clean_predictions
        }
        
        del npz_data, attn_maps
        gc.collect()
    
    return {
        'timesteps': sorted(timesteps, reverse=True),
        'attention_data': attention_data
    }


def integrate_attention_layers(attn_maps):
    """Integrate attention across layers with query weighting."""
    integrated_attention = None
    total_query_weight = 0
    
    for layer_name, ip_attn in attn_maps.items():
        num_heads, num_queries, num_tokens = ip_attn.shape
        
        # Average across heads and spatial queries
        layer_attention = ip_attn.mean(axis=(0, 1))  # [tokens]
        
        # Weight by query count
        query_weight = num_queries
        
        if integrated_attention is None:
            integrated_attention = layer_attention * query_weight
        else:
            integrated_attention += layer_attention * query_weight
        
        total_query_weight += query_weight
    
    if total_query_weight > 0:
        integrated_attention = integrated_attention / total_query_weight
    
    return integrated_attention


def create_parcel_contribution_frame(attention_weights, test_dataset, metadata, 
                                   subject="fsaverage", levels=5):
    """Create pycortex visualization with ranking-based color mapping."""
    # Get hemisphere sizes
    lh_hemi_size = len(next(iter(metadata['lh_rois'].values())))
    rh_hemi_size = len(next(iter(metadata['rh_rois'].values())))
    
    # Initialize cortical surface
    lh_values = np.zeros(lh_hemi_size, dtype=np.float32)
    rh_values = np.zeros(rh_hemi_size, dtype=np.float32)
    
    # Get selected voxel indices
    sel_vox = test_dataset.get_selected_voxel_indices()
    
    # Create ranking-based values
    ranking_values = create_ranking_values(attention_weights, levels)
    
    # Paint values onto cortical surface
    for hemi, values in [("lh", lh_values), ("rh", rh_values)]:
        parcel_indices = test_dataset.selected_parcel_idx[hemi]
        
        for parcel_pos, parcel_idx in enumerate(parcel_indices):
            vox_indices = sel_vox[hemi][parcel_pos]
            if isinstance(vox_indices, torch.Tensor):
                vox_indices = vox_indices.cpu().numpy()
            
            if vox_indices.size == 0:
                continue
            
            # Calculate token index
            if hemi == "lh":
                token_idx = parcel_pos
            else:
                token_idx = parcel_pos + test_dataset.num_parcels // 2
            
            if token_idx < len(ranking_values):
                values[vox_indices] = ranking_values[token_idx]
    
    # Combine hemispheres for Pycortex
    cortex_data = np.concatenate([lh_values, rh_values])
    
    # Create mask for unlabeled areas and set them to a special value
    unlabeled_mask = (cortex_data == 0)
    cortex_data_masked = cortex_data.copy()
    cortex_data_masked[unlabeled_mask] = -1  # Special value for unlabeled areas
    
    # Create custom colormap with light grey for unlabeled areas
    import matplotlib.colors as mcolors
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0, 1, levels))
    colors = np.vstack([[0.85, 0.85, 0.85, 1.0], colors])  # Add light grey at the beginning
    custom_cmap = mcolors.ListedColormap(colors)
    
    # Create vertex object
    vertex_data = cortex.Vertex(
        cortex_data_masked,
        subject=subject,
        cmap=custom_cmap,
        vmin=-1,
        vmax=levels
    )
    
    # Render frame
    with suppress_print():
        fig = cortex.quickflat.make_figure(vertex_data, with_curvature=True, with_colorbar=False)
        fig.set_size_inches(10, 5)
    
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = frame[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    
    return frame


def create_ranking_values(attention_weights, levels):
    """Create ranking-based values for visualization."""
    if len(attention_weights) == 0:
        return np.array([])
    
    # Sort and create percentile-based ranking
    sorted_indices = np.argsort(attention_weights)[::-1]  # Descending order
    ranking_values = np.zeros_like(attention_weights)
    
    n_tokens = len(attention_weights)
    tokens_per_level = n_tokens // levels
    
    for level in range(levels):
        start_idx = level * tokens_per_level
        end_idx = (level + 1) * tokens_per_level if level < levels - 1 else n_tokens
        
        # Assign level value (levels for top, 1 for bottom)
        level_value = levels - level
        
        for idx in sorted_indices[start_idx:end_idx]:
            ranking_values[idx] = level_value
    
    return ranking_values


def get_roi_token_indices(roi_name, test_dataset, metadata, dominant_rois):
    """Get token indices for specific ROI."""
    if roi_name not in ROI_GROUPS:
        print(f"ROI '{roi_name}' not found. Available: {list(ROI_GROUPS.keys())}")
        return []
    
    target_rois = ROI_GROUPS[roi_name]
    token_indices = []
    
    for hemi in ['lh', 'rh']:
        for parcel_idx, roi_name_found, overlap, num_voxels in dominant_rois[hemi]:
            if roi_name_found in target_rois:
                selected_parcels = test_dataset.selected_parcel_idx[hemi]
                if isinstance(selected_parcels, np.ndarray):
                    selected_parcels = selected_parcels.tolist()
                
                try:
                    parcel_pos = selected_parcels.index(parcel_idx)
                    if hemi == "lh":
                        token_idx = parcel_pos
                    else:
                        token_idx = parcel_pos + test_dataset.num_parcels // 2
                    token_indices.append(token_idx)
                except ValueError:
                    continue
    
    return token_indices


def create_roi_attention_map(attn_maps, roi_token_indices, image_shape):
    """Create ROI attention map showing where ROI tokens attend to."""
    if len(roi_token_indices) == 0:
        return None
    
    layer_maps = []
    
    for layer_name, A_layer in attn_maps.items():
        if isinstance(A_layer, np.ndarray):
            A_layer = torch.from_numpy(A_layer)
        
        H, q_l, p = A_layer.shape
        spatial_dim = int(np.sqrt(q_l))
        
        # Average attention across ROI tokens and heads
        roi_attention = A_layer[:, :, roi_token_indices].mean(dim=(0, 2))  # [q_l]
        
        # Reshape to spatial grid
        attention_2d = roi_attention.reshape(spatial_dim, spatial_dim)
        
        # Upsample to image size
        attention_tensor = attention_2d.float().unsqueeze(0).unsqueeze(0)
        target_shape = image_shape[:2] if len(image_shape) > 2 else image_shape
        
        upsampled = torch.nn.functional.interpolate(
            attention_tensor, size=target_shape, mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize
        if upsampled.sum() > 0:
            upsampled = upsampled / upsampled.sum()
        
        layer_maps.append(upsampled)
    
    if len(layer_maps) > 0:
        # Equal average across layers
        roi_map = np.mean(layer_maps, axis=0)
        if roi_map.sum() > 0:
            roi_map = roi_map / roi_map.sum()
        return roi_map
    
    return None


def create_attention_overlay(clean_image, attention_map, sigma=10):
    """Create attention overlay on clean image."""
    if attention_map is None:
        return clean_image
    
    # Smooth attention map
    smoothed = gaussian_filter(attention_map, sigma=sigma)

    enhanced = smoothed.copy()
    
    # Apply contrast enhancement
    p = 50  # Top 50%
    softness = 0.25
    tau = np.quantile(smoothed, 1 - p/100.0)
    beta = softness * (smoothed.max() - smoothed.min() + 1e-12)
    enhanced = 1.0 / (1.0 + np.exp(-(smoothed - tau) / (beta + 1e-12)))
    
    # Normalize
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    enhanced = np.nan_to_num(enhanced, nan=0.0)
    
    # Create colored overlay
    cmap = plt.get_cmap('viridis')
    alpha_values = np.linspace(0, 1, cmap.N)
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:, -1] = alpha_values
    alpha_cmap = ListedColormap(alpha_cmap)
    
    colored_attention = alpha_cmap(enhanced)
    
    # Apply overlay
    result = clean_image.copy().astype(np.float32) / 255.0
    overlay_alpha = colored_attention[:, :, 3:4]
    result = result * (1 - overlay_alpha) + colored_attention[:, :, :3] * overlay_alpha
    result = (result * 255).astype(np.uint8)
    
    return result


def resize_image(img, target_size):
    """Resize image using PIL."""
    if len(img.shape) == 2:
        pil_img = Image.fromarray(img, mode='L')
    else:
        pil_img = Image.fromarray(img, mode='RGB')
    
    resized = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    return np.array(resized)


def add_text_overlay(image, text, x, y, font_size=24):
    """Add text overlay to image with transparent background."""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load font
    font = None
    font_paths = [
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Windows/Fonts/arial.ttf"
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Draw text directly without background
    draw.text((x, y), text, font=font, fill=(0, 0, 0))  # Black text
    
    return np.array(img_pil)


def create_parcel_contribution_visualization(sample_id, model_dir, saved_epochs, 
                                           test_dataset, metadata, levels=5,
                                           interval=1, max_timesteps=None, fps=6):
    """Create parcel contribution visualization with ranking-based coloring."""
    print(f"Creating parcel contribution visualization for sample {sample_id}")
    
    # Construct sample directory
    base_dir = f"brain_adapter/attn_maps/{model_dir}/epoch_{saved_epochs}/subset"
    sample_dir = Path(base_dir) / f"sample_{sample_id:06d}"
    
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    
    # Load attention data
    data_dict = load_attention_data(sample_dir, interval, max_timesteps)
    timesteps = data_dict['timesteps']
    attention_data = data_dict['attention_data']
    
    print(f"Processing {len(timesteps)} timesteps (interval={interval})")
    
    frames = []
    
    for timestep in tqdm(timesteps, desc="Creating parcel contribution frames"):
        timestep_data = attention_data[timestep]
        integrated_attention = timestep_data['integrated_attention']
        
        if integrated_attention is None:
            continue
        
        # Create pycortex frame
        frame = create_parcel_contribution_frame(
            integrated_attention, test_dataset, metadata, levels=levels
        )
        
        # Add timestep text
        # frame = add_text_overlay(frame, f"Timestep {timestep}", 10, 10)
        frames.append(frame)
    
    # Save GIF
    output_dir = f"brain_adapter/denoising_vis/{model_dir}/epoch_{saved_epochs}"
    gif_path = f"{output_dir}/sample_{sample_id:06d}_parcel_contribution_levels{levels}_interval{interval}.gif"
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    
    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Parcel contribution GIF saved: {gif_path}")
    
    return {'frames': frames, 'gif_path': gif_path, 'timesteps': timesteps}


def create_roi_attention_visualization(sample_id, model_dir, saved_epochs, roi_name,
                                     test_dataset, metadata, dominant_rois,
                                     interval=1, max_timesteps=None, fps=6):
    """Create ROI attention visualization: target + reconstruction + attention overlay."""
    print(f"Creating ROI attention visualization for sample {sample_id}, ROI: {roi_name}")
    
    # Construct sample directory
    base_dir = f"brain_adapter/attn_maps/{model_dir}/epoch_{saved_epochs}/subset"
    sample_dir = Path(base_dir) / f"sample_{sample_id:06d}"
    
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    
    # Get original image
    original_img = test_dataset[sample_id]['img_encoder']
    if isinstance(original_img, torch.Tensor):
        if original_img.dim() == 3:
            original_img = original_img.permute(1, 2, 0)
        original_img = original_img.cpu().numpy()
    
    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
    
    # Get ROI token indices
    roi_token_indices = get_roi_token_indices(roi_name, test_dataset, metadata, dominant_rois)
    if not roi_token_indices:
        raise ValueError(f"No tokens found for ROI: {roi_name}")
    
    print(f"Found {len(roi_token_indices)} tokens for ROI {roi_name}")
    
    # Load attention data
    data_dict = load_attention_data(sample_dir, interval, max_timesteps)
    timesteps = data_dict['timesteps']
    attention_data = data_dict['attention_data']
    
    print(f"Processing {len(timesteps)} timesteps (interval={interval})")
    
    frames = []
    target_size = (400, 400)
    
    for timestep in tqdm(timesteps, desc="Creating ROI attention frames"):
        timestep_data = attention_data[timestep]
        attn_maps = timestep_data['attention_maps']
        clean_predictions = timestep_data['clean_predictions']
        
        if attn_maps is None or clean_predictions is None:
            continue
        
        # Get clean image for this timestep
        clean_img = clean_predictions[0] if len(clean_predictions.shape) == 4 else clean_predictions
        if clean_img.dtype != np.uint8:
            clean_img = (np.clip(clean_img, 0, 1) * 255).astype(np.uint8)
        
        # Create ROI attention map
        roi_map = create_roi_attention_map(attn_maps, roi_token_indices, clean_img.shape)
        
        # Create attention overlay
        overlay_img = create_attention_overlay(clean_img, roi_map)
        
        # Resize all images
        original_resized = resize_image(original_img, target_size)
        clean_resized = resize_image(clean_img, target_size)
        overlay_resized = resize_image(overlay_img, target_size)
        
        # Create 1x3 composite
        gap = 15
        composite_width = target_size[1] * 3 + gap * 2
        composite_height = target_size[0] + 80
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255
        
        # Add titles
        composite = add_text_overlay(composite, 'Target', 50, 10)
        composite = add_text_overlay(composite, 'Reconstruction', target_size[1] + gap + 50, 100)
        composite = add_text_overlay(composite, f'{roi_name} Attention', target_size[1] * 2 + gap * 2 + 50, 10)
        
        # Add timestep
        # composite = add_text_overlay(composite, f'Timestep {timestep}', 10, composite_height - 30)
        
        # Place images
        y_start = 40
        composite[y_start:y_start+target_size[0], 0:target_size[1]] = original_resized
        composite[y_start:y_start+target_size[0], target_size[1]+gap:target_size[1]*2+gap] = clean_resized
        composite[y_start:y_start+target_size[0], target_size[1]*2+gap*2:target_size[1]*3+gap*2] = overlay_resized
        
        frames.append(composite)
    
    # Save GIF
    output_dir = f"brain_adapter/denoising_vis/{model_dir}/epoch_{saved_epochs}"
    gif_path = f"{output_dir}/sample_{sample_id:06d}_roi_attention_{roi_name}_interval{interval}.gif"
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    
    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"ROI attention GIF saved: {gif_path}")
    
    return {'frames': frames, 'gif_path': gif_path, 'timesteps': timesteps, 'roi_name': roi_name}


def create_combined_visualization(sample_id, model_dir, saved_epochs, roi_name,
                                test_dataset, metadata, dominant_rois,
                                levels=5, interval=1, max_timesteps=None, fps=6):
    """
    Create combined visualization: target + reconstruction + attention overlay + brain surface (1x5 layout).
    
    Args:
        sample_id: Sample identifier
        model_dir: Model directory name
        saved_epochs: Epoch number
        roi_name: ROI name for attention overlay
        test_dataset: Dataset with parcel information
        metadata: Brain metadata with ROI mappings
        dominant_rois: Dominant ROI mappings
        levels: Number of ranking levels for brain surface
        interval: Load every N timesteps
        max_timesteps: Maximum number of timesteps to process
        fps: Frames per second for GIF
        
    Returns:
        Dictionary with results and file paths
    """
    print(f"Creating combined visualization for sample {sample_id}, ROI: {roi_name}")
    
    # Construct sample directory
    base_dir = f"brain_adapter/attn_maps/{model_dir}/epoch_{saved_epochs}/subset"
    sample_dir = Path(base_dir) / f"sample_{sample_id:06d}"
    
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    
    # Get original image
    original_img = test_dataset[sample_id]['img_encoder']
    if isinstance(original_img, torch.Tensor):
        if original_img.dim() == 3:
            original_img = original_img.permute(1, 2, 0)
        original_img = original_img.cpu().numpy()
    
    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
    
    # Get ROI token indices
    roi_token_indices = get_roi_token_indices(roi_name, test_dataset, metadata, dominant_rois)
    if not roi_token_indices:
        raise ValueError(f"No tokens found for ROI: {roi_name}")
    
    print(f"Found {len(roi_token_indices)} tokens for ROI {roi_name}")
    
    # Load attention data
    data_dict = load_attention_data(sample_dir, interval, max_timesteps)
    timesteps = data_dict['timesteps']
    attention_data = data_dict['attention_data']
    
    print(f"Processing {len(timesteps)} timesteps (interval={interval})")
    
    frames = []
    target_size = (400, 400)
    brain_size = (target_size[0], target_size[1] * 2)  # Double width for brain surface
    
    for timestep in tqdm(timesteps, desc="Creating combined frames"):
        timestep_data = attention_data[timestep]
        attn_maps = timestep_data['attention_maps']
        clean_predictions = timestep_data['clean_predictions']
        integrated_attention = timestep_data['integrated_attention']
        
        if attn_maps is None or clean_predictions is None or integrated_attention is None:
            continue
        
        # Get clean image for this timestep
        clean_img = clean_predictions[0] if len(clean_predictions.shape) == 4 else clean_predictions
        if clean_img.dtype != np.uint8:
            clean_img = (np.clip(clean_img, 0, 1) * 255).astype(np.uint8)
        
        # Create ROI attention map and overlay
        roi_map = create_roi_attention_map(attn_maps, roi_token_indices, clean_img.shape)
        overlay_img = create_attention_overlay(clean_img, roi_map)
        
        # Create pycortex brain surface visualization
        brain_frame = create_parcel_contribution_frame(
            integrated_attention, test_dataset, metadata, levels=levels
        )
        
        # Resize all images
        original_resized = resize_image(original_img, target_size)
        clean_resized = resize_image(clean_img, target_size)
        overlay_resized = resize_image(overlay_img, target_size)
        brain_resized = resize_image(brain_frame, brain_size)
        
        # Create 1x5 composite: Target + Clean + Overlay + Brain (2 columns)
        gap = 15
        left_right_padding = 15
        content_width = target_size[1] * 3 + brain_size[1] + gap * 3  # 3 regular + 1 double + 3 gaps
        composite_width = content_width + 2 * left_right_padding
        composite_height = target_size[0] + 100  # Increased from 80 to give more space
        
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255
        
        # Add titles with proper centering calculation
        subtitle_y = 10  # Increased from 5 to give more top margin
        font_size = 32
        
        # Calculate actual text widths using PIL for better centering
        from PIL import ImageFont
        font = None
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Windows/Fonts/arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (OSError, IOError):
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        titles = ['Target', 'Reconstruction', f'ROI Attention', 'Brain Parcel Contribution']
        
        # Calculate positions for each column
        x_offset = left_right_padding
        
        for i, title in enumerate(titles):
            # Get actual text width
            try:
                from PIL import ImageDraw
                temp_img = Image.new('RGB', (1, 1))
                temp_draw = ImageDraw.Draw(temp_img)
                bbox = temp_draw.textbbox((0, 0), title, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback estimation
                text_width = len(title) * (font_size * 0.6)
            
            # Calculate column width
            if i < 3:
                column_width = target_size[1]
            else:
                column_width = brain_size[1]
            
            # Center text within column
            title_x = x_offset + (column_width - text_width) // 2
            
            composite = add_text_overlay(composite, title, title_x, subtitle_y, font_size=font_size)
            
            # Move to next column
            if i < 2:
                x_offset += target_size[1] + gap
            elif i == 2:
                x_offset += target_size[1] + gap
            # For brain column (i == 3), it's already positioned
        
        # Place images with more vertical spacing
        y_start = 60  # Increased from 40 to give more space between titles and images
        x_offset = left_right_padding
        
        # Column 1: Target
        composite[y_start:y_start+target_size[0], x_offset:x_offset+target_size[1]] = original_resized
        x_offset += target_size[1] + gap
        
        # Column 2: Clean prediction
        composite[y_start:y_start+target_size[0], x_offset:x_offset+target_size[1]] = clean_resized
        x_offset += target_size[1] + gap
        
        # Column 3: Attention overlay
        composite[y_start:y_start+target_size[0], x_offset:x_offset+target_size[1]] = overlay_resized
        x_offset += target_size[1] + gap
        
        # Column 4-5: Brain surface (double width)
        composite[y_start:y_start+target_size[0], x_offset:x_offset+brain_size[1]] = brain_resized
        
        frames.append(composite)
    
    # Save GIF
    output_dir = f"brain_adapter/denoising_vis/{model_dir}/epoch_{saved_epochs}"
    gif_path = f"{output_dir}/sample_{sample_id:06d}_combined_{roi_name}_levels{levels}_interval{interval}.gif"
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    
    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Combined GIF saved: {gif_path}")
    
    return {'frames': frames, 'gif_path': gif_path, 'timesteps': timesteps, 'roi_name': roi_name}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Brain Adapter Attention Dynamics Visualization")
    
    # Required arguments
    parser.add_argument("--sample_id", type=int, required=True, help="Sample ID")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    parser.add_argument("--saved_epochs", type=str, required=True, help="Epoch number")
    
    # Visualization modes
    parser.add_argument("--parcel_contribution", action="store_true", 
                       help="Create parcel contribution visualization (considers all parcels)")
    parser.add_argument("--roi_attention", type=str,
                       help="Create ROI attention visualization for specified ROI")
    parser.add_argument("--combined", type=str, default="Body",
                       help="Create combined visualization (target + reconstruction + attention + brain surface) for specified ROI (default: Body)")
    
    # Optional arguments
    parser.add_argument("--levels", type=int, default=5,
                       help="Number of ranking levels for parcel contribution")
    parser.add_argument("--subject_id", type=int, default=1, help="Subject ID")
    parser.add_argument("--interval", type=int, default=1, 
                       help="Process every Nth timestep (1=all, 5=every 5th)")
    parser.add_argument("--max_timesteps", type=int, help="Max timesteps")
    parser.add_argument("--fps", type=int, default=6, help="GIF FPS")
    parser.add_argument("--topk", type=int, default=100, help="Top-k parcels")

    args = parser.parse_args()
    
    # Set default mode if none specified
    if args.roi_attention:
        args.combined = args.roi_attention  # Default to combined visualization with Body ROI
    
    print("Brain Adapter Attention Dynamics Visualization")
    print("=" * 50)
    print(f"Sample ID: {args.sample_id}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Saved Epochs: {args.saved_epochs}")
    print(f"Timestep Interval: {args.interval}")
    
    # Setup
    setup_cortex_inkscape()
    test_dataset, metadata = setup_brain_dataset(args.subject_id, args.topk)
    dominant_rois = setup_dominant_rois(test_dataset, metadata)
    
    try:
        # if args.parcel_contribution:
        #     print(f"\nCreating parcel contribution visualization with {args.levels} levels")
        #     print("Note: All parcels are considered for ranking-based visualization")
        #     result = create_parcel_contribution_visualization(
        #         args.sample_id, args.model_dir, args.saved_epochs,
        #         test_dataset, metadata, args.levels,
        #         args.interval, args.max_timesteps, args.fps
        #     )
        #     print(f"✓ Parcel contribution GIF: {result['gif_path']}")
        
        # if args.roi_attention:
        #     print(f"\nCreating ROI attention visualization for {args.roi_attention}")
        #     print(f"Note: Only {args.roi_attention} ROI tokens are considered for attention overlay")
        #     result = create_roi_attention_visualization(
        #         args.sample_id, args.model_dir, args.saved_epochs, args.roi_attention,
        #         test_dataset, metadata, dominant_rois,
        #         args.interval, args.max_timesteps, args.fps
        #     )
        #     print(f"✓ ROI attention GIF: {result['gif_path']}")
        
        if args.combined:
            print(f"\nCreating combined visualization for {args.combined}")
            print(f"Note: 1x5 layout with target + reconstruction + {args.combined} attention + parcel contribution brain surface")
            result = create_combined_visualization(
                args.sample_id, args.model_dir, args.saved_epochs, args.combined,
                test_dataset, metadata, dominant_rois, args.levels,
                args.interval, args.max_timesteps, args.fps
            )
            print(f"✓ Combined GIF: {result['gif_path']}")
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
