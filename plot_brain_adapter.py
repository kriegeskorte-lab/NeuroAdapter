"""
Brain Adapter Results Plotting Script

This script visualizes ground truth and predicted images from brain adapter decoding results.
It loads the saved evaluation results and creates comparison plots showing ground truth vs predicted images.

The script supports:
- Plotting specific ranges of samples
- Side-by-side comparison of ground truth and predicted images  
- Saving plots in organized directory structure
- Loading results from both subset and full dataset evaluations
"""

import os
import sys
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_sample_data(results_dir, dataset_idx):
    """
    Load data for a specific sample.
    
    Args:
        results_dir: Directory containing evaluation results
        dataset_idx: Dataset index of the sample to load
        
    Returns:
        Dictionary containing sample data or None if not found
    """
    sample_filename = f"sample_{dataset_idx:06d}.npz"
    sample_path = os.path.join(results_dir, sample_filename)
    
    if not os.path.exists(sample_path):
        return None
    
    data = np.load(sample_path, allow_pickle=True)
    
    sample_data = {
        'groundtruth_image': data['groundtruth_image'],
        'predicted_image': data['predicted_image'],
        'dataset_index': int(data['dataset_index']),
        'evaluation_index': int(data['evaluation_index'])
    }
    
    # Add optional data if available
    if 'candidate_images' in data:
        sample_data['candidate_images'] = data['candidate_images']
        sample_data['correlation_scores'] = data['correlation_scores']
        sample_data['best_candidate_idx'] = int(data['best_candidate_idx'])
    
    data.close()
    return sample_data


def load_evaluation_metadata(results_dir):
    """
    Load evaluation metadata and summary.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Tuple of (metadata, summary) dictionaries
    """
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
    
    return metadata, summary


def normalize_image(image):
    """
    Normalize image to [0, 1] range for display.
    
    Args:
        image: Image array (can be various formats)
        
    Returns:
        Normalized image array in [0, 1] range
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.ndim == 3 and image.shape[0] in [1, 3]:  # Handle channel-first format
        image = np.transpose(image, (1, 2, 0))  # Convert
    
    # Handle different image formats
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        return (image - image.min()) / (image.max() - image.min())
    else:
        return np.clip(image, 0, 1)


def plot_image_comparison(sample_data, save_path=None, show_details=True):
    """
    Plot ground truth vs predicted image comparison for a single sample.
    If candidate images are available, show all candidates in a row.
    
    Args:
        sample_data: Dictionary containing sample data
        save_path: Path to save the plot (optional)
        show_details: Whether to show additional details in the plot
        
    Returns:
        matplotlib figure object
    """
    gt_image = normalize_image(sample_data['groundtruth_image'])
    pred_image = normalize_image(sample_data['predicted_image'])
    
    # Check if we have candidate images
    if 'candidate_images' in sample_data:
        # Show ground truth + all candidates
        candidates = sample_data['candidate_images']
        scores = sample_data['correlation_scores']
        best_idx = sample_data['best_candidate_idx']
        
        n_images = len(candidates) + 1  # +1 for ground truth
        fig, axes = plt.subplots(1, n_images, figsize=(n_images * 3, 4))
        
        # Handle single candidate case
        if n_images == 2:
            axes = [axes[0], axes[1]]
        
        # Plot ground truth first
        axes[0].imshow(gt_image)
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot all candidates
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            ax_idx = i + 1
            candidate_norm = normalize_image(candidate)
            
            axes[ax_idx].imshow(candidate_norm)
            
            # Highlight best candidate
            title_color = 'red' if i == best_idx else 'black'
            title_weight = 'bold' if i == best_idx else 'normal'
            
            if show_details:
                axes[ax_idx].set_title(f'Pred {i}\nCorr: {score:.4f}', 
                                     fontsize=10, color=title_color, fontweight=title_weight)
            else:
                axes[ax_idx].set_title(f'Pred {i}', 
                                     fontsize=10, color=title_color, fontweight=title_weight)
            axes[ax_idx].axis('off')
            
            # Add border for best candidate
            if i == best_idx:
                for spine in axes[ax_idx].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
    else:
        # Show only ground truth + best predicted (1x2 layout)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Plot ground truth
        axes[0].imshow(gt_image)
        axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot prediction
        axes[1].imshow(pred_image)
        axes[1].set_title('Predicted', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    # Add sample information
    if show_details:
        dataset_idx = sample_data['dataset_index']
        eval_idx = sample_data['evaluation_index']
        
        # Add best correlation info if available
        if 'correlation_scores' in sample_data:
            best_idx = sample_data['best_candidate_idx']
            best_score = sample_data['correlation_scores'][best_idx]
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    return fig


def plot_image_grid(sample_data_list, ncols=4, save_path=None, show_details=True):
    """
    Plot multiple samples in a grid layout.
    
    Args:
        sample_data_list: List of sample data dictionaries
        ncols: Number of columns in the grid
        save_path: Path to save the plot (optional)
        show_details: Whether to show additional details
        
    Returns:
        matplotlib figure object
    """
    n_samples = len(sample_data_list)
    nrows = (n_samples + ncols - 1) // ncols  # Ceiling division
    
    # Create figure with subplots for ground truth and predicted images
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 4, nrows * 6))
    
    # Handle single row case
    if nrows == 1 and ncols == 1:
        axes = axes.reshape(2, 1)
    elif nrows == 1:
        axes = axes.reshape(2, ncols)
    elif ncols == 1:
        axes = axes.reshape(nrows * 2, 1)
    
    for i, sample_data in enumerate(sample_data_list):
        row = (i // ncols) * 2
        col = i % ncols
        
        gt_image = normalize_image(sample_data['groundtruth_image'])
        pred_image = normalize_image(sample_data['predicted_image'])
        
        # Plot ground truth
        axes[row, col].imshow(gt_image)
        if show_details:
            dataset_idx = sample_data['dataset_index']
            axes[row, col].set_title(f'GT - Sample {dataset_idx:06d}', fontsize=10)
        else:
            axes[row, col].set_title('Ground Truth', fontsize=10)
        axes[row, col].axis('off')
        
        # Plot prediction
        axes[row + 1, col].imshow(pred_image)
        if show_details:
            axes[row + 1, col].set_title(f'Pred - Sample {dataset_idx:06d}', fontsize=10)
        else:
            axes[row + 1, col].set_title('Predicted', fontsize=10)
        axes[row + 1, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, nrows * ncols):
        row = (i // ncols) * 2
        col = i % ncols
        axes[row, col].set_visible(False)
        axes[row + 1, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid plot: {save_path}")
    
    return fig


def plot_candidates_comparison(sample_data, save_path=None):
    """
    Plot all candidate images with their correlation scores.
    
    Args:
        sample_data: Dictionary containing sample data with candidates
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib figure object or None if no candidates available
    """
    if 'candidate_images' not in sample_data:
        print("No candidate images available for this sample")
        return None
    
    candidates = sample_data['candidate_images']
    scores = sample_data['correlation_scores']
    best_idx = sample_data['best_candidate_idx']
    gt_image = normalize_image(sample_data['groundtruth_image'])
    
    n_candidates = len(candidates)
    ncols = min(4, n_candidates + 1)  # +1 for ground truth
    nrows = (n_candidates + 1 + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    
    # Handle single row case
    if nrows == 1:
        axes = axes.reshape(1, -1) if ncols > 1 else [axes]
    
    # Plot ground truth first
    axes.flat[0].imshow(gt_image)
    axes.flat[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes.flat[0].axis('off')
    
    # Plot candidates
    for i, (candidate, score) in enumerate(zip(candidates, scores)):
        ax_idx = i + 1
        candidate_norm = normalize_image(candidate)
        
        axes.flat[ax_idx].imshow(candidate_norm)
        
        # Highlight best candidate
        title_color = 'red' if i == best_idx else 'black'
        title_weight = 'bold' if i == best_idx else 'normal'
        
        axes.flat[ax_idx].set_title(f'Candidate {i}\nCorr: {score:.4f}', 
                                   fontsize=10, color=title_color, fontweight=title_weight)
        axes.flat[ax_idx].axis('off')
        
        # Add border for best candidate
        if i == best_idx:
            rect = patches.Rectangle((0, 0), candidate_norm.shape[1]-1, candidate_norm.shape[0]-1, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            axes.flat[ax_idx].add_patch(rect)
    
    # Hide unused subplots
    for i in range(n_candidates + 1, len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    dataset_idx = sample_data['dataset_index']
    fig.suptitle(f'All Candidates for Sample {dataset_idx:06d}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved candidates plot: {save_path}")
    
    return fig


def create_output_directory(results_dir, base_output_dir="brain_adapter/plotted_stimuli"):
    """
    Create output directory structure matching the results directory.
    
    Args:
        results_dir: Input results directory path
        base_output_dir: Base directory for output plots
        
    Returns:
        Output directory path
    """
    # Extract path components from results_dir
    # Expected format: .../decoded_stimuli/model_name/epoch_X/subset_or_full
    path_parts = Path(results_dir).parts
    
    # Find the decoded_stimuli part and extract everything after it
    try:
        decoded_idx = path_parts.index('decoded_stimuli')
        relative_path = Path(*path_parts[decoded_idx + 1:])  # Skip 'decoded_stimuli'
        output_dir = Path(base_output_dir) / relative_path
    except ValueError:
        # If decoded_stimuli not found, use the last 3 parts (model/epoch/mode)
        relative_path = Path(*path_parts[-3:])
        output_dir = Path(base_output_dir) / relative_path
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def main():
    """Main function for plotting brain adapter results."""
    parser = argparse.ArgumentParser(
        description="Plot ground truth vs predicted images from brain adapter results"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing evaluation results (e.g., brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: auto-generated based on results_dir)"
    )
    
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Starting dataset index to plot (default: plot all available)"
    )
    
    parser.add_argument(
        "--end_idx", 
        type=int,
        default=None,
        help="Ending dataset index to plot (exclusive, default: plot all available)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to plot (default: no limit)"
    )
    
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["individual", "grid", "all"],
        default="all",
        help="Type of plots to generate (candidates are included in individual plots)"
    )
    
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Number of columns in grid plot (default: 4)"
    )
    
    parser.add_argument(
        "--show_details",
        action="store_true",
        default=True,
        help="Show sample indices and correlation scores in plots"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150)"
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    print(f"Loading results from: {args.results_dir}")
    
    # Load metadata and summary
    try:
        metadata, summary = load_evaluation_metadata(args.results_dir)
        print(f"Found {summary['num_samples']} samples in {summary['mode']} evaluation")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return 1
    
    # Create output directory
    if args.output_dir is None:
        output_dir = create_output_directory(args.results_dir)
    else:
        output_dir = args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Get dataset indices to plot
    available_indices = summary['dataset_indices']
    
    # Filter indices based on arguments
    if args.start_idx is not None:
        available_indices = [idx for idx in available_indices if idx >= args.start_idx]
    if args.end_idx is not None:
        available_indices = [idx for idx in available_indices if idx < args.end_idx]
    if args.max_samples is not None:
        available_indices = available_indices[:args.max_samples]
    
    if not available_indices:
        print("No samples match the specified criteria")
        return 1
    
    print(f"Plotting {len(available_indices)} samples: {available_indices[0]} to {available_indices[-1]}")
    
    # Load sample data
    sample_data_list = []
    missing_samples = []
    
    for dataset_idx in available_indices:
        sample_data = load_sample_data(args.results_dir, dataset_idx)
        if sample_data is not None:
            sample_data_list.append(sample_data)
        else:
            missing_samples.append(dataset_idx)
    
    if missing_samples:
        print(f"Warning: Could not load {len(missing_samples)} samples: {missing_samples}")
    
    if not sample_data_list:
        print("No valid samples found to plot")
        return 1
    
    print(f"Successfully loaded {len(sample_data_list)} samples")
    
    # Generate plots based on plot_type
    if args.plot_type in ["individual", "all"]:
        print("Generating individual comparison plots...")
        individual_dir = os.path.join(output_dir, "individual")
        Path(individual_dir).mkdir(exist_ok=True)
        
        for sample_data in sample_data_list:
            dataset_idx = sample_data['dataset_index']
            save_path = os.path.join(individual_dir, f"sample_{dataset_idx:06d}.png")
            
            fig = plot_image_comparison(sample_data, save_path, args.show_details)
            plt.close(fig)
    
    if args.plot_type in ["grid", "all"]:
        print("Generating grid plot...")
        grid_path = os.path.join(output_dir, "grid_comparison.png")
        
        fig = plot_image_grid(sample_data_list, args.grid_cols, grid_path, args.show_details)
        plt.close(fig)
    
    # Save plotting summary
    plot_summary = {
        'results_directory': args.results_dir,
        'output_directory': output_dir,
        'samples_plotted': len(sample_data_list),
        'dataset_indices': [s['dataset_index'] for s in sample_data_list],
        'plot_types_generated': args.plot_type,
        'metadata': metadata
    }
    
    summary_path = os.path.join(output_dir, "plot_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(plot_summary, f, indent=2)
    
    print(f"\nPlotting completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Samples plotted: {len(sample_data_list)}")
    print(f"Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    """
    Usage Examples:
    
    # 1. Plot all samples from a subset evaluation:
    python plot_brain_adapter.py --results_dir brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset

    # 2. Plot specific range of samples:
    python plot_brain_adapter.py \
        --results_dir brain_adapter/decoded_stimuli/08_14_2025-00_21/epoch_200/subset \
        --start_idx 0 --end_idx 8
    
    # 3. Plot only grid comparison:
    python plot_brain_adapter.py brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset \
        --plot_type grid --grid_cols 3
    
    # 4. Plot only individual samples (with all candidates in each):
    python plot_brain_adapter.py brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset \
        --plot_type individual
    
    # 5. Plot with custom output directory:
    python plot_brain_adapter.py brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset \
        --output_dir custom_plots/my_experiment
    
    # 6. Plot maximum 20 samples:
    python plot_brain_adapter.py brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset \
        --max_samples 20
        
    # 7. Plot for multi-subject training
    python plot_brain_adapter.py \
        --results_dir brain_adapter/decoded_stimuli/08_10_2025-15_50/epoch_100/subset/8 \
        --start_idx 0 --end_idx 8
    
    # Individual plot format:
    # - If candidates available: 1 x (N+1) layout with GT + all N candidates
    # - If no candidates: 1 x 2 layout with GT + best prediction
    # - Best candidate highlighted with red border and title
    
    # Output Structure:
    # plotted_stimuli/
    # └── model_name/
    #     └── epoch_100/
    #         └── subset/
    #             ├── individual/
    #             │   ├── sample_000001.png  (GT + all candidates in one row)
    #             │   ├── sample_000002.png
    #             │   └── ...
    #             ├── grid_comparison.png
    #             └── plot_summary.json
    """
    import sys
    exit_code = main()
    sys.exit(exit_code)
