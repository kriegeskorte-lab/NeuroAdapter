"""
Brain Adapter Metrics Evaluation Script

This script evaluates the quality of brain-decoded images using multiple metrics:
- Pixel-wise correlation
- SSIM (Structural Similarity Index)  
- Deep feature similarity (AlexNet, Inception, CLIP, EfficientNet, SwAV)

The evaluation uses two-way identification tasks to measure how well reconstructed
images can be matched to their original counterparts based on feature similarity.

Reference:
- https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/Reconstruction_Metrics.ipynb
"""

import os
import argparse
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import pandas as pd
import scipy as sp

import warnings
warnings.filterwarnings("ignore")

from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    inception_v3,
    Inception_V3_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
import clip


# Configuration constants
@dataclass
class MetricConfig:
    """Configuration for metric evaluation"""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    # Model configurations
    ALEX_SIZE = 256
    INCEPTION_SIZE = 342
    CLIP_SIZE = 224
    EFFICIENT_SIZE = 255
    SWAV_SIZE = 224
    
    EPS = 1e-8  # Small epsilon for numerical stability

@torch.no_grad()
def compute_two_way_identification(
    reconstructed_images: torch.Tensor, 
    original_images: torch.Tensor, 
    model: nn.Module, 
    preprocess: transforms.Compose, 
    feature_layer: Optional[str] = None, 
    return_avg: bool = True, 
    device: str = "cpu"
) -> Tuple[float, int]:
    """
    Compute two-way identification accuracy between reconstructed and original images.
    
    This metric measures how well reconstructed images can be matched to their original
    counterparts based on feature similarity. Higher scores indicate better reconstruction quality.
    
    Args:
        reconstructed_images: Generated/reconstructed images [N, 3, H, W] in [0, 1]
        original_images: Ground truth images [N, 3, H, W] in [0, 1] 
        model: Pre-trained model for feature extraction
        preprocess: Preprocessing transforms for the model
        feature_layer: Specific layer to extract features from (if model returns dict)
        return_avg: Whether to return average score or per-image scores
        device: Device to run computation on
        
    Returns:
        If return_avg=True: (average_accuracy, total_comparisons)
        If return_avg=False: (per_image_scores, total_comparisons)
    """
    # Ensure all images are in [0, 1] range before preprocessing
    if reconstructed_images.max() > 1.0:
        reconstructed_images = reconstructed_images / 255.0
    if original_images.max() > 1.0:
        original_images = original_images / 255.0
        
    # Stack and preprocess images
    recons = torch.stack([preprocess(img) for img in reconstructed_images]).to(device)
    originals = torch.stack([preprocess(img) for img in original_images]).to(device)

    # Extract features
    recon_features = model(recons)
    original_features = model(originals)

    # Handle models that return dictionaries (e.g., feature extractors)
    if feature_layer is not None:
        recon_features = recon_features[feature_layer]
        original_features = original_features[feature_layer]

    # Flatten features to (N, D) for correlation computation
    recon_flat = recon_features.float().flatten(1).cpu().numpy()
    original_flat = original_features.float().flatten(1).cpu().numpy()

    # Compute correlation matrix between all reconstructions and originals
    correlation_matrix = np.corrcoef(original_flat, recon_flat)
    n_images = len(original_images)
    
    # Extract the cross-correlation block (originals vs reconstructions)
    cross_correlations = correlation_matrix[:n_images, n_images:]
    
    # Get diagonal elements (correct matches)
    correct_match_correlations = np.diag(cross_correlations)
    
    # For each original image, count how many reconstructed images have lower correlation with it
    # than the correct match (excluding the correct match itself)
    success_counts = np.zeros(n_images)
    for i in range(n_images):
        # Get correlations between this original and all reconstructions
        corrs = cross_correlations[i, :]
        # Correct match correlation
        correct_corr = corrs[i]
        # Count reconstructions with lower correlation (= successful identification)
        # We're counting entries where corr < correct_corr, so we want correct_corr to be high
        success_counts[i] = (corrs < correct_corr).sum()
    
    # Normalize by total comparisons (n_images - 1 for each image)
    total_comparisons = n_images - 1
    
    if return_avg:
        avg_success = success_counts.mean() / total_comparisons if total_comparisons > 0 else 0
        return avg_success
    else:
        return success_counts, total_comparisons


def compute_pixel_correlation(
    original_images: torch.Tensor, 
    reconstructed_images: torch.Tensor
) -> np.ndarray:
    """
    Compute pixel-wise Pearson correlation between original and reconstructed images.
    Based on reference implementation from meshconv-decoding.
    
    Args:
        original_images: Ground truth images [N, 3, H, W] in [0, 1]
        reconstructed_images: Generated images [N, 3, H, W] in [0, 1]
        
    Returns:
        Array of correlation scores for each image pair
    """
    # Preprocess: resize to standard size (425x425 as in reference)
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    # Apply preprocessing and flatten while keeping batch dimension
    orig_processed = preprocess(original_images).reshape(len(original_images), -1).cpu()
    recon_processed = preprocess(reconstructed_images).reshape(len(reconstructed_images), -1).cpu()
    
    # Compute correlation for each image pair
    correlations = []
    for i in range(len(original_images)):
        corr_matrix = np.corrcoef(orig_processed[i], recon_processed[i])
        correlation = corr_matrix[0, 1]
        
        # Handle NaN cases (when std is 0)
        if np.isnan(correlation):
            correlation = 0.0
            
        correlations.append(correlation)
    
    return np.array(correlations)


def compute_ssim_correlation(
    original_images: torch.Tensor, 
    reconstructed_images: torch.Tensor
) -> List[float]:
    """
    Compute SSIM (Structural Similarity Index) between original and reconstructed images.
    Based on reference implementation from meshconv-decoding.
    
    Args:
        original_images: Ground truth images [N, 3, H, W] in [0, 1]
        reconstructed_images: Generated images [N, 3, H, W] in [0, 1]
        
    Returns:
        List of SSIM scores for each image pair
    """
    
    # Preprocess: resize to standard size (425x425 as in reference)
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    # Convert to format for rgb2gray: (N, H, W, C)
    orig_processed = preprocess(original_images).permute(0, 2, 3, 1).cpu()
    recon_processed = preprocess(reconstructed_images).permute(0, 2, 3, 1).cpu()
    
    # Convert to grayscale
    img_gray = rgb2gray(orig_processed)
    recon_gray = rgb2gray(recon_processed)
    
    print("Converted to grayscale, now calculating SSIM...")
    
    # Compute SSIM for each image pair
    ssim_scores = []
    for orig_img, recon_img in zip(img_gray, recon_gray):
        try:
            ssim_score = ssim(
                recon_img, orig_img,  # Note: recon first, orig second as in reference
                multichannel=False,   # Grayscale images
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0
            )
            ssim_scores.append(ssim_score)
        except Exception as e:
            print(f"Warning: SSIM computation failed: {e}")
            ssim_scores.append(0.0)
    
    return ssim_scores


def create_model_preprocessor(
    backbone_fn: callable, 
    weights: object, 
    return_nodes: List[str], 
    input_size: int, 
    mean: List[float], 
    std: List[float], 
    device: str
) -> Tuple[nn.Module, transforms.Compose]:
    """
    Create a feature extraction model and its corresponding preprocessor.
    
    Args:
        backbone_fn: Function to create the backbone model
        weights: Pre-trained weights for the model
        return_nodes: List of layer names to extract features from
        input_size: Target input size for the model
        mean: Normalization mean values
        std: Normalization std values
        device: Device to load model on
        
    Returns:
        Tuple of (model, preprocessor)
    """
    model = create_feature_extractor(
        backbone_fn(weights=weights), 
        return_nodes=return_nodes
    ).to(device).eval().requires_grad_(False)
    
    # Include ToPILImage to ensure proper image format and resizing
    preprocessor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return model, preprocessor


def compute_feature_similarity(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor, 
    model: nn.Module, 
    preprocessor: transforms.Compose,
    feature_key: str,
    device: str
) -> np.ndarray:
    """
    Compute correlation-based feature similarity using a pre-trained model.
    
    Args:
        original_images: Ground truth images [N, 3, H, W]
        reconstructed_images: Generated images [N, 3, H, W]
        model: Feature extraction model
        preprocessor: Image preprocessing pipeline
        feature_key: Key to extract features from model output
        device: Device for computation
        
    Returns:
        Array of correlation scores for each image pair
    """
    # Extract features for original images
    orig_preprocessed = torch.stack([preprocessor(img) for img in original_images]).to(device)
    orig_features = model(orig_preprocessed)[feature_key].flatten(1).cpu().numpy()
    
    # Extract features for reconstructed images  
    recon_preprocessed = torch.stack([preprocessor(img) for img in reconstructed_images]).to(device)
    recon_features = model(recon_preprocessed)[feature_key].flatten(1).cpu().numpy()
    
    # Compute per-image correlations
    correlations = np.array([
        sp.spatial.distance.correlation(orig_features[i], recon_features[i]) 
        for i in range(len(orig_features))
    ])
    
    return correlations


class MetricEvaluator:
    """
    Comprehensive evaluator for brain-decoded image quality using multiple metrics.
    
    This class provides a clean interface for computing various image quality metrics
    including pixel correlation, SSIM, and deep feature similarities from multiple
    pre-trained models (AlexNet, Inception, CLIP, EfficientNet, SwAV).
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.config = MetricConfig()
    
    def compute_pixel_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Dict[str, pd.Series]:
        """Compute pixel-level metrics (correlation and SSIM)."""
        metrics = {}
        
        print(">>> Computing pixel correlation...")
        pixel_correlations = compute_pixel_correlation(original_images, reconstructed_images)
        metrics["PixCorr"] = pd.Series(pixel_correlations, name="PixCorr")
        
        print(">>> Computing SSIM correlation...")
        ssim_scores = compute_ssim_correlation(original_images, reconstructed_images)
        metrics["SSIM"] = pd.Series(ssim_scores, name="SSIM")
        
        return metrics
    
    def compute_alexnet_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Dict[str, pd.Series]:
        """Compute AlexNet feature-based metrics."""
        metrics = {}
        
        print(">>> Computing AlexNet features...")
        
        # AlexNet layer 2 -> features.4
        model, preprocessor = create_model_preprocessor(
            alexnet, AlexNet_Weights.IMAGENET1K_V1, ["features.4"],
            self.config.ALEX_SIZE, self.config.IMAGENET_MEAN, self.config.IMAGENET_STD, 
            self.device
        )
        
        # Use proper cross-correlation calculation as in the reference implementation
        scores = compute_two_way_identification(
            reconstructed_images, original_images, model, preprocessor,
            feature_layer="features.4", return_avg=False, device=self.device
        )
        # Scale to percentages and create series
        metrics["Alex(2)"] = pd.Series(scores[0] / scores[1] * 100, name="Alex(2)")

        model.cpu()
        del model

        # AlexNet layer 5 -> features.11
        model, preprocessor = create_model_preprocessor(
            alexnet, AlexNet_Weights.IMAGENET1K_V1, ["features.11"],
            self.config.ALEX_SIZE, self.config.IMAGENET_MEAN, self.config.IMAGENET_STD, 
            self.device
        )
        
        # Use proper cross-correlation calculation as in the reference implementation
        scores = compute_two_way_identification(
            reconstructed_images, original_images, model, preprocessor,
            feature_layer="features.11", return_avg=False, device=self.device
        )
        # Scale to percentages and create series
        metrics["Alex(5)"] = pd.Series(scores[0] / scores[1] * 100, name="Alex(5)")

        model.cpu()
        del model
        
        return metrics
    
    def compute_inception_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Dict[str, pd.Series]:
        """Compute Inception-v3 feature-based metrics."""
        print(">>> Computing Inception features...")
        
        model, preprocessor = create_model_preprocessor(
            inception_v3, Inception_V3_Weights.DEFAULT, ["avgpool"],
            self.config.INCEPTION_SIZE, self.config.IMAGENET_MEAN, self.config.IMAGENET_STD,
            self.device
        )
        
        # Use proper cross-correlation calculation as in the reference implementation
        scores = compute_two_way_identification(
            reconstructed_images, original_images, model, preprocessor,
            feature_layer="avgpool", return_avg=False, device=self.device
        )
        
        model.cpu()
        del model
        
        # Scale to percentages and create series
        return {"Incep": pd.Series(scores[0] / scores[1] * 100, name="Incep")}
    
    def compute_clip_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Dict[str, pd.Series]:
        """Compute CLIP feature-based metrics."""
        print(">>> Computing CLIP features...")
        
        clip_model, _ = clip.load("ViT-L/14", device=self.device)
        preprocessor = transforms.Compose([
            transforms.Resize(self.config.CLIP_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=self.config.CLIP_MEAN, std=self.config.CLIP_STD),
        ])
        
        scores, total_comparisons = compute_two_way_identification(
            reconstructed_images, original_images, clip_model.encode_image, preprocessor,
            feature_layer=None, return_avg=False, device=self.device
        )
        
        del clip_model
        
        return {"CLIP": pd.Series(scores / total_comparisons * 100, name="CLIP")}
    
    def compute_efficientnet_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Dict[str, pd.Series]:
        """Compute EfficientNet feature-based metrics."""
        print(">>> Computing EfficientNet features...")
        
        model, preprocessor = create_model_preprocessor(
            efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, ["avgpool"],
            self.config.EFFICIENT_SIZE, 
            EfficientNet_B1_Weights.DEFAULT.meta.get("mean", self.config.IMAGENET_MEAN),
            EfficientNet_B1_Weights.DEFAULT.meta.get("std", self.config.IMAGENET_STD),
            self.device
        )
        
        correlations = compute_feature_similarity(
            original_images, reconstructed_images, model, preprocessor,
            "avgpool", self.device
        )
        
        model.cpu()
        del model
        
        return {"Eff": pd.Series(correlations, name="Eff")}
    
    def compute_swav_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Dict[str, pd.Series]:
        """Compute SwAV feature-based metrics."""
        print(">>> Computing SwAV features...")
        
        swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(self.device).eval().requires_grad_(False)
        
        preprocessor = transforms.Compose([
            transforms.Resize(self.config.SWAV_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD),
        ])
        
        correlations = compute_feature_similarity(
            original_images, reconstructed_images, model, preprocessor,
            "avgpool", self.device
        )
        
        model.cpu()
        del model
        
        return {"SwAV": pd.Series(correlations, name="SwAV")}
    
    def compute_all_metrics(
        self, 
        original_images: torch.Tensor, 
        reconstructed_images: torch.Tensor
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all available metrics for image quality evaluation.
        
        Args:
            original_images: Ground truth images [N, 3, H, W] in [0, 1]
            reconstructed_images: Generated images [N, 3, H, W] in [0, 1]
            
        Returns:
            Tuple of (detailed_results_df, summary_stats_df)
        """
        print(f"Evaluating {len(original_images)} image pairs...")
        
        all_metrics = {}
        
        # Compute all metric categories
        all_metrics.update(self.compute_pixel_metrics(original_images, reconstructed_images))
        all_metrics.update(self.compute_alexnet_metrics(original_images, reconstructed_images))
        all_metrics.update(self.compute_inception_metrics(original_images, reconstructed_images))
        all_metrics.update(self.compute_clip_metrics(original_images, reconstructed_images))
        all_metrics.update(self.compute_efficientnet_metrics(original_images, reconstructed_images))
        all_metrics.update(self.compute_swav_metrics(original_images, reconstructed_images))
        
        # Combine results
        results_df = pd.concat(list(all_metrics.values()), axis=1)
        summary_df = pd.DataFrame({
            "mean": results_df.mean(),
            "std": results_df.std()
        })
        
        print("\n=== Metric Summary ===")
        print(summary_df)
        
        return results_df, summary_df


def load_and_validate_data(results_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and validate image data from brain adapter results directory.
    
    Args:
        results_dir: Path to the results directory containing individual .npz files
        
    Returns:
        Tuple of (original_images, predicted_images) as tensors
        
    Raises:
        FileNotFoundError: If the results directory doesn't exist
        ValueError: If the data format is invalid
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Loading data from: {results_dir}")
    
    # Load metadata to get sample information
    metadata_path = os.path.join(results_dir, "evaluation_metadata.json")
    summary_path = os.path.join(results_dir, "sample_summary.json")
    
    if not os.path.exists(metadata_path) or not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing metadata files in {results_dir}")
    
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Found {summary['num_samples']} samples in {summary['mode']} evaluation")
    
    # Load all individual sample files
    dataset_indices = summary['dataset_indices']
    original_images_list = []
    predicted_images_list = []
    
    for dataset_idx in tqdm(dataset_indices):
        sample_filename = f"sample_{dataset_idx:06d}.npz"
        sample_path = os.path.join(results_dir, sample_filename)
        
        if not os.path.exists(sample_path):
            print(f"Warning: Missing sample file {sample_filename}")
            continue
        
        # Load sample data
        sample_data = np.load(sample_path, allow_pickle=True)
        
        # Get ground truth image
        gt_image = torch.tensor(sample_data['groundtruth_image'])
        original_images_list.append(gt_image)
        
        # Get predicted image(s)
        if 'candidate_images' in sample_data and 'correlation_scores' in sample_data:
            # Multiple candidates available - select best correlated one
            candidates = sample_data['candidate_images']
            scores = sample_data['correlation_scores']
            best_idx = np.argmax(scores)
            
            print(f"Sample {dataset_idx}: Using candidate {best_idx} with correlation {scores[best_idx]:.4f}")
            pred_image = torch.tensor(candidates[best_idx])
        else:
            # Single prediction available
            pred_image = torch.tensor(sample_data['predicted_image'])
        
        predicted_images_list.append(pred_image)
        sample_data.close()
    
    if not original_images_list or not predicted_images_list:
        raise ValueError("No valid samples found in the results directory")
    
    # Stack into tensors
    original_images = torch.stack(original_images_list)
    predicted_images = torch.stack(predicted_images_list)
    
    # Convert to proper format if needed
    if original_images.dtype == torch.uint8:
        original_images = original_images.float() / 255.0
    if predicted_images.dtype == torch.uint8:
        predicted_images = predicted_images.float() / 255.0
    
    # Ensure proper dimension order [N, C, H, W]
    if original_images.dim() == 4 and original_images.shape[-1] == 3:
        original_images = original_images.permute(0, 3, 1, 2)
    if predicted_images.dim() == 4 and predicted_images.shape[-1] == 3:
        predicted_images = predicted_images.permute(0, 3, 1, 2)
    
    # Validate data shapes and types
    print(f"Original images: {original_images.shape}, dtype: {original_images.dtype}")
    print(f"Value range: [{original_images.min():.3f}, {original_images.max():.3f}]")
    print(f"Predicted images: {predicted_images.shape}, dtype: {predicted_images.dtype}")
    print(f"Value range: [{predicted_images.min():.3f}, {predicted_images.max():.3f}]")
    
    if len(original_images.shape) != 4:
        raise ValueError(f"Expected 4D tensors [N,C,H,W], got shape {original_images.shape}")
    if original_images.shape != predicted_images.shape:
        predicted_images = resize_images(predicted_images, original_images.shape[2:])

    return original_images, predicted_images

def resize_images(
    images: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Resize images to the target size.

    Args:
        images: Input image tensor
        target_size: Target size as (height, width)

    Returns:
        Resized image tensor
    """
    return torch.nn.functional.interpolate(images, size=target_size, mode="bilinear", align_corners=False)


def save_results(
    results_df: pd.DataFrame, 
    summary_df: pd.DataFrame, 
    results_dir: str,
    evaluation_mode: str = "subset"
) -> None:
    """
    Save evaluation results to JSON files.
    
    Args:
        results_df: Detailed per-image results
        summary_df: Summary statistics
        results_dir: Results directory path
        evaluation_mode: "subset" or "full" to determine filename
    """
    import json
    
    # Determine output filename based on evaluation mode
    output_filename = f"metric_{evaluation_mode}.json"
    output_path = os.path.join(results_dir, output_filename)
    
    # Convert DataFrames to dictionaries for JSON serialization
    per_image_results = results_df.to_dict('records')
    summary_stats = summary_df.to_dict('index')
    
    # Create comprehensive results structure
    results_data = {
        "evaluation_metadata": {
            "evaluation_mode": evaluation_mode,
            "num_samples": len(results_df),
            "metric_types": list(results_df.columns),
            "timestamp": pd.Timestamp.now().isoformat()
        },
        "summary_statistics": summary_stats,
        "per_image_metrics": per_image_results,
        "metric_descriptions": {
            "PixCorr": "Pixel-wise Pearson correlation",
            "SSIM": "Structural Similarity Index correlation", 
            "Alex(2)": "AlexNet layer 2 two-way identification accuracy (%)",
            "Alex(5)": "AlexNet layer 5 two-way identification accuracy (%)",
            "Incep": "Inception-v3 avgpool two-way identification accuracy (%)",
            "CLIP": "CLIP ViT-L/14 two-way identification accuracy (%)",
            "Eff": "EfficientNet-B1 avgpool feature correlation",
            "SwAV": "SwAV ResNet-50 avgpool feature correlation"
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

def convert_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images to uint8 format, handling both float [0,1] and uint8 [0,255] inputs.
    
    Args:
        images: Input image tensor
        
    Returns:
        Images converted to uint8 format
    """
    if images.dtype in [torch.float32, torch.float64]:
        # Assume float images are in [0,1] range
        images = images.clamp(0, 1) * 255
    return images.to(torch.uint8)


def create_comparison_grid(
    original_images: torch.Tensor, 
    predicted_images: torch.Tensor, 
    save_path: str, 
    grid_width: int = 8
) -> None:
    """
    Create and save a comparison grid of original vs predicted images.
    
    Args:
        original_images: Original images tensor
        predicted_images: Predicted images tensor  
        save_path: Path to save the grid image
        grid_width: Number of images per row in the grid
    """
    from torchvision.utils import make_grid
    from PIL import Image
    
    # Convert to uint8 and move to CPU
    orig_uint8 = convert_to_uint8(original_images.cpu())
    pred_uint8 = convert_to_uint8(predicted_images.cpu())
    
    # Concatenate original and predicted images
    combined_images = torch.cat([orig_uint8, pred_uint8], dim=0)
    
    # Create grid
    grid = make_grid(combined_images, nrow=grid_width, padding=2)
    
    # Convert to PIL and save
    grid_numpy = grid.permute(1, 2, 0).numpy()
    grid_image = Image.fromarray(grid_numpy)
    grid_image.save(save_path)
    
    print(f"Comparison grid saved: {save_path}")


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Construct results directory path
    if args.results_dir is not None:
        results_dir = args.results_dir
        print(f"Using provided results directory: {results_dir}")
        model_name = args.results_dir.split("/")[-3]
    else:
        model_name = args.model_weights_dir.split("/")[-1]
        results_dir = os.path.join(
            args.decoded_stimuli_dir, 
            model_name, 
            f"epoch_{args.saved_epochs}", 
            args.evaluation_mode
        )
        print(f"Auto-constructed results directory: {results_dir}")
    
    print("=== Brain Adapter Metrics Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Epoch: {args.saved_epochs}")
    print(f"Evaluation mode: {args.evaluation_mode}")
    print(f"Device: {device}")
    print(f"Results directory: {results_dir}")
    
    # Load and validate data
    try:
        original_images, predicted_images = load_and_validate_data(results_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Initialize evaluator and compute metrics
    evaluator = MetricEvaluator(device=device)

    try:
        results_df, summary_df = evaluator.compute_all_metrics(
            original_images, predicted_images
        )
    except Exception as e:
        print(f"Error during metric computation: {e}")
        return 1
    
    # Save results
    save_results(results_df, summary_df, results_dir, args.evaluation_mode)
    
    # Create and save comparison visualization (optional)
    if args.create_visualization:
        comparison_grid_path = os.path.join(results_dir, f"metric_comparison_grid.png")
        try:
            create_comparison_grid(
                original_images, predicted_images, comparison_grid_path
            )
        except Exception as e:
            print(f"Warning: Could not create comparison grid: {e}")
    
    print("\n=== Evaluation completed successfully ===")
    return 0

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate brain-decoded image quality using comprehensive metrics from brain adapter results"
    )
    parser.add_argument(
        "--model_weights_dir", 
        type=str, 
        default="brain_adapter/model_weights/07_26_2025-22_29",
        help="Directory containing model weights (used to determine model name)"
    )
    parser.add_argument(
        "--decoded_stimuli_dir", 
        type=str, 
        default="brain_adapter/decoded_stimuli",
        help="Base directory containing decoded stimuli results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Direct path to results directory (overrides automatic path construction)"
    )
    parser.add_argument(
        "--saved_epochs",
        type=str,
        default="100", 
        help="Epoch number to evaluate (e.g., '100')"
    )
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        choices=["subset", "full", "illusion"],
        default="subset",
        help="Evaluation mode: 'subset', 'full', or 'illusion' dataset"
    )
    parser.add_argument(
        "--create_visualization",
        action="store_true",
        default=False,
        help="Create comparison grid visualization"
    )
    return parser

if __name__ == "__main__":
    """
    Usage Examples:

    # Evaluate subset results from epoch 100:
    python metric_brain_adapter.py \
        --model_weights_dir brain_adapter/decoded_stimuli/09_25_2025-14_54 \
        --saved_epochs 500 \
        --evaluation_mode illusion
    
    # Evaluate full dataset results from epoch 200:
    python metric_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/08_16_2025-18_03 \
        --saved_epochs 200 \
        --evaluation_mode full
    
    # With custom directories and visualization:
    python metric_brain_adapter.py \
        --model_weights_dir /path/to/model \
        --decoded_stimuli_dir /path/to/results \
        --saved_epochs 100 \
        --evaluation_mode subset \
        --create_visualization
    
    # Expected directory structure:
    # brain_adapter/decoded_stimuli/
    # └── 07_26_2025-22_29/
    #     └── epoch_100/
    #         ├── subset/
    #         │   ├── sample_000001.npz
    #         │   ├── sample_000002.npz
    #         │   ├── evaluation_metadata.json
    #         │   ├── sample_summary.json
    #         │   └── metric_subset.json  (output)
    #         └── full/
    #             ├── sample_000001.npz
    #             ├── sample_000002.npz
    #             ├── evaluation_metadata.json
    #             ├── sample_summary.json
    #             └── metric_full.json  (output)
    """
    import sys
    exit_code = main()
    sys.exit(exit_code)
