"""
Brain Adapter Metric Class

This module provides a clean interface for computing image similarity metrics
between ground truth and brain-decoded images. It's designed to be compatible
with metric_brain_adapter.py while providing a simpler API.

Metrics implemented:
- Pixel correlation
- SSIM (Structural Similarity)
- AlexNet layer 2 & 5
- Inception
- CLIP
- EfficientNet
- SwAV
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
import scipy as sp
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Union, List, Tuple, Optional
from dataclasses import dataclass

# Import models
from torchvision.models import (
    alexnet, AlexNet_Weights,
    inception_v3, Inception_V3_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
)
import clip

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


class BrainDecodingMetrics:
    """
    Comprehensive metric evaluator for brain-decoded images.
    
    This class provides methods to compute various image similarity metrics
    between ground truth images and brain-decoded reconstructions.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the metrics evaluator.
        
        Args:
            device: Device to use for computation (default: CUDA if available)
        """
        self.device = device
        self.config = MetricConfig()
        self.models = {}  # Cache for loaded models
        
    def _ensure_tensor_format(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Ensure image is a properly formatted tensor.
        
        Args:
            img: Input image as numpy array or tensor
            
        Returns:
            Properly formatted tensor [C, H, W] in range [0, 1]
        """
        if isinstance(img, np.ndarray):
            # Handle different input formats
            if len(img.shape) == 3 and img.shape[-1] in [1, 3, 4]:
                img = torch.from_numpy(img).permute(2, 0, 1).float()
            else:
                img = torch.from_numpy(img).float()
                
            # Ensure values are in [0, 1] range
            if img.max() > 1.0:
                img = img / 255.0
                
        elif isinstance(img, torch.Tensor):
            # Ensure tensor is in [0, 1] range
            if img.max() > 1.0:
                img = img / 255.0
                
        return img.to('cpu')  # Move to CPU to avoid unnecessary GPU memory usage

    def _extract_single_image_features(self, img: Union[np.ndarray, torch.Tensor], model, preprocess_fn, feature_layer: Optional[str] = None) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            img: Input image
            model: Feature extraction model
            preprocess_fn: Preprocessing function
            feature_layer: Name of feature layer (if model returns dict)
            
        Returns:
            Feature array as numpy array
        """
        img_tensor = self._ensure_tensor_format(img)
        processed = preprocess_fn(img_tensor).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            if feature_layer is not None:
                features = model(processed)[feature_layer].flatten().cpu().numpy()
            else:
                features = model(processed).flatten().cpu().numpy()
        
        # Clean up immediately
        del processed, img_tensor
        torch.cuda.empty_cache()
        
        return features

    @torch.no_grad()
    def _two_way_identification_set_level(
        self, 
        gt_imgs_sets: List[List[Union[np.ndarray, torch.Tensor]]], 
        decoded_imgs_sets: List[List[Union[np.ndarray, torch.Tensor]]], 
        model, 
        preprocess_fn, 
        feature_layer: Optional[str] = None
    ) -> float:
        """
        Compute set-level two-way identification using averaged similarities.
        
        For each GT, we have K reconstructions. We compute correlations between 
        the GT features and each reconstruction's features, then average over the K 
        reconstructions to get one score per (GT, recon-set) pairing. Finally, we 
        compute how often the true set beats each other set for the same GT.
        
        Args:
            gt_imgs_sets: List of ground truth images for each condition (N conditions, 1 GT each)
            decoded_imgs_sets: List of sets of decoded images for each condition (N conditions, K recons each)
            model: Feature extraction model
            preprocess_fn: Preprocessing function for images
            feature_layer: Name of feature layer (if model returns dict)
            
        Returns:
            Two-way identification accuracy in [0,1]
        """
        n_conditions = len(gt_imgs_sets)
        if n_conditions != len(decoded_imgs_sets):
            raise ValueError("Number of GT and decoded sets must match")
        
        if n_conditions < 2:
            print("Warning: Need at least 2 conditions for two-way identification")
            return 0.0
        
        print(f"Processing {n_conditions} conditions for set-level 2WC...")
        
        # Extract features for all GT images (N conditions, 1 GT each)
        gt_features_list = []
        for condition_idx, gt_imgs in enumerate(gt_imgs_sets):
            if len(gt_imgs) == 0:
                print(f"Warning: No GT images for condition {condition_idx}")
                return 0.0
            
            # Take the first (and typically only) GT image
            gt_img = gt_imgs[0]
            try:
                features = self._extract_single_image_features(gt_img, model, preprocess_fn, feature_layer)
                gt_features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to extract GT features for condition {condition_idx}: {e}")
                return 0.0
        
        # Extract features for all decoded image sets (N conditions, K recons each)
        decoded_features_by_condition = []
        for condition_idx, decoded_imgs in enumerate(decoded_imgs_sets):
            if len(decoded_imgs) == 0:
                print(f"Warning: No decoded images for condition {condition_idx}")
                return 0.0
                
            condition_features = []
            for img_idx, img in enumerate(decoded_imgs):
                try:
                    features = self._extract_single_image_features(img, model, preprocess_fn, feature_layer)
                    condition_features.append(features)
                except Exception as e:
                    print(f"Warning: Failed to extract decoded features for condition {condition_idx}, image {img_idx}: {e}")
                    continue
            
            if len(condition_features) == 0:
                print(f"Warning: No valid decoded features for condition {condition_idx}")
                return 0.0
                
            decoded_features_by_condition.append(condition_features)
        
        # Compute set-level averaged similarities
        # avg_similarities[i][j] = average correlation between GT_i and recon_set_j
        avg_similarities = np.zeros((n_conditions, n_conditions))
        
        for i in range(n_conditions):  # For each GT
            gt_features_i = gt_features_list[i]
            
            for j in range(n_conditions):  # For each recon set
                decoded_features_j = decoded_features_by_condition[j]
                
                # Compute correlations between GT_i and all reconstructions in set_j
                correlations = []
                for decoded_feat in decoded_features_j:
                    # Ensure features have same length and are valid
                    if len(gt_features_i) == len(decoded_feat) and not (np.isnan(gt_features_i).any() or np.isnan(decoded_feat).any()):
                        # Use Pearson correlation for similarity
                        if np.std(gt_features_i) > 1e-8 and np.std(decoded_feat) > 1e-8:
                            corr = np.corrcoef(gt_features_i, decoded_feat)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                # Average correlation between GT_i and recon_set_j
                if correlations:
                    avg_similarities[i, j] = np.mean(correlations)
                else:
                    raise ValueError(f"Invalid similarity between GT_{i} and recon_set_{j}")
                    # avg_similarities[i, j] = -1.0  # Invalid similarity
                    
        
        # Apply pairwise 2WC logic using averaged similarities
        # Extract diagonal (own similarities) and apply vectorized comparison
        congruents = np.diag(avg_similarities)
        
        # Check if each GT's own similarity is greater than all others
        # success[i, j] = True if avg_similarities[i, j] < congruents[i] (own similarity beats other)
        success = avg_similarities < congruents[:, None]
        
        # Count successes per GT (excluding diagonal comparison with itself)
        success_cnt = np.sum(success, axis=1)    # Subtract 1 to exclude self-comparison
        
        # Calculate performance: average success rate
        perf = float(np.mean(success_cnt) / (n_conditions - 1))
        
        # Ensure accuracy is in [0, 1] range
        perf = max(0.0, min(1.0, perf))
        assert 0.0 <= perf <= 1.0, f"2WC accuracy {perf} not in [0,1] range"
        
        print(f"  Set-level two-way identification: success_counts={success_cnt}, perf={perf:.3f}")
        
        # Clean up feature data to free memory
        del gt_features_list, decoded_features_by_condition, avg_similarities
        torch.cuda.empty_cache()
        
        return perf
        
        

    def compute_pixel_correlation(self, gt_img: Union[np.ndarray, torch.Tensor], decoded_img: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute pixel-wise correlation between ground truth and decoded image.
        
        Args:
            gt_img: Ground truth image
            decoded_img: Decoded/reconstructed image
            
        Returns:
            Pearson correlation coefficient
        """
        gt_img = self._ensure_tensor_format(gt_img)
        decoded_img = self._ensure_tensor_format(decoded_img)
        
        # Preprocess: resize to standard size (425x425 as in reference)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((425, 425), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        gt_img_processed = preprocess(gt_img)
        decoded_img_processed = preprocess(decoded_img)
        
        # Flatten and compute correlation
        gt_flat = gt_img_processed.flatten().cpu().numpy()
        decoded_flat = decoded_img_processed.flatten().cpu().numpy()
        
        # Clean up tensors
        del gt_img_processed, decoded_img_processed
        
        # Ensure both arrays have the same size
        assert gt_flat.shape == decoded_flat.shape, f"Shape mismatch: {gt_flat.shape} vs {decoded_flat.shape}"
        
        correlation = np.corrcoef(gt_flat, decoded_flat)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def compute_ssim(self, gt_img: Union[np.ndarray, torch.Tensor], decoded_img: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute SSIM between ground truth and decoded image.
        
        Args:
            gt_img: Ground truth image
            decoded_img: Decoded/reconstructed image
            
        Returns:
            SSIM score
        """
        gt_img = self._ensure_tensor_format(gt_img)
        decoded_img = self._ensure_tensor_format(decoded_img)
        
        # Preprocess: resize to standard size (425x425 as in reference)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((425, 425), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        gt_img_processed = preprocess(gt_img)
        decoded_img_processed = preprocess(decoded_img)
        
        # Convert to format for rgb2gray: (H, W, C)
        gt_img_hwc = gt_img_processed.permute(1, 2, 0).cpu().numpy()
        decoded_img_hwc = decoded_img_processed.permute(1, 2, 0).cpu().numpy()
        
        # Convert to grayscale
        gt_gray = rgb2gray(gt_img_hwc)
        decoded_gray = rgb2gray(decoded_img_hwc)
        
        # Compute SSIM (recon first, orig second as in reference)
        ssim_val = ssim(
            decoded_gray, gt_gray,  # Note: recon first, orig second as in reference
            multichannel=False,  # Grayscale images
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0
        )
        
        return ssim_val


    def compute_efficientnet(self, gt_img: Union[np.ndarray, torch.Tensor], decoded_img: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute EfficientNet feature correlation.
        
        Args:
            gt_img: Ground truth image
            decoded_img: Decoded/reconstructed image
            
        Returns:
            Feature correlation score
        """
        gt_img = self._ensure_tensor_format(gt_img)
        decoded_img = self._ensure_tensor_format(decoded_img)
        
        # Load model if not cached
        if 'efficientnet' not in self.models:
            self.models['efficientnet'] = create_feature_extractor(
                efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
                return_nodes=["avgpool"]
            ).to(self.device).eval()
            
        model = self.models['efficientnet']
        
        # Preprocessing function
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.EFFICIENT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=EfficientNet_B1_Weights.DEFAULT.meta.get("mean", self.config.IMAGENET_MEAN),
                std=EfficientNet_B1_Weights.DEFAULT.meta.get("std", self.config.IMAGENET_STD)
            )
        ])
        
        # Calculate feature correlation
        with torch.no_grad():
            gt_processed = preprocess(gt_img).unsqueeze(0).to(self.device)
            decoded_processed = preprocess(decoded_img).unsqueeze(0).to(self.device)
            
            gt_features = model(gt_processed)["avgpool"].flatten().cpu().numpy()
            decoded_features = model(decoded_processed)["avgpool"].flatten().cpu().numpy()
            
            # Compute spatial distance correlation
            correlation = sp.spatial.distance.correlation(gt_features, decoded_features)
            
        return correlation

    def compute_swav(self, gt_img: Union[np.ndarray, torch.Tensor], decoded_img: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute SwAV feature correlation.
        
        Args:
            gt_img: Ground truth image
            decoded_img: Decoded/reconstructed image
            
        Returns:
            Feature correlation score
        """
        gt_img = self._ensure_tensor_format(gt_img)
        decoded_img = self._ensure_tensor_format(decoded_img)
        
        # Load model if not cached
        if 'swav' not in self.models:
            swav_base = torch.hub.load("facebookresearch/swav:main", "resnet50")
            self.models['swav'] = create_feature_extractor(
                swav_base, return_nodes=["avgpool"]
            ).to(self.device).eval()
            
        model = self.models['swav']
        
        # Preprocessing function
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.SWAV_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD)
        ])
        
        # Calculate feature correlation
        with torch.no_grad():
            gt_processed = preprocess(gt_img).unsqueeze(0).to(self.device)
            decoded_processed = preprocess(decoded_img).unsqueeze(0).to(self.device)
            
            gt_features = model(gt_processed)["avgpool"].flatten().cpu().numpy()
            decoded_features = model(decoded_processed)["avgpool"].flatten().cpu().numpy()
            
            # Compute spatial distance correlation
            correlation = sp.spatial.distance.correlation(gt_features, decoded_features)
            
        return correlation
    
    def compute_metric(self, metric_name: str, gt_img: Union[np.ndarray, torch.Tensor], decoded_img: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute a specified metric between ground truth and decoded image.
        
        Args:
            metric_name: Name of the metric to compute
            gt_img: Ground truth image
            decoded_img: Decoded/reconstructed image
            
        Returns:
            Metric score
            
        Raises:
            ValueError: If metric_name is not recognized
        """
        # Map metric names to compute functions
        metric_functions = {
            'pixCorr': self.compute_pixel_correlation,
            'SSIM': self.compute_ssim,
            'alexnet_2': self.compute_alexnet_2,
            'alexnet_5': self.compute_alexnet_5,
            'inception': self.compute_inception,
            'clip': self.compute_clip,
            'efficientnet': self.compute_efficientnet,
            'swav': self.compute_swav
        }
        
        if metric_name not in metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(metric_functions.keys())}")
            
        return metric_functions[metric_name](gt_img, decoded_img)
    

    def compute_all_two_way_identifications(
        self, 
        gt_imgs_sets: List[List[Union[np.ndarray, torch.Tensor]]], 
        decoded_imgs_sets: List[List[Union[np.ndarray, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        Compute two-way identification for all relevant models.
        
        Args:
            gt_imgs_sets: List of ground truth images for each condition
            decoded_imgs_sets: List of sets of decoded images for each condition
            
        Returns:
            Dictionary mapping model names to two-way identification accuracies
        """
        results = {}
        
        # AlexNet Layer 2
        print("Computing 2WC for AlexNet Layer 2...")
        try:
            if 'alexnet' not in self.models:
                self.models['alexnet'] = create_feature_extractor(
                    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
                    return_nodes=["features.4", "features.11"]
                ).to(self.device).eval()
            
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.ALEX_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD)
            ])
            
            results['alexnet_2_2wc'] = self._two_way_identification_set_level(
                gt_imgs_sets, decoded_imgs_sets, self.models['alexnet'], preprocess, "features.4"
            ) * 100  # Convert to percentage
            
        except Exception as e:
            print(f"Error computing AlexNet Layer 2 2WC: {e}")
            results['alexnet_2_2wc'] = 0.0
        
        # Clear cache between models
        torch.cuda.empty_cache()
        
        # AlexNet Layer 5
        print("Computing 2WC for AlexNet Layer 5...")
        try:
            if 'alexnet' not in self.models:
                self.models['alexnet'] = create_feature_extractor(
                    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
                    return_nodes=["features.4", "features.11"]
                ).to(self.device).eval()
            
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.ALEX_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD)
            ])
            
            results['alexnet_5_2wc'] = self._two_way_identification_set_level(
                gt_imgs_sets, decoded_imgs_sets, self.models['alexnet'], preprocess, "features.11"
            ) * 100  # Convert to percentage
            
        except Exception as e:
            print(f"Error computing AlexNet Layer 5 2WC: {e}")
            results['alexnet_5_2wc'] = 0.0
        
        # Clean up AlexNet model
        if 'alexnet' in self.models:
            model = self.models.pop('alexnet')
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        torch.cuda.empty_cache()
        
        # Inception
        print("Computing 2WC for Inception...")
        try:
            if 'inception' not in self.models:
                self.models['inception'] = create_feature_extractor(
                    inception_v3(weights=Inception_V3_Weights.DEFAULT),
                    return_nodes=["avgpool"]
                ).to(self.device).eval()
            
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.INCEPTION_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD)
            ])
            
            results['inception_2wc'] = self._two_way_identification_set_level(
                gt_imgs_sets, decoded_imgs_sets, self.models['inception'], preprocess, "avgpool"
            ) * 100  # Convert to percentage
            
        except Exception as e:
            print(f"Error computing Inception 2WC: {e}")
            results['inception_2wc'] = 0.0
        
        # Clean up Inception model
        if 'inception' in self.models:
            model = self.models.pop('inception')
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        torch.cuda.empty_cache()
        
        # CLIP
        print("Computing 2WC for CLIP...")
        try:
            if 'clip' not in self.models:
                self.models['clip'], _ = clip.load("ViT-L/14", device=self.device)
            
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.CLIP_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.CLIP_MEAN, std=self.config.CLIP_STD)
            ])
            
            results['clip_2wc'] = self._two_way_identification_set_level(
                gt_imgs_sets, decoded_imgs_sets, self.models['clip'].encode_image, preprocess, None
            ) * 100  # Convert to percentage
            
        except Exception as e:
            print(f"Error computing CLIP 2WC: {e}")
            results['clip_2wc'] = 0.0
        
        # Clean up CLIP model
        if 'clip' in self.models:
            model = self.models.pop('clip')
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        torch.cuda.empty_cache()
        
        return results
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        for model_name in list(self.models.keys()):
            model = self.models[model_name]
            if hasattr(model, 'cpu'):
                model.cpu()
            del self.models[model_name]
        
        self.models = {}
        torch.cuda.empty_cache()
        
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.clear_cache()
        torch.cuda.empty_cache()


class MetricsEvaluator:
    """
    Wrapper class for easy integration of BrainDecodingMetrics into the decoding pipeline.
    
    This class provides a simplified interface for computing metrics between ground truth
    and decoded images, handling device management and memory optimization.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the metrics evaluator.
        
        Args:
            device: Device to use for computation
        """
        self.device = device
        self.metrics_engine = BrainDecodingMetrics(device=device)
    
    def evaluate_all_metrics(self, decoded_img: Union[np.ndarray, torch.Tensor], gt_img: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate all pairwise metrics between decoded and ground truth image.
        
        Args:
            decoded_img: Decoded/reconstructed image
            gt_img: Ground truth image
            
        Returns:
            Dictionary containing all metric scores
        """
        # Compute pairwise metrics (pixCorr, SSIM, EfficientNet, SwAV)
        metrics = {}
        
        try:
            # Pixel correlation
            metrics['pixCorr'] = self.metrics_engine.compute_pixel_correlation(gt_img, decoded_img)
            
            # SSIM 
            metrics['SSIM'] = self.metrics_engine.compute_ssim(gt_img, decoded_img)
            
            # EfficientNet-B1 (cosine similarity)
            metrics['efficientnet'] = self.metrics_engine.compute_efficientnet(gt_img, decoded_img)
            
            # Clean up EfficientNet model after use
            if 'efficientnet' in self.metrics_engine.models:
                model = self.metrics_engine.models.pop('efficientnet')
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
                torch.cuda.empty_cache()
            
            # SwAV-R50 (cosine similarity)
            metrics['swav'] = self.metrics_engine.compute_swav(gt_img, decoded_img)
            
            # Clean up SwAV model after use
            if 'swav' in self.metrics_engine.models:
                model = self.metrics_engine.models.pop('swav')
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Warning: Error computing metrics: {e}")
            # Set default values on error
            for metric_name in ['pixCorr', 'SSIM', 'efficientnet', 'swav']:
                if metric_name not in metrics:
                    metrics[metric_name] = 0.0
        
        return metrics
    
    def compute_set_level_2wc(
        self, 
        gt_imgs_sets: List[List[Union[np.ndarray, torch.Tensor]]], 
        decoded_imgs_sets: List[List[Union[np.ndarray, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        Compute set-level two-way identification for AlexNet, Inception, and CLIP.
        
        Args:
            gt_imgs_sets: List of ground truth images for each condition
            decoded_imgs_sets: List of sets of decoded images for each condition
            
        Returns:
            Dictionary mapping model names to 2WC scores (percentages)
        """
        return self.metrics_engine.compute_all_two_way_identifications(gt_imgs_sets, decoded_imgs_sets)
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        self.metrics_engine.clear_cache()