import torch
import torch.nn.functional as F
from typing import Optional

def min_snr_loss_weights(timesteps, noise_scheduler, gamma=5.0):
    """
    Compute Min-SNR loss weights for given timesteps.
    
    Args:
        timesteps: Current timesteps in the batch
        noise_scheduler: The noise scheduler (e.g., DDPMScheduler)
        gamma: Min-SNR gamma parameter (typically 5.0)
        
    Returns:
        Loss weights for each timestep
    """
    # Get SNR for each timestep
    alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps]
    eps = 1e-8
    snr = alphas_cumprod / (1 - alphas_cumprod + eps)

    # Min-SNR weighting: min(snr, gamma) / snr
    min_snr_weights = torch.minimum(snr, torch.ones_like(snr) * gamma) / (snr + eps)

    return min_snr_weights


def dispersive_loss(features, temperature=1.0, eps=1e-8):
    B, N, D = features.shape
    features_norm = F.normalize(features, p=2, dim=-1, eps=eps)
    sim_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, N, N]
    mask = torch.eye(N, device=features.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
    # Loss: encourage low similarity (diversity)
    loss = torch.logsumexp(sim_matrix / temperature, dim=2).mean()
    return loss
