"""
Loss Functions Module
====================
Championship-level loss functions for NTIRE 2025 Super-Resolution.

Available Losses:
- L1Loss, L2Loss, CharbonnierLoss: Basic pixel-wise losses
- SSIMLoss: Structural similarity loss
- VGGPerceptualLoss: Feature matching with VGG19
- FFTLoss: Frequency domain loss
- SWTLoss: Samsung's secret (Stationary Wavelet Transform)
- EdgeLoss: Gradient-based edge preservation
- CLIPPerceptualLoss: SNUCV's technique (Track B, 0.5 threshold)
- CombinedLoss: Multi-stage training with all losses

Expected PSNR boost: +0.8-1.0 dB with proper combination!
"""

from .perceptual_loss import (
    # Basic losses
    L1Loss,
    L2Loss,
    CharbonnierLoss,
    # Structural loss
    SSIMLoss,
    # Perceptual losses
    VGGPerceptualLoss,
    VGGFeatureExtractor,
    # Frequency losses
    FFTLoss,
    SWTLoss,
    # Edge loss
    EdgeLoss,
    # CLIP loss (Track B)
    CLIPPerceptualLoss,
    # Combined loss
    CombinedLoss,
    # Availability flags
    PYWT_AVAILABLE,
    LPIPS_AVAILABLE,
    CLIP_AVAILABLE,
)

__all__ = [
    # Basic losses
    'L1Loss',
    'L2Loss',
    'CharbonnierLoss',
    # Structural loss
    'SSIMLoss',
    # Perceptual losses
    'VGGPerceptualLoss',
    'VGGFeatureExtractor',
    # Frequency losses
    'FFTLoss',
    'SWTLoss',
    # Edge loss
    'EdgeLoss',
    # CLIP loss
    'CLIPPerceptualLoss',
    # Combined loss
    'CombinedLoss',
    # Availability flags
    'PYWT_AVAILABLE',
    'LPIPS_AVAILABLE',
    'CLIP_AVAILABLE',
]
