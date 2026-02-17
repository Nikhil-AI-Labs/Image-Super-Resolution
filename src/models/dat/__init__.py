"""
DAT - Dual Aggregation Transformer Module
==========================================
ICCV 2023 Paper implementation for image super-resolution.

Key exports:
- DAT: Main model class
- create_dat_model: Factory function with default config
- create_dat_s: DAT-S (standard) configuration
- create_dat_light: Lightweight version
- create_dat_2: Larger version

Usage:
    from src.models.dat import create_dat_model, DAT
    
    model = create_dat_model(upscale=4)
    sr_image = model(lr_image)  # [B,3,64,64] -> [B,3,256,256]
"""

from .dat_arch import (
    DAT,
    create_dat_model,
    create_dat_s,
    create_dat_light,
    create_dat_2,
)

# Mark DAT as available (no special CUDA requirements like MambaIR)
DAT_AVAILABLE = True

__all__ = [
    'DAT',
    'DAT_AVAILABLE',
    'create_dat_model',
    'create_dat_s',
    'create_dat_light',
    'create_dat_2',
]
