"""
TSD-SR Architecture Module
==========================
Implementation of TSD-SR (Target Score Distillation Super-Resolution).

Components:
- DiT: Diffusion Transformer backbone
- TSDSRDiT: One-step refinement model

Usage:
    from src.models.tsdsr import create_tsdsr_dit_base
    model = create_tsdsr_dit_base()
    
Reference: https://github.com/Microtreei/TSD-SR
"""

from .dit import (
    DiT,
    DiTBlock,
    TSDSRDiT,
    create_tsdsr_dit_small,
    create_tsdsr_dit_base,
    create_tsdsr_dit_large,
)

__all__ = [
    'DiT',
    'DiTBlock',
    'TSDSRDiT',
    'create_tsdsr_dit_small',
    'create_tsdsr_dit_base',
    'create_tsdsr_dit_large',
]
