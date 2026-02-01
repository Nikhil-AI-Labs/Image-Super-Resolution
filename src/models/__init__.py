"""
Models Module
=============
Provides expert models, fusion network, and TSD-SR for NTIRE 2025 SR.

Expert Models (Frozen):
- HAT: Hybrid Attention Transformer (Samsung 1st Track A)
- MambaIR: Mamba-based Image Restoration (SNUCV 1st Track B)
- NAFNet: Nonlinear Activation Free Network (Samsung)

Fusion Network (Trainable):
- FrequencyRouter: Lightweight CNN for expert routing based on frequency
- FrequencyAwareFusion: Combines expert outputs based on frequency content
- MultiFusionSR: Complete pipeline combining experts + fusion

TSD-SR (Diffusion Refinement):
- TSDSRInference: One-step/multi-step diffusion refinement
- VAEWrapper: Latent space encoder/decoder
"""

from .expert_loader import ExpertEnsemble
from .fusion_network import (
    FrequencyRouter,
    FrequencyAwareFusion,
    MultiFusionSR,
    ChannelAttention,
    SpatialAttention,
    ChannelSpatialAttention,
    MultiScaleFeatureExtractor,
)
from .tsdsr_wrapper import (
    TSDSRInference,
    VAEWrapper,
    load_tsdsr_models,
    create_tsdsr_refinement_pipeline,
)

__all__ = [
    # Expert Ensemble
    'ExpertEnsemble',
    
    # Fusion Network Components
    'FrequencyRouter',
    'FrequencyAwareFusion',
    'MultiFusionSR',
    
    # Attention Modules
    'ChannelAttention',
    'SpatialAttention',
    'ChannelSpatialAttention',
    'MultiScaleFeatureExtractor',
    
    # TSD-SR Refinement
    'TSDSRInference',
    'VAEWrapper',
    'load_tsdsr_models',
    'create_tsdsr_refinement_pipeline',
]
