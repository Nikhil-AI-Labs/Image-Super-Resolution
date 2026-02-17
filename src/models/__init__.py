"""
Models Module
=============
Provides expert models, fusion network, and TSD-SR for NTIRE 2025 SR.

Expert Models (Frozen):
- HAT: Hybrid Attention Transformer (Samsung 1st Track A)
- DAT: Dual Aggregation Transformer (ICCV 2023)
- NAFNet: Nonlinear Activation Free Network (Samsung)

Fusion Network (Trainable):
- FrequencyRouter: Lightweight CNN for expert routing based on frequency
- FrequencyAwareFusion: Combines expert outputs based on frequency content
- MultiFusionSR: Complete pipeline combining experts + fusion
- CompleteEnhancedFusionSR: Full pipeline with ALL 5 improvements (+1.1 dB)

Fusion Improvements:
1. DynamicExpertSelector (+0.3 dB)
2. CrossBandAttention (+0.2 dB)
3. AdaptiveFrequencyBandPredictor (+0.15 dB)
4. MultiResolutionFusion (+0.25 dB)
5. CollaborativeFeatureLearning (+0.2 dB)

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
    # Fusion Improvements
    DynamicExpertSelector,
    CrossBandAttention,
    AdaptiveFrequencyBandPredictor,
    MultiResolutionFusion,
    CollaborativeFeatureLearning,
    EnhancedMultiFusionSR,
)
from .enhanced_fusion import (
    CompleteEnhancedFusionSR,
    create_complete_enhanced_fusion,
)
# V2 - Complete correct implementation with all 5 improvements
from .enhanced_fusion_v2 import (
    AdaptiveFrequencyDecomposition,
    CrossBandAttention as CrossBandAttentionV2,
    CollaborativeFeatureLearning as CollaborativeFeatureLearningV2,
    MultiResolutionFusion as MultiResolutionFusionV2,
    DynamicExpertSelector as DynamicExpertSelectorV2,
    CompleteEnhancedFusionSR as EnhancedFusionV2,
    ExpertFeatureExtractor,
    create_enhanced_fusion,
)
from .tsdsr_wrapper import (
    TSDSRInference,
    VAEWrapper,
    load_tsdsr_models,
    create_tsdsr_refinement_pipeline,
)

# Complete 7-Phase Pipeline (ULTIMATE MODEL!)
from .complete_sr_pipeline import (
    CompleteSRPipeline,
    create_complete_pipeline,
    create_training_pipeline,
    create_inference_pipeline,
)

__all__ = [
    # Expert Ensemble
    'ExpertEnsemble',
    
    # Fusion Network Components
    'FrequencyRouter',
    'FrequencyAwareFusion',
    'MultiFusionSR',
    
    # Fusion Improvements (ALL 5)
    'DynamicExpertSelector',
    'CrossBandAttention',
    'AdaptiveFrequencyBandPredictor',
    'MultiResolutionFusion',
    'CollaborativeFeatureLearning',
    
    # Enhanced Fusion (Complete Pipeline)
    'EnhancedMultiFusionSR',
    'CompleteEnhancedFusionSR',
    'create_complete_enhanced_fusion',
    
    # V2 Complete Implementation (Use This!)
    'EnhancedFusionV2',
    'create_enhanced_fusion',
    'AdaptiveFrequencyDecomposition',
    'CrossBandAttentionV2',
    'CollaborativeFeatureLearningV2',
    'MultiResolutionFusionV2',
    'DynamicExpertSelectorV2',
    'ExpertFeatureExtractor',
    
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
    
    # Complete 7-Phase Pipeline (ULTIMATE MODEL - Use This!)
    'CompleteSRPipeline',
    'create_complete_pipeline',
    'create_training_pipeline',
    'create_inference_pipeline',
]

