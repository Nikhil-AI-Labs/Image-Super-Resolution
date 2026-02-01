"""
HAT Architecture Module
=======================
Provides HAT model for super-resolution.
Wraps the official XPixelGroup/HAT implementation.

Sets up basicsr mocks before importing to avoid dependency issues.
"""
import sys
import types

# ============================================================================
# Setup basicsr mocks BEFORE importing hat_arch
# ============================================================================

def _setup_basicsr_mocks():
    """Setup basicsr mock modules to avoid import errors."""
    from timm.models.layers import to_2tuple, trunc_normal_
    
    class MockRegistry:
        def __init__(self):
            self._obj_map = {}
        
        def register(self, name=None):
            def decorator(cls):
                return cls
            return decorator
        
        def get(self, name):
            return self._obj_map.get(name)
    
    # Create mock modules if they don't exist
    if 'basicsr' not in sys.modules:
        sys.modules['basicsr'] = types.ModuleType('basicsr')
    
    if 'basicsr.utils' not in sys.modules:
        sys.modules['basicsr.utils'] = types.ModuleType('basicsr.utils')
    
    if 'basicsr.utils.registry' not in sys.modules:
        registry_module = types.ModuleType('basicsr.utils.registry')
        registry_module.ARCH_REGISTRY = MockRegistry()
        sys.modules['basicsr.utils.registry'] = registry_module
    
    if 'basicsr.archs' not in sys.modules:
        sys.modules['basicsr.archs'] = types.ModuleType('basicsr.archs')
    
    if 'basicsr.archs.arch_util' not in sys.modules:
        arch_util = types.ModuleType('basicsr.archs.arch_util')
        arch_util.to_2tuple = to_2tuple
        arch_util.trunc_normal_ = trunc_normal_
        sys.modules['basicsr.archs.arch_util'] = arch_util


# Setup mocks immediately on import
_setup_basicsr_mocks()

# Now we can safely import HAT
from .hat_arch import HAT

__all__ = ['HAT', 'create_hat_model']


def create_hat_model(
    embed_dim: int = 180,
    depths: list = None,
    num_heads: list = None,
    window_size: int = 16,
    upscale: int = 4,
    img_range: float = 1.0
):
    """
    Create an official HAT-L model with correct architecture for pretrained weights.
    
    HAT-L configuration (NTIRE 2025 winner):
    - embed_dim: 180
    - depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6] (12 stages, 6 blocks each)
    - num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    - window_size: 16
    - compress_ratio: 3
    - squeeze_factor: 30
    - conv_scale: 0.01
    - overlap_ratio: 0.5 (for OCAB)
    - mlp_ratio: 2
    
    Args:
        embed_dim: Embedding dimension (default 180 for HAT-L)
        depths: Number of blocks per stage
        num_heads: Number of attention heads per stage
        window_size: Window size for local attention
        upscale: Upscaling factor (4 for 4x SR)
        img_range: Image range (1.0 for normalized [0,1])
    
    Returns:
        HAT model instance
    """
    if depths is None:
        depths = [6] * 12
    if num_heads is None:
        num_heads = [6] * 12
    
    model = HAT(
        upscale=upscale,
        in_chans=3,
        img_size=64,  # Not used for inference
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    return model
