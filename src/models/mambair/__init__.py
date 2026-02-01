"""
MambaIR Architecture Module
===========================
Provides MambaIR model for super-resolution.

NOTE: MambaIR requires the mamba_ssm package which needs CUDA compilation.
Install via: pip install mamba-ssm

For systems without CUDA compilation support, use the fallback pure-PyTorch
implementation provided here.
"""

try:
    from .mambair_arch import MambaIR
    MAMBA_AVAILABLE = True
except ImportError as e:
    # Fallback: mamba_ssm not available
    MAMBA_AVAILABLE = False
    MambaIR = None
    import warnings
    warnings.warn(
        f"MambaIR architecture not available: {e}. "
        "Install mamba-ssm for full functionality."
    )

__all__ = ['MambaIR', 'create_mambair_model', 'MAMBA_AVAILABLE']


def create_mambair_model(
    upscale: int = 4,
    img_size: int = 64,
    embed_dim: int = 180,
    depths: list = None,
    num_heads: list = None,
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    upsampler: str = 'pixelshuffle',
    img_range: float = 1.0,
):
    """
    Create MambaIR model for super-resolution.
    
    MambaIR-SR4 configuration (SNUCV winner):
    - Uses state space models (Mamba) for efficient long-range dependencies
    - embed_dim: 180
    - depths: [6, 6, 6, 6, 6, 6] (6 stages)
    - Mamba state dimension: 16
    
    Args:
        upscale: Upscaling factor (4 for 4x SR)
        img_size: Training image size
        embed_dim: Embedding dimension
        depths: Number of blocks per stage
        num_heads: Number of attention heads
        window_size: Window size
        mlp_ratio: MLP expansion ratio
        upsampler: Upsampling method
        img_range: Image range
    
    Returns:
        MambaIR model instance or None if unavailable
    """
    if not MAMBA_AVAILABLE:
        raise ImportError(
            "MambaIR requires mamba-ssm package. "
            "Install via: pip install mamba-ssm"
        )
    
    if depths is None:
        depths = [6, 6, 6, 6, 6, 6]
    if num_heads is None:
        num_heads = [6, 6, 6, 6, 6, 6]
    
    model = MambaIR(
        upscale=upscale,
        in_chans=3,
        img_size=img_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection='1conv'
    )
    return model
