"""
DRCT Architecture Module
========================
DRCT-L: Dense Residual Connected Transformer for Super-Resolution.
Paper:  https://arxiv.org/abs/2404.00722  (CVPR 2024 NTIRE 3rd place)
Repo:   https://github.com/ming053l/DRCT

Pretrained weights: pretrained/drct/DRCT-L_X4.pth
Feature hook target: self.conv_after_body → [B, 180, H, W]
"""

import sys
import types


def _setup_basicsr_mocks():
    """
    Mock basicsr modules so drct_arch.py can be imported without basicsr installed.
    DRCT needs: basicsr.utils.registry, basicsr.archs.arch_util
    """
    from timm.models.layers import to_2tuple, trunc_normal_

    class MockRegistry:
        """Absorbs @ARCH_REGISTRY.register() decorators silently."""
        def __init__(self):
            self._obj_map = {}

        def register(self, name=None):
            def decorator(cls):
                return cls
            return decorator

        def get(self, name):
            return self._obj_map.get(name)

    if 'basicsr' not in sys.modules:
        sys.modules['basicsr'] = types.ModuleType('basicsr')

    if 'basicsr.utils' not in sys.modules:
        sys.modules['basicsr.utils'] = types.ModuleType('basicsr.utils')

    if 'basicsr.utils.registry' not in sys.modules:
        registry_module = types.ModuleType('basicsr.utils.registry')
        registry_module.ARCH_REGISTRY = MockRegistry()
        sys.modules['basicsr.utils.registry'] = registry_module
    else:
        sys.modules['basicsr.utils.registry'].ARCH_REGISTRY = MockRegistry()

    if 'basicsr.archs' not in sys.modules:
        sys.modules['basicsr.archs'] = types.ModuleType('basicsr.archs')

    if 'basicsr.archs.arch_util' not in sys.modules:
        arch_util = types.ModuleType('basicsr.archs.arch_util')
        arch_util.to_2tuple = to_2tuple
        arch_util.trunc_normal_ = trunc_normal_
        sys.modules['basicsr.archs.arch_util'] = arch_util

    if 'basicsr.ops' not in sys.modules:
        sys.modules['basicsr.ops'] = types.ModuleType('basicsr.ops')


# Run mocks immediately at import time
_setup_basicsr_mocks()

# ============================================================================
# Import DRCT class
# ============================================================================

DRCT_AVAILABLE = False
DRCT = None

try:
    from .drct_arch import DRCT
    DRCT_AVAILABLE = True
    print("✓ DRCT arch loaded successfully")
except ImportError as e:
    print(f"⚠  DRCT arch not found: {e}")
except Exception as e:
    print(f"✗  DRCT arch failed to load: {e}")
    import traceback
    traceback.print_exc()

__all__ = ['DRCT', 'DRCT_AVAILABLE', 'create_drct_model']


def create_drct_model(
    upscale:      int   = 4,
    img_size:     int   = 64,
    window_size:  int   = 16,
    embed_dim:    int   = 180,
    depths:       list  = None,
    num_heads:    list  = None,
    img_range:    float = 1.0,
    upsampler:    str   = 'pixelshuffle',
    resi_connection: str = '1conv',
):
    """
    Create DRCT-L model configured for ×4 SR.
    Config EXACTLY matches DRCT-L_X4.pth pretrained weights.

    Returns:
        DRCT model instance (untrained, not frozen)
    """
    if not DRCT_AVAILABLE:
        raise ImportError(
            "DRCT architecture file missing.\n"
            "Run: Copy-Item \"$env:TEMP\\drct\\drct\\archs\\drct_arch.py\" "
            "\"src\\models\\drct\\drct_arch.py\""
        )

    if depths is None:
        depths = [6] * 12       # DRCT-L: 12 stages × 6 blocks

    if num_heads is None:
        num_heads = [6] * 12    # 6 attention heads per stage

    model = DRCT(
        upscale          = upscale,
        in_chans         = 3,
        img_size         = img_size,
        window_size      = window_size,
        compress_ratio   = 3,
        squeeze_factor   = 30,
        conv_scale       = 0.01,
        overlap_ratio    = 0.5,
        img_range        = img_range,
        depths           = depths,
        embed_dim        = embed_dim,
        num_heads        = num_heads,
        mlp_ratio        = 2,
        upsampler        = upsampler,
        resi_connection  = resi_connection,
    )

    return model
