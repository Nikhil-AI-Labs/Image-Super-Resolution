"""
GRL Architecture Module
=======================
GRL-B: Graph Representation Learning for Image Restoration.
Paper: https://arxiv.org/abs/2303.00748  (CVPR 2023)
Repo:  https://github.com/ofsoundof/GRL-Image-Restoration

Pretrained weights: pretrained/grl/GRL-B_SR_x4.pth
Feature hook target: self.conv_after_body → [B, 180, H, W]

Config source: grl_arch.py __main__ "Large, 20.13 M" block (lines 741-758)
    embed_dim=180, depths=[4,4,8,8,8,4,4], num_heads=[3]*7, local_connection=True

Note: GRL uses fairscale (activation checkpointing) and omegaconf (config).
fairscale is mocked here since checkpointing is not needed for frozen inference.
omegaconf must be installed: pip install omegaconf
"""

import sys
import types
import logging


def _setup_grl_mocks():
    """
    Mock fairscale for GRL. GRL only uses fairscale.nn.checkpoint_wrapper
    for activation checkpointing during training — not needed for inference.
    """
    # ── Mock fairscale ────────────────────────────────────────────────────
    if 'fairscale' not in sys.modules:
        fairscale_mod = types.ModuleType('fairscale')
        sys.modules['fairscale'] = fairscale_mod

    if 'fairscale.nn' not in sys.modules:
        fairscale_nn = types.ModuleType('fairscale.nn')

        def _checkpoint_wrapper_noop(module, **kwargs):
            """No-op: return module unchanged (skip checkpointing)."""
            return module

        fairscale_nn.checkpoint_wrapper = _checkpoint_wrapper_noop
        sys.modules['fairscale.nn'] = fairscale_nn

        # Also set attribute on parent
        sys.modules['fairscale'].nn = fairscale_nn


# Run mocks immediately at import time
_setup_grl_mocks()

# ============================================================================
# Import GRL class
# ============================================================================

GRL_AVAILABLE = False
GRL = None

try:
    from .grl_arch import GRL
    GRL_AVAILABLE = True
    print("✓ GRL arch loaded successfully")
except ImportError as e:
    print(f"⚠  GRL arch not found: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗  GRL arch failed to load: {e}")
    import traceback
    traceback.print_exc()

__all__ = ['GRL', 'GRL_AVAILABLE', 'create_grl_model']


def create_grl_model(
    upscale:                   int   = 4,
    img_size:                  int   = 64,
    window_size:               int   = 8,
    # ── CORRECTED: embed_dim=180 not 128, 7 stages not 6 ──────────────────
    embed_dim:                 int   = 180,
    depths:                    list  = None,
    num_heads_w:               list  = None,
    num_heads_s:               list  = None,
    stripe_size:               list  = None,
    img_range:                 float = 1.0,
    anchor_one_stage:          bool  = True,
    # ── CORRECTED: missing params that exist in GRL-B_SR_x4.pth ──────────
    local_connection:          bool  = True,     # checkpoint has these weights (6M params)
    anchor_window_down_factor: int   = 2,        # controls anchor resolution
    qkv_proj_type:             str   = 'linear',
    out_proj_type:             str   = 'linear',
    conv_type:                 str   = '1conv',
    mlp_ratio:                 float = 2.0,
):
    """
    Create GRL-B model (20.13M params) matching GRL-B_SR_x4.pth.

    Config from grl_arch.py __main__ "Large, 20.13 M" + checkpoint inspection:
        embed_dim=180, depths=[4,4,8,8,8,4,4], num_heads_w/s=[3]*7
        local_connection=True, anchor_window_down_factor=2

    CHANGED FROM ORIGINAL (was creating wrong 5M model):
        embed_dim:        128 → 180
        depths:           [4]*6 → [4,4,8,8,8,4,4]  (7 stages not 6)
        num_heads:        [4]*6 → [3]*7
        local_connection: False → True  (GRL-B requires this)

    Hook target: self.conv_after_body = build_last_conv('1conv', 180)
                 → output shape: [B, 180, H, W]  (was [B, 128, H, W] before fix)
    """
    if not GRL_AVAILABLE:
        raise ImportError(
            "GRL architecture not available. Check error messages above."
        )

    if depths is None:
        depths = [4, 4, 8, 8, 8, 4, 4]       # 7 stages, NOT 6

    if num_heads_w is None:
        num_heads_w = [3, 3, 3, 3, 3, 3, 3]   # 3 heads per stage, NOT 4

    if num_heads_s is None:
        num_heads_s = [3, 3, 3, 3, 3, 3, 3]   # matches num_heads_w

    if stripe_size is None:
        stripe_size = [8, 8]                   # GRL default from constructor

    model = GRL(
        upscale                   = upscale,
        img_size                  = img_size,
        window_size               = window_size,
        img_range                 = img_range,
        depths                    = depths,
        embed_dim                 = embed_dim,
        num_heads_window          = num_heads_w,
        num_heads_stripe          = num_heads_s,
        stripe_size               = stripe_size,
        stripe_groups             = [None, None],
        stripe_shift              = False,
        mlp_ratio                 = mlp_ratio,
        qkv_bias                  = True,
        qkv_proj_type             = qkv_proj_type,
        anchor_proj_type          = 'avgpool',
        anchor_one_stage          = anchor_one_stage,
        anchor_window_down_factor = anchor_window_down_factor,
        out_proj_type             = out_proj_type,
        local_connection          = local_connection,   # ← KEY FIX
        upsampler                 = 'pixelshuffle',
        conv_type                 = conv_type,
        init_method               = 'n',
        fairscale_checkpoint      = False,
    )

    return model
