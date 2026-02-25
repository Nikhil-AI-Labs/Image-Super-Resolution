"""
EDSR Architecture Module
========================
EDSR-L: Enhanced Deep Super-Resolution (CVPRW 2017).
Standalone — no basicsr dependency, no mocks needed.

Pretrained weights: pretrained/edsr/EDSR_Lx4_f256b32_DIV2K.pth
Feature hook target: self.conv_after_body → [B, 256, H, W]
"""

from .edsr_arch import EDSR, ResBlock, Upsample

EDSR_AVAILABLE = True

__all__ = ['EDSR', 'EDSR_AVAILABLE', 'create_edsr_model']


def create_edsr_model(
    num_feat:   int   = 256,
    num_block:  int   = 32,
    upscale:    int   = 4,
    res_scale:  float = 0.1,
    img_range:  float = 255.0,
) -> EDSR:
    """
    Create EDSR-L model for ×4 SR. Config matches EDSR_Lx4_f256b32_DIV2K.pth.

    ⚠ img_range=255.0 is MANDATORY — pretrained weights expect 255x internal scaling.
    """
    return EDSR(
        num_in_ch  = 3,
        num_out_ch = 3,
        num_feat   = num_feat,
        num_block  = num_block,
        upscale    = upscale,
        res_scale  = res_scale,
        img_range  = img_range,
    )
