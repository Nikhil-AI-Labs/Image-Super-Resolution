"""
NAFNet Architecture Module
==========================
Provides NAFNet model for image restoration, adapted for SR.

NOTE: NAFNet is designed for denoising/deblurring, not super-resolution.
For SR tasks, we wrap it with a pixel shuffle upsampler.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import core NAFNet components from standalone implementation
from .nafnet_arch import NAFNet, NAFBlock, SimpleGate, LayerNorm2d

__all__ = ['NAFNet', 'NAFNetSR', 'NAFBlock', 'create_nafnet_sr_model']


class NAFNetSR(nn.Module):
    """
    NAFNet adapted for Super-Resolution.
    
    The original NAFNet is designed for denoising (same input/output size).
    This wrapper adds:
    1. Initial feature extraction
    2. NAFNet-style processing with NAFBlocks
    3. Pixel shuffle upsampling for SR
    
    This is similar to Samsung's approach in NTIRE 2025 where they
    used NAFNet blocks as part of their fusion network.
    """
    
    def __init__(
        self,
        upscale: int = 4,
        img_channel: int = 3,
        width: int = 64,
        middle_blk_num: int = 12,
        enc_blk_nums: list = None,
        dec_blk_nums: list = None,
    ):
        """
        Args:
            upscale: Upscaling factor
            img_channel: Number of input channels
            width: Base channel width
            middle_blk_num: Number of middle NAF blocks
            enc_blk_nums: Encoder block counts (not used in SR variant)
            dec_blk_nums: Decoder block counts (not used in SR variant)
        """
        super().__init__()
        
        self.upscale = upscale
        self.img_range = 1.0
        
        # Initial feature extraction
        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        # NAF blocks for feature processing
        self.body = nn.Sequential(
            *[NAFBlock(width) for _ in range(middle_blk_num)]
        )
        
        # Residual connection conv
        self.conv_after_body = nn.Conv2d(width, width, 3, 1, 1)
        
        # Upsampling with pixel shuffle
        upsample_layers = []
        if upscale == 4:
            upsample_layers.extend([
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        elif upscale == 2:
            upsample_layers.extend([
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        else:
            raise ValueError(f"Upscale {upscale} not supported")
        
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv_last = nn.Conv2d(width, img_channel, 3, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            SR image [B, 3, H*upscale, W*upscale]
        """
        # Bicubic upsampling for residual connection
        base = F.interpolate(
            x, 
            scale_factor=self.upscale, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Feature extraction and processing
        feat = self.intro(x)
        feat = self.conv_after_body(self.body(feat)) + feat
        
        # Upsampling
        feat = self.upsample(feat)
        out = self.conv_last(feat)
        
        # Residual connection
        return out + base


def create_nafnet_sr_model(
    upscale: int = 4,
    width: int = 64,
    middle_blk_num: int = 12,
):
    """
    Create NAFNet-SR model for super-resolution.
    
    NAFNet-SIDD-width64 configuration (Samsung NTIRE 2025):
    - width: 64
    - middle_blk_num: 12 (based on SIDD config)
    - Uses simple gate and simplified channel attention
    
    Args:
        upscale: Upscaling factor (4 for 4x SR)
        width: Base channel width
        middle_blk_num: Number of NAF blocks in middle
    
    Returns:
        NAFNetSR model instance
    """
    model = NAFNetSR(
        upscale=upscale,
        img_channel=3,
        width=width,
        middle_blk_num=middle_blk_num
    )
    return model
