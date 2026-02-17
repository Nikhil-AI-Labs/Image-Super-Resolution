"""
NAFNet Architecture Module
==========================
Provides NAFNet model for image restoration, adapted for SR.

NOTE: NAFNet-SIDD is designed for denoising (same input/output size).
For SR tasks, we wrap it with bicubic upscaling + refinement.

NAFNet-SIDD-width64 architecture:
- UNet-style with enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]
- middle_blk_num=12
- width=64
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Import core NAFNet components from standalone implementation
from .nafnet_arch import NAFNet, NAFBlock, SimpleGate, LayerNorm2d

__all__ = ['NAFNet', 'NAFNetSR', 'NAFBlock', 'create_nafnet_sr_model']


class NAFNetSR(nn.Module):
    """
    NAFNet-SIDD adapted for Super-Resolution.
    
    Uses the original NAFNet UNet architecture (compatible with SIDD weights)
    and applies it to bicubic-upscaled images for refinement.
    
    Approach: Bicubic upscale -> NAFNet refinement (denoising/enhancement)
    
    This allows loading the official NAFNet-SIDD-width64.pth weights directly.
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
            enc_blk_nums: Encoder block counts (SIDD default: [2,2,4,8])
            dec_blk_nums: Decoder block counts (SIDD default: [2,2,2,2])
        """
        super().__init__()
        
        self.upscale = upscale
        self.img_range = 1.0
        
        # Default to NAFNet-SIDD configuration
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]  # Official SIDD config
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]  # Official SIDD config
        
        # Use the original NAFNet UNet architecture (compatible with SIDD weights)
        self.nafnet = NAFNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums
        )
        
        # Expose key attributes for feature extraction hooks
        self.intro = self.nafnet.intro       # [B, 64, H, W] output
        self.ending = self.nafnet.ending     # Input is [B, 64, H, W] (the final feature map!)
        self.middle_blks = self.nafnet.middle_blks
        
        # Body attribute for hook compatibility (points to middle_blks)
        # WARNING: middle_blks outputs 1024 channels (bottleneck), NOT 64!
        self.body = self.nafnet.middle_blks
    
    def load_nafnet_weights(self, state_dict: dict) -> dict:
        """
        Load weights from NAFNet-SIDD checkpoint into the nafnet backbone.
        
        Args:
            state_dict: Checkpoint state dict
            
        Returns:
            Info dict with loading statistics
        """
        nafnet_state = self.nafnet.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in nafnet_state:
                if value.shape == nafnet_state[key].shape:
                    nafnet_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key}: shape {value.shape} vs {nafnet_state[key].shape}")
            else:
                skipped_keys.append(f"{key}: not in model")
        
        self.nafnet.load_state_dict(nafnet_state, strict=False)
        
        return {
            'loaded': len(loaded_keys),
            'skipped': len(skipped_keys),
            'total': len(nafnet_state),
            'skipped_keys': skipped_keys[:5] if skipped_keys else []
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with bicubic upscaling + NAFNet refinement.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            SR image [B, 3, H*upscale, W*upscale]
        """
        # Step 1: Bicubic upscale to target resolution
        x_up = F.interpolate(
            x, 
            scale_factor=self.upscale, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Step 2: Apply NAFNet for refinement/enhancement
        # NAFNet outputs residual, so result = input + residual
        out = self.nafnet(x_up)
        
        return out.clamp(0, 1)


def create_nafnet_sr_model(
    upscale: int = 4,
    width: int = 64,
    middle_blk_num: int = 12,
    enc_blk_nums: list = None,
    dec_blk_nums: list = None,
):
    """
    Create NAFNet-SR model for super-resolution.
    
    NAFNet-SIDD-width64 official configuration:
    - width: 64
    - enc_blk_nums: [2, 2, 4, 8]
    - middle_blk_num: 12
    - dec_blk_nums: [2, 2, 2, 2]
    
    Args:
        upscale: Upscaling factor (4 for 4x SR)
        width: Base channel width
        middle_blk_num: Number of NAF blocks in middle
        enc_blk_nums: Encoder block counts
        dec_blk_nums: Decoder block counts
    
    Returns:
        NAFNetSR model instance
    """
    # Use SIDD defaults if not specified
    if enc_blk_nums is None:
        enc_blk_nums = [2, 2, 4, 8]
    if dec_blk_nums is None:
        dec_blk_nums = [2, 2, 2, 2]
        
    model = NAFNetSR(
        upscale=upscale,
        img_channel=3,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )
    return model
