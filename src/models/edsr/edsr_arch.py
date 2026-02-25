"""
EDSR-L Standalone Architecture
================================
EDSR: Enhanced Deep Super-Resolution Network.
Paper:  https://arxiv.org/abs/1707.02921  (CVPRW 2017)
Weights: EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth (BasicSR release)

Standalone implementation — zero basicsr dependency.
Parameter names EXACTLY match BasicSR's EDSR for weight compatibility.

Architecture: EDSR-L
    num_feat   = 256, num_block  = 32, res_scale  = 0.1
    img_range  = 255.0 (MUST be 255.0 to match pretrained weights)

Feature hook target:
    self.conv_after_body → [B, 256, H, W]
"""

import math
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block for EDSR.
    Structure: Conv → ReLU → Conv → ×res_scale → Add input
    Parameter names match BasicSR exactly: self.conv1, self.conv2, self.relu
    """
    def __init__(self, num_feat: int = 256, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.res_scale


class Upsample(nn.Sequential):
    """
    Pixel-shuffle upsampler.
    For scale=4: two ×2 pixel-shuffle stages.
    """
    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:     # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale factor {scale}.")
        super().__init__(*m)


class EDSR(nn.Module):
    """
    EDSR-L: Enhanced Deep Super-Resolution Network.

    CRITICAL — img_range:
        Official weights trained with img_range=255.0.
        Model scales [0,1] input ×255 internally, then back to [0,1].
        Setting img_range=1.0 produces completely wrong outputs.

    Parameter names match BasicSR for weight loading:
        conv_first, body.N.conv1, body.N.conv2, conv_after_body,
        upsample.0/1/2/3, conv_last
    """
    def __init__(
        self,
        num_in_ch:  int   = 3,
        num_out_ch: int   = 3,
        num_feat:   int   = 256,
        num_block:  int   = 32,
        upscale:    int   = 4,
        res_scale:  float = 0.1,
        img_range:  float = 255.0,
        rgb_mean:   tuple = (0.4488, 0.4371, 0.4040),
    ):
        super().__init__()
        self.img_range = img_range
        self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResBlock(num_feat, res_scale) for _ in range(num_block)]
        )
        # HOOK TARGET → [B, num_feat, H, W]
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input [B,3,H,W] in [0,1] → SR output in [0,1]."""
        self.mean = self.mean.to(x.device)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res = res + x   # Global residual connection

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        return x
