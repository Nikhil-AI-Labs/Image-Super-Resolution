"""
NTIRE 2025 Super-Resolution Training Script (Colab Edition)
==============================================================
Complete training script for the frequency-aware fusion network.

REQUIRED PACKAGES (install in Colab):
    !pip install torch torchvision einops timm pywt opencv-python tqdm

OPTIONAL (for MambaIR):
    !pip install mamba-ssm causal-conv1d

USAGE:
    1. Upload entire Image-Super-Resolution folder to Google Drive
    2. Mount drive: from google.colab import drive; drive.mount('/content/drive')
    3. Change to project dir: %cd /content/drive/MyDrive/Image-Super-Resolution
    4. Run: !python colab_trainable.py

Author: NTIRE SR Team
"""

import os
import sys
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import cv2

# Optional imports
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets not installed. SWT Loss unavailable. Run: pip install PyWavelets")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration for Google Colab."""
    
    # Paths (adjust for your Google Drive)
    DATASET_ROOT = "/content/drive/MyDrive/datasets/DF2K"
    PRETRAINED_ROOT = "/content/drive/MyDrive/Image-Super-Resolution/checkpoints/pretrained_weights"
    OUTPUT_DIR = "/content/drive/MyDrive/Image-Super-Resolution/outputs"
    
    # Expert weight filenames
    HAT_WEIGHTS = "HAT-L_SRx4_ImageNet-pretrain.pth"
    MAMBAIR_WEIGHTS = "MambaIR_SR4_x4.pth"
    NAFNET_WEIGHTS = "NAFNet-SIDD-width64.pth"
    
    # Training settings
    SCALE = 4
    BATCH_SIZE = 16  # Reduced for memory safety on T4
    NUM_WORKERS = 2
    LR_PATCH_SIZE = 64
    TOTAL_EPOCHS = 200
    WARMUP_EPOCHS = 5
    ACCUMULATION_STEPS = 2  # Effective batch = BATCH_SIZE * ACCUMULATION_STEPS
    
    # Learning rate
    INITIAL_LR = 2e-4
    MIN_LR = 1e-7
    
    # Model settings
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # Loss stages: (start_epoch, loss_config)
    LOSS_STAGES = [
        (0, {'l1': 1.0}),
        (100, {'l1': 0.8, 'swt': 0.15, 'fft': 0.05}),
        (150, {'l1': 0.7, 'swt': 0.15, 'fft': 0.05, 'ssim': 0.1}),
    ]
    
    # Validation
    VALIDATE_EVERY = 5
    SAVE_EVERY = 10
    CROP_BORDER = 4
    TEST_Y_CHANNEL = True
    
    # Dataset
    REPEAT_FACTOR = 20
    
    @classmethod
    def get_checkpoint_dir(cls): 
        return os.path.join(cls.OUTPUT_DIR, "checkpoints")
    
    @classmethod
    def get_log_dir(cls):
        return os.path.join(cls.OUTPUT_DIR, "logs")
    
    @classmethod
    def get_samples_dir(cls):
        return os.path.join(cls.OUTPUT_DIR, "samples")
    
    @classmethod
    def get_train_hr_dir(cls):
        """Flexible path detection for training HR images."""
        paths = [
            os.path.join(cls.DATASET_ROOT, "train/HR"),
            os.path.join(cls.DATASET_ROOT, "train_HR"),
            os.path.join(cls.DATASET_ROOT, "HR"),
            cls.DATASET_ROOT,
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return paths[0]  # Default fallback
    
    @classmethod
    def get_val_hr_dir(cls):
        """Flexible path detection for validation HR images."""
        paths = [
            os.path.join(cls.DATASET_ROOT, "val/HR"),
            os.path.join(cls.DATASET_ROOT, "val_HR"),
            os.path.join(cls.DATASET_ROOT, "valid/HR"),
            os.path.join(cls.DATASET_ROOT, "test/HR"),
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return paths[0]  # Default fallback


# ============================================================================
# FREQUENCY DECOMPOSITION
# ============================================================================

class FrequencyDecomposition(nn.Module):
    """DCT-based frequency decomposition for expert routing."""
    
    def __init__(self, block_size: int = 8, low_freq_ratio: float = 0.25, high_freq_ratio: float = 0.25):
        super().__init__()
        self.block_size = block_size
        self.low_freq_ratio = low_freq_ratio
        self.high_freq_ratio = high_freq_ratio
        
        self.register_buffer('dct_matrix', self._create_dct_matrix(block_size))
        low_mask, mid_mask, high_mask = self._create_frequency_masks(block_size)
        self.register_buffer('low_mask', low_mask)
        self.register_buffer('mid_mask', mid_mask)
        self.register_buffer('high_mask', high_mask)
    
    def _create_dct_matrix(self, n: int) -> torch.Tensor:
        dct_matrix = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_matrix[k, i] = 1.0 / math.sqrt(n)
                else:
                    dct_matrix[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        return dct_matrix
    
    def _zigzag_indices(self, n: int) -> torch.Tensor:
        indices = torch.zeros(n, n, dtype=torch.long)
        i, j = 0, 0
        for idx in range(n * n):
            indices[i, j] = idx
            if (i + j) % 2 == 0:
                if j == n - 1: i += 1
                elif i == 0: j += 1
                else: i -= 1; j += 1
            else:
                if i == n - 1: j += 1
                elif j == 0: i += 1
                else: i += 1; j -= 1
        return indices
    
    def _create_frequency_masks(self, block_size: int):
        zigzag_idx = self._zigzag_indices(block_size)
        total_coeffs = block_size * block_size
        low_threshold = int(total_coeffs * self.low_freq_ratio)
        high_threshold = int(total_coeffs * (1 - self.high_freq_ratio))
        
        low_mask = torch.zeros(block_size, block_size)
        mid_mask = torch.zeros(block_size, block_size)
        high_mask = torch.zeros(block_size, block_size)
        
        for i in range(block_size):
            for j in range(block_size):
                idx = zigzag_idx[i, j]
                if idx < low_threshold: low_mask[i, j] = 1.0
                elif idx >= high_threshold: high_mask[i, j] = 1.0
                else: mid_mask[i, j] = 1.0
        
        return low_mask, mid_mask, high_mask
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        low, mid, high = self.decompose(x)
        return {'low_freq': low, 'mid_freq': mid, 'high_freq': high, 'original': x}
    
    def decompose(self, x: torch.Tensor):
        B, C, H, W = x.shape
        bs = self.block_size
        
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h > 0 or pad_w > 0) else x
        _, _, H_pad, W_pad = x_padded.shape
        
        x_blocks = x_padded.view(B, C, H_pad // bs, bs, W_pad // bs, bs).permute(0, 1, 2, 4, 3, 5)
        dct_blocks = torch.matmul(torch.matmul(self.dct_matrix, x_blocks), self.dct_matrix.T)
        
        dct_low = dct_blocks * self.low_mask
        dct_mid = dct_blocks * self.mid_mask
        dct_high = dct_blocks * self.high_mask
        
        def idct_and_reshape(dct):
            spatial = torch.matmul(torch.matmul(self.dct_matrix.T, dct), self.dct_matrix)
            spatial = spatial.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H_pad, W_pad)
            return spatial[:, :, :H, :W] if (pad_h > 0 or pad_w > 0) else spatial
        
        return idct_and_reshape(dct_low), idct_and_reshape(dct_mid), idct_and_reshape(dct_high)


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False)
        )
    
    def forward(self, x):
        return x * torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


# ============================================================================
# FREQUENCY ROUTER
# ============================================================================

class FrequencyRouter(nn.Module):
    """Predicts routing weights for each expert per frequency band."""
    
    def __init__(self, in_channels: int = 3, num_experts: int = 3, num_bands: int = 3,
                 hidden_channels: List[int] = [32, 64, 64, 32], use_attention: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.num_bands = num_bands
        self.use_attention = use_attention
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1, bias=False),
            nn.ReLU(inplace=True), nn.BatchNorm2d(hidden_channels[0]),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, padding=1, bias=False),
            nn.ReLU(inplace=True), nn.BatchNorm2d(hidden_channels[1])
        )
        
        if use_attention:
            self.attention1 = ChannelSpatialAttention(hidden_channels[1])
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], 3, padding=1, bias=False),
            nn.ReLU(inplace=True), nn.BatchNorm2d(hidden_channels[2]),
            nn.Conv2d(hidden_channels[2], hidden_channels[3], 3, padding=1, bias=False),
            nn.ReLU(inplace=True), nn.BatchNorm2d(hidden_channels[3])
        )
        
        if use_attention:
            self.attention2 = SpatialAttention(kernel_size=5)
        
        self.output_conv = nn.Conv2d(hidden_channels[3], num_experts * num_bands, 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, lr_input: torch.Tensor) -> torch.Tensor:
        B, C, H, W = lr_input.shape
        x = self.conv_block1(lr_input)
        if self.use_attention: x = self.attention1(x)
        x = self.conv_block2(x)
        if self.use_attention: x = self.attention2(x)
        x = self.output_conv(x).view(B, self.num_experts, self.num_bands, H, W)
        return F.softmax(x, dim=1)


# ============================================================================
# MULTI-SCALE FEATURE EXTRACTOR
# ============================================================================

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.conv_1x = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                     nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels))
        self.conv_2x = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                     nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels))
        self.conv_4x = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                     nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels))
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
    
    def forward(self, x):
        H, W = x.shape[-2:]
        feat_1x = self.conv_1x(x)
        feat_2x = F.interpolate(self.conv_2x(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)),
                                size=(H, W), mode='bilinear', align_corners=False)
        feat_4x = F.interpolate(self.conv_4x(F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)),
                                size=(H, W), mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([feat_1x, feat_2x, feat_4x], dim=1))


# ============================================================================
# FREQUENCY-AWARE FUSION
# ============================================================================

class FrequencyAwareFusion(nn.Module):
    """Fuses expert outputs based on frequency content."""
    
    def __init__(self, num_experts: int = 3, num_bands: int = 3, block_size: int = 8,
                 use_residual: bool = True, use_multiscale: bool = True, upscale: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.num_bands = num_bands
        self.use_residual = use_residual
        self.use_multiscale = use_multiscale
        self.upscale = upscale
        
        self.freq_decomp = FrequencyDecomposition(block_size=block_size)
        
        if use_multiscale:
            self.multiscale = MultiScaleFeatureExtractor(in_channels=3, out_channels=32)
            self.freq_router = FrequencyRouter(in_channels=32, num_experts=num_experts, num_bands=num_bands)
        else:
            self.freq_router = FrequencyRouter(in_channels=3, num_experts=num_experts, num_bands=num_bands)
        
        self.expert_weights = nn.Parameter(torch.ones(num_experts, num_bands))
        self.band_importance = nn.Parameter(torch.ones(num_bands))
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1, bias=False)
        )
        
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, lr_input: torch.Tensor, expert_outputs: Union[List[torch.Tensor], Dict[str, torch.Tensor]]):
        B, C, H, W = lr_input.shape
        
        if isinstance(expert_outputs, dict):
            expert_outputs = list(expert_outputs.values())
        
        num_experts = len(expert_outputs)
        expert_stack = torch.stack(expert_outputs, dim=1)
        _, _, _, H_hr, W_hr = expert_stack.shape
        
        router_input = self.multiscale(lr_input) if self.use_multiscale else lr_input
        routing_weights = self.freq_router(router_input)
        
        if num_experts < self.num_experts:
            routing_weights = routing_weights[:, :num_experts]
        
        routing_flat = routing_weights.view(B, num_experts * self.num_bands, H, W)
        routing_hr = F.interpolate(routing_flat, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        routing_hr = routing_hr.view(B, num_experts, self.num_bands, H_hr, W_hr)
        
        expert_w = self.expert_weights[:num_experts].view(1, num_experts, self.num_bands, 1, 1)
        band_w = F.softmax(self.band_importance, dim=0).view(1, 1, self.num_bands, 1, 1)
        weighted_routing = routing_hr * expert_w * band_w
        
        aggregated = weighted_routing.sum(dim=2)
        aggregated = aggregated / (aggregated.sum(dim=1, keepdim=True) + 1e-8)
        aggregated = aggregated.unsqueeze(2)
        
        fused_sr = (expert_stack * aggregated).sum(dim=1)
        fused_sr = fused_sr + self.refine_conv(fused_sr) * 0.1
        
        if self.use_residual:
            bilinear_up = F.interpolate(lr_input, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
            fused_sr = fused_sr + self.residual_weight * bilinear_up
        
        return fused_sr.clamp(0, 1)


# ============================================================================
# NAFNet ARCHITECTURE (Inline - No external dependencies)
# ============================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNetSR(nn.Module):
    """NAFNet adapted for Super-Resolution with pixel shuffle upsampling."""
    
    def __init__(self, upscale: int = 4, img_channel: int = 3, width: int = 64, middle_blk_num: int = 12):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(middle_blk_num)])
        self.conv_after_body = nn.Conv2d(width, width, 3, 1, 1)
        
        if upscale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(width, width * 4, 3, 1, 1), nn.PixelShuffle(2),
                nn.Conv2d(width, width * 4, 3, 1, 1), nn.PixelShuffle(2)
            )
        elif upscale == 2:
            self.upsample = nn.Sequential(nn.Conv2d(width, width * 4, 3, 1, 1), nn.PixelShuffle(2))
        
        self.conv_last = nn.Conv2d(width, img_channel, 3, 1, 1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        feat = self.intro(x)
        feat = self.conv_after_body(self.body(feat)) + feat
        feat = self.upsample(feat)
        return self.conv_last(feat) + base


# ============================================================================
# EXPERT ENSEMBLE
# ============================================================================

class ExpertEnsemble(nn.Module):
    """Manages frozen expert models."""
    
    def __init__(self, upscale: int = 4, window_size: int = 16, device='cuda', checkpoint_dir=None):
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(Config.PRETRAINED_ROOT)
        
        self.hat = None
        self.mambair = None
        self.nafnet = None
        self._experts_loaded = {'hat': False, 'mambair': False, 'nafnet': False}
    
    def load_nafnet(self, checkpoint_path=None, freeze=True):
        """Load NAFNet-SR model."""
        try:
            self.nafnet = NAFNetSR(upscale=self.upscale, width=64, middle_blk_num=12)
            
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'nafnet' / Config.NAFNET_WEIGHTS
            
            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                state_dict = ckpt.get('params_ema') or ckpt.get('params') or ckpt.get('state_dict') or ckpt
                self.nafnet.load_state_dict(state_dict, strict=False)
                print(f"✓ NAFNet loaded from {checkpoint_path}")
            else:
                print(f"⚠ NAFNet checkpoint not found, using random weights")
            
            if freeze:
                for p in self.nafnet.parameters(): p.requires_grad = False
                self.nafnet.eval()
            
            self.nafnet = self.nafnet.to(self.device)
            self._experts_loaded['nafnet'] = True
            return True
        except Exception as e:
            print(f"✗ Failed to load NAFNet: {e}")
            return False
    
    def load_hat(self, checkpoint_path=None, freeze=True):
        """Load HAT model (requires external src/models/hat)."""
        try:
            from src.models.hat import create_hat_model
            
            self.hat = create_hat_model(
                embed_dim=180, depths=[6]*12, num_heads=[6]*12,
                window_size=self.window_size, upscale=self.upscale, img_range=1.0
            )
            
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'hat' / Config.HAT_WEIGHTS
            
            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                state_dict = ckpt.get('params_ema') or ckpt.get('params') or ckpt.get('state_dict') or ckpt
                state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
                self.hat.load_state_dict(state_dict, strict=False)
                print(f"✓ HAT loaded from {checkpoint_path}")
            else:
                print(f"⚠ HAT checkpoint not found")
            
            if freeze:
                for p in self.hat.parameters(): p.requires_grad = False
                self.hat.eval()
            
            self.hat = self.hat.to(self.device)
            self._experts_loaded['hat'] = True
            return True
        except Exception as e:
            print(f"✗ Failed to load HAT: {e}")
            return False
    
    def load_mambair(self, checkpoint_path=None, freeze=True):
        """Load MambaIR model (requires mamba-ssm package)."""
        try:
            from src.models.mambair import create_mambair_model, MAMBA_AVAILABLE
            if not MAMBA_AVAILABLE:
                print("⚠ MambaIR not available (mamba-ssm not installed)")
                return False
            
            self.mambair = create_mambair_model(upscale=self.upscale, embed_dim=180, depths=[6]*6,
                                                 num_heads=[6]*6, window_size=8, img_range=1.0)
            
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'mambair' / Config.MAMBAIR_WEIGHTS
            
            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                state_dict = ckpt.get('params_ema') or ckpt.get('params') or ckpt.get('state_dict') or ckpt
                self.mambair.load_state_dict(state_dict, strict=False)
                print(f"✓ MambaIR loaded from {checkpoint_path}")
            
            if freeze:
                for p in self.mambair.parameters(): p.requires_grad = False
                self.mambair.eval()
            
            self.mambair = self.mambair.to(self.device)
            self._experts_loaded['mambair'] = True
            return True
        except Exception as e:
            print(f"✗ Failed to load MambaIR: {e}")
            return False
    
    def load_all_experts(self, freeze=True):
        print("\n" + "="*60 + "\nLoading Expert Models\n" + "="*60)
        results = {
            'hat': self.load_hat(freeze=freeze),
            'mambair': self.load_mambair(freeze=freeze),
            'nafnet': self.load_nafnet(freeze=freeze)
        }
        print(f"Loaded {sum(results.values())}/3 experts\n" + "="*60)
        return results
    
    def _pad_for_window(self, x):
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w
    
    @torch.no_grad()
    def forward_all(self, x, return_dict=False):
        outputs = {}
        _, _, h, w = x.shape
        target_h, target_w = h * self.upscale, w * self.upscale
        
        if self._experts_loaded['hat'] and self.hat is not None:
            x_pad, orig_h, orig_w = self._pad_for_window(x)
            sr = self.hat(x_pad)
            outputs['hat'] = sr[:, :, :target_h, :target_w].clamp(0, 1)
        
        if self._experts_loaded['mambair'] and self.mambair is not None:
            outputs['mambair'] = self.mambair(x).clamp(0, 1)
        
        if self._experts_loaded['nafnet'] and self.nafnet is not None:
            outputs['nafnet'] = self.nafnet(x).clamp(0, 1)
        
        return outputs if return_dict else list(outputs.values())
    
    def get_loaded_experts(self):
        return [k for k, v in self._experts_loaded.items() if v]


# ============================================================================
# MULTI-FUSION SR MODEL
# ============================================================================

class MultiFusionSR(nn.Module):
    """Complete frequency-aware fusion pipeline."""
    
    def __init__(self, expert_ensemble, num_experts=3, block_size=8, upscale=4):
        super().__init__()
        self.expert_ensemble = expert_ensemble
        for p in self.expert_ensemble.parameters(): p.requires_grad = False
        
        self.fusion = FrequencyAwareFusion(
            num_experts=num_experts, num_bands=3, block_size=block_size,
            use_residual=True, use_multiscale=True, upscale=upscale
        )
        self.upscale = upscale
    
    def forward(self, lr_input, return_intermediates=False):
        with torch.no_grad():
            expert_outputs = self.expert_ensemble.forward_all(lr_input, return_dict=True)
        
        if len(expert_outputs) == 0:
            return F.interpolate(lr_input, scale_factor=self.upscale, mode='bicubic', align_corners=False).clamp(0, 1)
        
        fused_sr = self.fusion(lr_input, expert_outputs)
        return fused_sr
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / (2 * sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window_2d.expand(channel, 1, window_size, window_size).contiguous())
    
    def forward(self, pred, target):
        C1, C2 = 0.01**2, 0.03**2
        window = self.window.to(pred.device)
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=target.size(1))
        mu1_sq, mu2_sq = mu1**2, mu2**2
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=pred.size(1)) - mu1 * mu2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim.mean()


class FFTLoss(nn.Module):
    def __init__(self, high_freq_weight=2.0):
        super().__init__()
        self.high_freq_weight = high_freq_weight
    
    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
        
        mag_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        phase_loss = F.l1_loss(torch.angle(pred_fft), torch.angle(target_fft))
        return mag_loss + 0.1 * phase_loss


class SWTLoss(nn.Module):
    """Stationary Wavelet Transform Loss - Samsung's NTIRE 2025 technique.
    
    Uses numpy-based SWT on CPU for correctness (pywt doesn't support GPU).
    Emphasizes high-frequency details for sharper SR outputs.
    """
    
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.band_weights = {'a': 0.5, 'h': 1.5, 'v': 1.5, 'd': 2.0}
    
    def forward(self, pred, target):
        """Compute SWT loss using numpy (CPU-based, accurate)."""
        if not PYWT_AVAILABLE:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)
        
        B, C, H, W = pred.shape
        device = pred.device
        dtype = pred.dtype
        
        # Ensure dimensions are compatible with SWT
        min_size = 2 ** self.level
        if H < min_size or W < min_size:
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        total_loss = 0.0
        count = 0
        
        # Process on CPU with numpy (pywt requirement)
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        for b in range(B):
            for c in range(C):
                try:
                    # Pad to make dimensions even if needed
                    p_img = pred_np[b, c]
                    t_img = target_np[b, c]
                    
                    h, w = p_img.shape
                    pad_h = (min_size - h % min_size) % min_size
                    pad_w = (min_size - w % min_size) % min_size
                    
                    if pad_h > 0 or pad_w > 0:
                        p_img = np.pad(p_img, ((0, pad_h), (0, pad_w)), mode='reflect')
                        t_img = np.pad(t_img, ((0, pad_h), (0, pad_w)), mode='reflect')
                    
                    # Compute SWT coefficients
                    pred_coeffs = pywt.swt2(p_img, self.wavelet, level=self.level)
                    target_coeffs = pywt.swt2(t_img, self.wavelet, level=self.level)
                    
                    # Compare coefficients at each level
                    for (pa, (ph, pv, pd)), (ta, (th, tv, td)) in zip(pred_coeffs, target_coeffs):
                        total_loss += self.band_weights['a'] * np.abs(pa - ta).mean()
                        total_loss += self.band_weights['h'] * np.abs(ph - th).mean()
                        total_loss += self.band_weights['v'] * np.abs(pv - tv).mean()
                        total_loss += self.band_weights['d'] * np.abs(pd - td).mean()
                    
                    count += self.level
                    
                except Exception:
                    # Skip if SWT fails (dimension issues, etc.)
                    continue
        
        # Normalize and return as tensor
        if count > 0:
            total_loss = total_loss / (count * 4)  # 4 bands per level
        
        return torch.tensor(float(total_loss), device=device, dtype=dtype, requires_grad=False)


class CombinedLoss(nn.Module):
    def __init__(self, use_swt=True, use_fft=True):
        super().__init__()
        self.l1 = L1Loss()
        self.ssim = SSIMLoss()
        self.fft = FFTLoss() if use_fft else None
        self.swt = SWTLoss() if (use_swt and PYWT_AVAILABLE) else None
        self.weights = {'l1': 1.0, 'ssim': 0.0, 'fft': 0.0, 'swt': 0.0}
    
    def set_weights(self, weights):
        self.weights.update(weights)
    
    def forward(self, pred, target):
        loss = self.weights['l1'] * self.l1(pred, target)
        if self.weights.get('ssim', 0) > 0:
            loss += self.weights['ssim'] * self.ssim(pred, target)
        if self.fft and self.weights.get('fft', 0) > 0:
            loss += self.weights['fft'] * self.fft(pred, target)
        if self.swt and self.weights.get('swt', 0) > 0:
            loss += self.weights['swt'] * self.swt(pred, target)
        return loss


# ============================================================================
# DATASET
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, patch_size=64, scale=4, augment=True, repeat=1):
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.repeat = repeat
        
        self.hr_files = sorted(list(self.hr_dir.glob("*.png")) + list(self.hr_dir.glob("*.jpg")))
        if not self.hr_files:
            raise ValueError(f"No images found in {hr_dir}")
        print(f"Found {len(self.hr_files)} images (repeat={repeat})")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        idx = idx % len(self.hr_files)
        hr_path = self.hr_files[idx]
        
        hr = cv2.imread(str(hr_path))
        if hr is None:
            raise ValueError(f"Failed to load {hr_path}")
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        
        h, w = hr.shape[:2]
        hr_ps = self.patch_size * self.scale
        
        if h < hr_ps or w < hr_ps:
            hr = cv2.resize(hr, (max(w, hr_ps), max(h, hr_ps)), interpolation=cv2.INTER_CUBIC)
            h, w = hr.shape[:2]
        
        top = random.randint(0, h - hr_ps)
        left = random.randint(0, w - hr_ps)
        hr_patch = hr[top:top+hr_ps, left:left+hr_ps]
        
        lr_patch = cv2.resize(hr_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        
        if self.augment:
            if random.random() < 0.5: hr_patch, lr_patch = hr_patch[:, ::-1], lr_patch[:, ::-1]
            if random.random() < 0.5: hr_patch, lr_patch = hr_patch[::-1], lr_patch[::-1]
            k = random.randint(0, 3)
            hr_patch, lr_patch = np.rot90(hr_patch, k), np.rot90(lr_patch, k)
        
        hr_tensor = torch.from_numpy(hr_patch.copy()).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch.copy()).permute(2, 0, 1).float() / 255.0
        
        return {'lr': lr_tensor, 'hr': hr_tensor, 'filename': hr_path.stem}


class SRValDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, scale=4):
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.hr_files = sorted(list(self.hr_dir.glob("*.png")) + list(self.hr_dir.glob("*.jpg")))[:20]
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        hr = cv2.imread(str(hr_path))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        
        h, w = hr.shape[:2]
        h = h - h % self.scale
        w = w - w % self.scale
        hr = hr[:h, :w]
        
        lr = cv2.resize(hr, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)
        
        hr_tensor = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        
        return {'lr': lr_tensor, 'hr': hr_tensor, 'filename': hr_path.stem}


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict.copy()


def rgb_to_y(img):
    return 16./255. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 255.


def calculate_psnr(sr, hr, crop_border=4, test_y=True):
    if crop_border > 0:
        sr = sr[:, crop_border:-crop_border, crop_border:-crop_border]
        hr = hr[:, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y:
        sr_y = rgb_to_y(sr)
        hr_y = rgb_to_y(hr)
    else:
        sr_y, hr_y = sr, hr
    
    mse = ((sr_y - hr_y) ** 2).mean()
    return 10 * math.log10(1.0 / (mse.item() + 1e-10))


def get_loss_stage(epoch, loss_stages):
    current_stage = loss_stages[0]
    for start_epoch, config in loss_stages:
        if epoch >= start_epoch:
            current_stage = (start_epoch, config)
    return current_stage[1], current_stage[0]


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, config, ema=None):
    """Train one epoch with gradient accumulation for memory efficiency."""
    model.train()
    total_loss = 0
    accumulation_steps = getattr(config, 'ACCUMULATION_STEPS', 2)
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        lr, hr = batch['lr'].to(device), batch['hr'].to(device)
        
        sr = model(lr)
        loss = criterion(sr, hr) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema: ema.update(model)
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    # Handle remaining gradients if batches don't divide evenly
    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if ema: ema.update(model)
    
    return {'loss': total_loss / len(loader)}


@torch.no_grad()
def validate_epoch(model, loader, device, config, ema=None):
    model.eval()
    if ema: ema.apply_shadow(model)
    
    psnrs = []
    for batch in tqdm(loader, desc="Validating"):
        lr, hr = batch['lr'].to(device), batch['hr'].to(device)
        sr = model(lr).clamp(0, 1)
        
        for i in range(sr.size(0)):
            psnr = calculate_psnr(sr[i], hr[i], config.CROP_BORDER, config.TEST_Y_CHANNEL)
            psnrs.append(psnr)
    
    if ema: ema.restore(model)
    return {'psnr': np.mean(psnrs), 'ssim': 0.0}


@torch.no_grad()
def save_sample_image(model, val_loader, device, epoch, config):
    model.eval()
    samples_dir = Path(config.get_samples_dir())
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    batch = next(iter(val_loader))
    lr, hr = batch['lr'].to(device), batch['hr'].to(device)
    sr = model(lr).clamp(0, 1)
    
    lr_up = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
    
    def to_np(t):
        return (t[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    lr_np, sr_np, hr_np = to_np(lr_up), to_np(sr), to_np(hr)
    h, w = sr_np.shape[:2]
    comparison = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    comparison[:, :w] = lr_np
    comparison[:, w+10:w*2+10] = sr_np
    comparison[:, w*2+20:] = hr_np
    
    comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.putText(comparison, 'LR', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'SR', (w+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'HR', (w*2+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    save_path = samples_dir / f'epoch_{epoch:04d}.png'
    cv2.imwrite(str(save_path), comparison)
    print(f"  📸 Saved: {save_path.name}")


def train(config=Config):
    print("\n" + "="*70 + "\nNTIRE 2025 SR TRAINING - COMPLETE VERSION\n" + "="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(config.get_checkpoint_dir(), exist_ok=True)
    os.makedirs(config.get_samples_dir(), exist_ok=True)
    
    # Load experts
    ensemble = ExpertEnsemble(device=device, checkpoint_dir=config.PRETRAINED_ROOT)
    ensemble.load_all_experts(freeze=True)
    ensemble = ensemble.to(device)
    
    num_experts = len(ensemble.get_loaded_experts())
    if num_experts == 0:
        print("ERROR: No experts loaded! Check checkpoint paths.")
        return
    
    # Create model
    model = MultiFusionSR(ensemble, num_experts=num_experts, upscale=config.SCALE).to(device)
    print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Datasets
    # Use flexible path detection
    train_hr_dir = config.get_train_hr_dir()
    val_hr_dir = config.get_val_hr_dir()
    print(f"Train HR dir: {train_hr_dir}")
    print(f"Val HR dir: {val_hr_dir}")
    
    train_set = SRDataset(
        hr_dir=train_hr_dir,
        patch_size=config.LR_PATCH_SIZE, scale=config.SCALE, repeat=config.REPEAT_FACTOR
    )
    val_set = SRValDataset(
        hr_dir=val_hr_dir, scale=config.SCALE
    )
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    
    # Training setup
    criterion = CombinedLoss(use_swt=PYWT_AVAILABLE, use_fft=True)
    optimizer = optim.AdamW(model.fusion.parameters(), lr=config.INITIAL_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TOTAL_EPOCHS - config.WARMUP_EPOCHS,
                                                      eta_min=config.MIN_LR)
    ema = EMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None
    
    best_psnr = 0.0
    
    for epoch in range(config.TOTAL_EPOCHS):
        # Update loss weights based on stage
        loss_weights, stage_start = get_loss_stage(epoch, config.LOSS_STAGES)
        criterion.set_weights(loss_weights)
        
        stage_name = f"Stage {list(loss_weights.keys())}"
        print(f"\nEpoch {epoch}/{config.TOTAL_EPOCHS-1} [{stage_name}] | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, ema)
        
        if epoch >= config.WARMUP_EPOCHS:
            scheduler.step()
        
        val_metrics = None
        if epoch % config.VALIDATE_EVERY == 0 or epoch == config.TOTAL_EPOCHS - 1:
            val_metrics = validate_epoch(model, val_loader, device, config, ema)
        
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        if val_metrics:
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB | Best: {best_psnr:.2f} dB")
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_psnr': best_psnr,
                    'ema_state_dict': ema.state_dict() if ema else None
                }, os.path.join(config.get_checkpoint_dir(), 'best_model.pth'))
        
        save_sample_image(model, val_loader, device, epoch, config)
        
        if epoch % config.SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'psnr': val_metrics['psnr'] if val_metrics else 0
            }, os.path.join(config.get_checkpoint_dir(), f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"\n" + "="*70 + f"\nTRAINING COMPLETE! Best PSNR: {best_psnr:.2f} dB\n" + "="*70)
    return best_psnr


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n=== NTIRE 2025 SR Training Configuration ===")
    print(f"Dataset: {Config.DATASET_ROOT}")
    print(f"Pretrained: {Config.PRETRAINED_ROOT}")
    print(f"Output: {Config.OUTPUT_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Total Epochs: {Config.TOTAL_EPOCHS}")
    
    print("\n=== Required Packages ===")
    print("pip install torch torchvision einops timm pywt opencv-python tqdm")
    print("pip install mamba-ssm causal-conv1d  # Optional, for MambaIR")
    
    train(Config)
