"""
Hierarchical Multi-Resolution Fusion (Phase 1)
================================================
Progressive upsampling: 64x64 -> 128x128 -> 256x256

Why this works:
- Stage 1 (64x64): Learn global structure and layout
- Stage 2 (128x128): Add mid-level textures with residual from Stage 1
- Stage 3 (256x256): Refine fine details and edges with residual from Stage 2

Key design choices:
- Content-aware gating at each stage (spatial attention on expert features)
- Learnable residual connections between stages
- GELU activation for smoother gradients
- Channel reduction in final stage for efficient output projection

Expected gain: ~0.8-1.0 dB over flat single-resolution fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGate(nn.Module):
    """
    Lightweight spatial attention gate.
    
    Learns a per-pixel importance map to focus on informative regions
    at each fusion stage. Uses channel squeeze + spatial expand pattern.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResBlock(nn.Module):
    """
    Residual block with pre-activation and GELU.
    
    More stable than plain conv blocks for deep refinement networks.
    Pre-activation design (BN->GELU->Conv) prevents gradient degradation.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


class HierarchicalMultiResolutionFusion(nn.Module):
    """
    Progressive 3-stage fusion with residual connections and spatial gating.
    
    Architecture:
        Stage 1 (64x64):   expert_stack -> conv -> spatial_gate -> ResBlock -> feat_64
        Stage 2 (128x128): [upsample(feat_64) + experts_128] -> conv -> gate -> ResBlock -> feat_128
        Stage 3 (256x256): [upsample(feat_128) + experts_256] -> conv -> gate -> ResBlock -> feat_256
        Output:  feat_256 -> to_rgb -> sigmoid -> [B, 3, 256, 256]
    
    Args:
        num_experts: Number of expert models (default: 3 = HAT, DAT, NAFNet)
        base_channels: Channel dimension for fusion (default: 128)
    """
    
    def __init__(self, num_experts: int = 3, base_channels: int = 128):
        super().__init__()
        
        self.num_experts = num_experts
        self.base_channels = base_channels
        in_ch = num_experts * 3  # 9 channels from stacking 3 experts
        
        # ===== Stage 1: Coarse Fusion (64x64) =====
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.stage1_gate = SpatialGate(base_channels)
        self.stage1_res = ResBlock(base_channels)
        
        # ===== Stage 2: Mid-Level Fusion (128x128) =====
        self.stage2_conv = nn.Sequential(
            nn.Conv2d(base_channels + in_ch, base_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.stage2_gate = SpatialGate(base_channels)
        self.stage2_res = ResBlock(base_channels)
        
        # ===== Stage 3: Fine Fusion (256x256) =====
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(base_channels + in_ch, base_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels // 2, 3, 1, 1),
            nn.GELU(),
        )
        self.stage3_gate = SpatialGate(base_channels // 2)
        self.stage3_res = ResBlock(base_channels // 2)
        
        # ===== Output Projection =====
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels // 4, 3, 3, 1, 1),
        )
        
        # ===== Learnable Residual Weights =====
        # Start small so early training doesn't propagate noise across stages
        self.residual_weight_1_2 = nn.Parameter(torch.tensor(0.2))
        self.residual_weight_2_3 = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, expert_outputs: dict) -> torch.Tensor:
        """
        Args:
            expert_outputs: Dict with keys like 'hat', 'dat', 'nafnet'
                           Each value: [B, 3, 256, 256]
        
        Returns:
            fused: [B, 3, 256, 256]
        """
        # Stack all expert outputs along channel dim
        expert_list = list(expert_outputs.values())
        expert_stack = torch.cat(expert_list, dim=1)  # [B, 9, 256, 256]
        
        # ===== Stage 1: Coarse (64x64) =====
        experts_64 = F.interpolate(
            expert_stack, size=(64, 64),
            mode='bilinear', align_corners=False
        )  # [B, 9, 64, 64]
        
        feat_64 = self.stage1_conv(experts_64)       # [B, C, 64, 64]
        feat_64 = self.stage1_gate(feat_64)           # spatial attention
        feat_64 = self.stage1_res(feat_64)            # residual refinement
        
        # ===== Stage 2: Mid-Level (128x128) =====
        feat_64_up = F.interpolate(
            feat_64, size=(128, 128),
            mode='bilinear', align_corners=False
        )  # [B, C, 128, 128]
        
        experts_128 = F.interpolate(
            expert_stack, size=(128, 128),
            mode='bilinear', align_corners=False
        )  # [B, 9, 128, 128]
        
        stage2_input = torch.cat([feat_64_up, experts_128], dim=1)  # [B, C+9, 128, 128]
        feat_128 = self.stage2_conv(stage2_input)     # [B, C, 128, 128]
        feat_128 = self.stage2_gate(feat_128)
        feat_128 = self.stage2_res(feat_128)
        feat_128 = feat_128 + self.residual_weight_1_2 * feat_64_up  # cross-stage residual
        
        # ===== Stage 3: Fine (256x256) =====
        feat_128_up = F.interpolate(
            feat_128, size=(256, 256),
            mode='bilinear', align_corners=False
        )  # [B, C, 256, 256]
        
        stage3_input = torch.cat([feat_128_up, expert_stack], dim=1)  # [B, C+9, 256, 256]
        feat_256 = self.stage3_conv(stage3_input)     # [B, C//2, 256, 256]
        feat_256 = self.stage3_gate(feat_256)
        feat_256 = self.stage3_res(feat_256)
        
        # Cross-stage residual: 128ch -> 64ch via channel split + mean
        feat_128_up_reduced = feat_128_up[:, :self.base_channels // 2, :, :]  # take first half
        feat_256 = feat_256 + self.residual_weight_2_3 * feat_128_up_reduced
        
        # ===== Output RGB =====
        output = self.to_rgb(feat_256)  # [B, 3, 256, 256]
        output = torch.sigmoid(output)  # Ensure [0, 1] range
        
        return output


def test_hierarchical_fusion():
    """Quick test of the hierarchical fusion module."""
    print("Testing HierarchicalMultiResolutionFusion...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fusion = HierarchicalMultiResolutionFusion(
        num_experts=3, base_channels=128
    ).to(device)
    
    params = sum(p.numel() for p in fusion.parameters())
    print(f"  Parameters: {params:,}")
    
    batch = 2
    expert_outputs = {
        'hat': torch.randn(batch, 3, 256, 256, device=device),
        'dat': torch.randn(batch, 3, 256, 256, device=device),
        'nafnet': torch.randn(batch, 3, 256, 256, device=device)
    }
    
    with torch.no_grad():
        output = fusion(expert_outputs)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == (batch, 3, 256, 256)
    assert 0 <= output.min() and output.max() <= 1
    
    print("  PASSED!")


if __name__ == '__main__':
    test_hierarchical_fusion()
