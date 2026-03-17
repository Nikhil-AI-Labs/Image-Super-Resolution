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
        num_experts: Number of expert models (default: 4 = DRCT, GRL, NAFNet, MambaIR)
        base_channels: Channel dimension for fusion (default: 128)
    """
    
    def __init__(self, num_experts: int = 4, base_channels: int = 128):
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
        Progressive 3-stage hierarchical fusion.
        Dynamic resolution: derives levels from input HR size (no hardcoded 64/128/256).

        Args:
            expert_outputs: Dict with keys like 'drct', 'grl', 'nafnet', 'mamba'
                           Each value: [B, 3, H_hr, W_hr]

        Returns:
            fused: [B, 3, H_hr, W_hr]
        """
        # Stack all expert outputs along channel dim
        expert_list = list(expert_outputs.values())
        expert_stack = torch.cat(expert_list, dim=1)  # [B, N*3, H_hr, W_hr]

        # Read the exact spatial size of the incoming HR expert outputs
        _, _, H_hr, W_hr = expert_stack.shape

        # Dynamic scale levels derived from HR size (preserves aspect ratio!)
        h1, w1 = max(H_hr // 4, 1), max(W_hr // 4, 1)  # Level 1 (Coarse)
        h2, w2 = max(H_hr // 2, 1), max(W_hr // 2, 1)  # Level 2 (Mid)
        # Level 3 = full HR size (H_hr, W_hr)

        # ===== Stage 1: Coarse =====
        experts_s1 = F.interpolate(
            expert_stack, size=(h1, w1),
            mode='bilinear', align_corners=False
        )

        feat_s1 = self.stage1_conv(experts_s1)
        feat_s1 = self.stage1_gate(feat_s1)
        feat_s1 = self.stage1_res(feat_s1)

        # ===== Stage 2: Mid-Level =====
        feat_s1_up = F.interpolate(
            feat_s1, size=(h2, w2),
            mode='bilinear', align_corners=False
        )

        experts_s2 = F.interpolate(
            expert_stack, size=(h2, w2),
            mode='bilinear', align_corners=False
        )

        stage2_input = torch.cat([feat_s1_up, experts_s2], dim=1)
        feat_s2 = self.stage2_conv(stage2_input)
        feat_s2 = self.stage2_gate(feat_s2)
        feat_s2 = self.stage2_res(feat_s2)
        feat_s2 = feat_s2 + self.residual_weight_1_2 * feat_s1_up

        # ===== Stage 3: Fine (full HR resolution) =====
        feat_s2_up = F.interpolate(
            feat_s2, size=(H_hr, W_hr),
            mode='bilinear', align_corners=False
        )

        stage3_input = torch.cat([feat_s2_up, expert_stack], dim=1)
        feat_s3 = self.stage3_conv(stage3_input)
        feat_s3 = self.stage3_gate(feat_s3)
        feat_s3 = self.stage3_res(feat_s3)

        # Cross-stage residual: take first half of channels
        feat_s2_up_reduced = feat_s2_up[:, :self.base_channels // 2, :, :]
        feat_s3 = feat_s3 + self.residual_weight_2_3 * feat_s2_up_reduced

        # ===== Output RGB =====
        output = self.to_rgb(feat_s3)
        output = torch.sigmoid(output)

        return output


def test_hierarchical_fusion():
    """Quick test of the hierarchical fusion module."""
    print("Testing HierarchicalMultiResolutionFusion...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fusion = HierarchicalMultiResolutionFusion(
        num_experts=4, base_channels=128
    ).to(device)
    
    params = sum(p.numel() for p in fusion.parameters())
    print(f"  Parameters: {params:,}")
    
    batch = 2
    expert_outputs = {
        'drct': torch.randn(batch, 3, 256, 256, device=device),
        'grl': torch.randn(batch, 3, 256, 256, device=device),
        'nafnet': torch.randn(batch, 3, 256, 256, device=device),
        'mamba': torch.randn(batch, 3, 256, 256, device=device),
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
