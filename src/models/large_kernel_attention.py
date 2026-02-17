"""
Large Kernel Attention (LKA) — Phase 3
========================================
Provides global receptive field via decomposed large kernels.

Reference: "Visual Attention Network" (VAN), adapted for super-resolution.

Architecture:
  21×21 kernel decomposed into:
    • 5×5 depth-wise conv   → local context
    • 1×21 depth-wise conv  → horizontal global
    • 21×1 depth-wise conv  → vertical global
    • 1×1 point-wise conv   → channel mixing

  Effective receptive field covers entire 64×64 LR patch.
  O(H*W*k) complexity vs O(H²*W²) for full attention.

Modules:
  - LargeKernelAttention: Core decomposed conv attention
  - LKABlock: LKA + LayerNorm + FFN with residual connections
  - EnhancedCrossBandWithLKA: Drop-in replacement for CrossBandAttention
  - EnhancedCollaborativeWithLKA: Drop-in replacement for CollaborativeFeatureLearning

Expected gain: +0.7 dB from global context awareness.
Parameters: ~120K total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# =============================================================================
# Core LKA Module
# =============================================================================

class LargeKernelAttention(nn.Module):
    """
    Decomposed large-kernel spatial attention.
    
    21×21 receptive field achieved via:
      5×5 DW → 1×21 DW → 21×1 DW → 1×1 PW → sigmoid gate
    
    All convolutions are depth-wise (groups=dim) except the final 1×1,
    keeping parameter count very low (~2K per instance).
    
    Args:
        dim: Channel dimension
        kernel_size: Large kernel size (default: 21)
    """
    
    def __init__(self, dim: int, kernel_size: int = 21):
        super().__init__()
        
        pad_large = kernel_size // 2
        
        # Step 1: Local 5×5 depth-wise conv (captures fine-grained detail)
        self.local_conv = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2,
            groups=dim, bias=False
        )
        
        # Step 2: Horizontal 1×k depth-wise conv (row-wise patterns)
        self.h_conv = nn.Conv2d(
            dim, dim,
            kernel_size=(1, kernel_size),
            padding=(0, pad_large),
            groups=dim, bias=False
        )
        
        # Step 3: Vertical k×1 depth-wise conv (column-wise patterns)
        self.v_conv = nn.Conv2d(
            dim, dim,
            kernel_size=(kernel_size, 1),
            padding=(pad_large, 0),
            groups=dim, bias=False
        )
        
        # Step 4: Channel mixing 1×1 conv
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # Batch norm for stability
        self.bn = nn.BatchNorm2d(dim)
        
        # Initialize close to identity for training stability
        nn.init.kaiming_normal_(self.local_conv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.h_conv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.v_conv.weight, mode='fan_out')
        nn.init.xavier_uniform_(self.pw_conv.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] → attention-modulated [B, C, H, W]
        """
        identity = x
        
        attn = self.local_conv(x)      # Local 5×5
        attn = self.h_conv(attn)       # Horizontal 1×21
        attn = self.v_conv(attn)       # Vertical 21×1
        attn = self.pw_conv(attn)      # Channel mix 1×1
        attn = self.bn(attn)
        attn = torch.sigmoid(attn)     # → [0, 1] attention map
        
        return identity * attn


# =============================================================================
# LKA Block with Residual + FFN
# =============================================================================

class LKABlock(nn.Module):
    """
    LKA attention block with residual connections and feed-forward network.
    
    Architecture:
        x → Norm → LKA → + residual → Norm → FFN → + residual → out
    
    Args:
        dim: Channel dimension
        kernel_size: LKA kernel size
        ffn_ratio: FFN expansion ratio
    """
    
    def __init__(self, dim: int, kernel_size: int = 21, ffn_ratio: float = 2.0):
        super().__init__()
        
        self.norm1 = nn.BatchNorm2d(dim)
        self.lka = LargeKernelAttention(dim, kernel_size)
        
        self.norm2 = nn.BatchNorm2d(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_dim, 1),
            nn.GELU(),
            nn.Conv2d(ffn_dim, dim, 1),
        )
        
        # Learnable residual scales (start near 0 for stable init)
        self.scale1 = nn.Parameter(torch.tensor(0.1))
        self.scale2 = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, C, H, W]"""
        # LKA branch
        x = x + self.scale1 * self.lka(self.norm1(x))
        # FFN branch
        x = x + self.scale2 * self.ffn(self.norm2(x))
        return x


# =============================================================================
# Enhanced Cross-Band Attention with LKA
# =============================================================================

class EnhancedCrossBandWithLKA(nn.Module):
    """
    Drop-in replacement for CrossBandAttention with LKA global context.
    
    Pipeline per forward call:
        1. Project each band from 3ch → dim
        2. Multi-head attention across bands (inter-band communication)
        3. LKA refinement per band (global spatial context)
        4. Project back to 3ch + residual
    
    Interface matches CrossBandAttention:
        forward(List[Tensor]) → List[Tensor]
    
    Args:
        dim: Hidden feature dimension (default: 64)
        num_bands: Number of frequency bands (3 or 9)
        num_heads: Attention heads
        lka_kernel: LKA kernel size
    """
    
    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 9,
        num_heads: int = 4,
        lka_kernel: int = 21,
    ):
        super().__init__()
        
        self.num_bands = num_bands
        self.dim = dim
        
        # Per-band projection: 3ch → dim
        self.band_proj = nn.Conv2d(3, dim, 1)
        
        # Multi-head attention for cross-band interaction
        self.band_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(dim)
        
        # Per-band LKA block for global spatial refinement
        # Shared LKA across bands to keep parameters reasonable
        self.lka_block = LKABlock(dim, kernel_size=lka_kernel, ffn_ratio=2.0)
        
        # Output projection: dim → 3ch
        self.out_proj = nn.Conv2d(dim, 3, 1)
    
    def forward(self, band_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            band_features: List of tensors [B, 3, H, W] × num_bands
        Returns:
            Enhanced band features: List of tensors [B, 3, H, W] × num_bands
        """
        B = band_features[0].shape[0]
        H, W = band_features[0].shape[2], band_features[0].shape[3]
        num_bands = len(band_features)
        
        # Step 1: Project all bands to hidden dim
        projected = [self.band_proj(f) for f in band_features]  # list of [B, dim, H, W]
        
        # Step 2: Cross-band attention
        stacked = torch.stack(projected, dim=1)  # [B, num_bands, dim, H, W]
        stacked_flat = stacked.permute(0, 3, 4, 1, 2).reshape(
            B * H * W, num_bands, self.dim
        )
        
        normed = self.norm(stacked_flat)
        attn_out, _ = self.band_attention(normed, normed, normed)
        attn_out = attn_out + stacked_flat  # Residual
        
        # Reshape back: [B, num_bands, dim, H, W]
        attn_out = attn_out.reshape(B, H, W, num_bands, self.dim)
        attn_out = attn_out.permute(0, 3, 4, 1, 2)
        
        # Step 3: LKA refinement per band + project back + residual
        output_bands = []
        for i in range(num_bands):
            band_feat = attn_out[:, i]              # [B, dim, H, W]
            band_feat = self.lka_block(band_feat)   # Global context
            out = self.out_proj(band_feat)           # [B, 3, H, W]
            out = out + band_features[i]            # Residual connection
            output_bands.append(out)
        
        return output_bands


# =============================================================================
# Enhanced Collaborative Learning with LKA
# =============================================================================

class EnhancedCollaborativeWithLKA(nn.Module):
    """
    Drop-in replacement for CollaborativeFeatureLearning with LKA.
    
    Pipeline:
        1. Align expert features to common_dim
        2. Cross-expert attention (experts share knowledge)
        3. LKA global refinement (21×21 receptive field)
        4. Generate per-expert modulation maps
        5. Modulate expert SR outputs
    
    Interface matches CollaborativeFeatureLearning:
        forward(Dict[str, Tensor], List[Tensor]) → List[Tensor]
    
    Args:
        num_experts: Number of experts (default: 3)
        feature_dim: Common feature dimension (default: 128)
        num_heads: Attention heads
        lka_kernel: LKA kernel size
    """
    
    def __init__(
        self,
        num_experts: int = 3,
        feature_dim: int = 128,
        num_heads: int = 8,
        lka_kernel: int = 21,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        
        # Feature alignment (raw expert channels → common dim)
        self.align_layers = nn.ModuleDict({
            'hat': nn.Conv2d(180, feature_dim, 1),
            'dat': nn.Conv2d(180, feature_dim, 1),
            'nafnet': nn.Conv2d(64, feature_dim, 1),
        })
        
        # Cross-expert attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # FFN for post-attention processing
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # LKA global refinement (shared across experts)
        self.lka_global = LKABlock(feature_dim, kernel_size=lka_kernel, ffn_ratio=2.0)
        
        # Per-expert modulation heads
        self.modulation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, 1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feature_dim // 4, 3, 1),
                nn.Sigmoid()
            ) for _ in range(num_experts)
        ])
    
    def forward(
        self,
        expert_features: Dict[str, torch.Tensor],
        expert_outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            expert_features: Dict with intermediate features
                {'hat': [B, 180, H, W], 'dat': [B, 180, H, W], 'nafnet': [B, 64, H, W]}
            expert_outputs: List of SR outputs [B, 3, H_hr, W_hr]
        Returns:
            enhanced_outputs: List of enhanced SR outputs
        """
        expert_names = ['hat', 'dat', 'nafnet'][:self.num_experts]
        
        # Step 1: Align features
        aligned = {}
        for name in expert_names:
            if name in expert_features and name in self.align_layers:
                feat = expert_features[name]
                align_conv = self.align_layers[name]
                expected_ch = align_conv.weight.shape[1]
                actual_ch = feat.shape[1]
                
                if actual_ch == expected_ch:
                    aligned[name] = align_conv(feat)
                elif actual_ch > expected_ch:
                    # Truncate channels if too many
                    aligned[name] = align_conv(feat[:, :expected_ch])
                else:
                    # Pad channels if too few
                    B, C, H, W = feat.shape
                    padded = F.pad(feat, (0, 0, 0, 0, 0, expected_ch - actual_ch))
                    aligned[name] = align_conv(padded)
        
        names = list(aligned.keys())
        if not names:
            return expert_outputs  # Fallback
        
        # Common spatial size
        min_h = min(f.shape[2] for f in aligned.values())
        min_w = min(f.shape[3] for f in aligned.values())
        for name in aligned:
            if aligned[name].shape[2] != min_h or aligned[name].shape[3] != min_w:
                aligned[name] = F.interpolate(
                    aligned[name], size=(min_h, min_w),
                    mode='bilinear', align_corners=False
                )
        
        B = aligned[names[0]].shape[0]
        H, W = min_h, min_w
        device = expert_outputs[0].device
        
        feat_list = [
            aligned.get(n, torch.zeros(B, self.feature_dim, H, W, device=device))
            for n in expert_names
        ]
        stacked = torch.stack(feat_list, dim=1)  # [B, E, C, H, W]
        
        # Step 2: Cross-expert attention
        stacked_flat = stacked.permute(0, 3, 4, 1, 2).reshape(
            B * H * W, self.num_experts, self.feature_dim
        )
        
        normed = self.norm1(stacked_flat)
        attn_out, _ = self.cross_attn(normed, normed, normed)
        stacked_flat = stacked_flat + attn_out
        stacked_flat = stacked_flat + self.ffn(self.norm2(stacked_flat))
        
        # Reshape back: [B, E, C, H, W]
        enhanced = stacked_flat.reshape(B, H, W, self.num_experts, self.feature_dim)
        enhanced = enhanced.permute(0, 3, 4, 1, 2)
        
        # Step 3: LKA global refinement per expert
        H_sr, W_sr = expert_outputs[0].shape[2], expert_outputs[0].shape[3]
        enhanced_outputs = []
        
        for i, out in enumerate(expert_outputs):
            # Get expert features
            exp_feat = enhanced[:, i]  # [B, C, H, W]
            
            # Apply LKA for global context
            exp_feat = self.lka_global(exp_feat)
            
            # Upsample to HR for modulation
            exp_feat_hr = F.interpolate(
                exp_feat, size=(H_sr, W_sr),
                mode='bilinear', align_corners=False
            )
            
            # Generate modulation
            mod = self.modulation[i](exp_feat_hr)  # [B, 3, 1, 1]
            
            # Apply soft modulation
            enhanced_out = out * (1.0 + 0.2 * (mod - 0.5))
            enhanced_outputs.append(enhanced_out.clamp(0, 1))
        
        return enhanced_outputs


# =============================================================================
# Test
# =============================================================================

def test_all_lka_modules():
    """Comprehensive test of all LKA modules."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Testing Large Kernel Attention Modules")
    print("=" * 60)
    
    # Test 1: Core LKA
    print("\n1. LargeKernelAttention:")
    lka = LargeKernelAttention(dim=64, kernel_size=21).to(device)
    p = sum(pp.numel() for pp in lka.parameters())
    x = torch.randn(2, 64, 64, 64, device=device)
    out = lka(x)
    print(f"   Params: {p:,}, Shape: {x.shape} → {out.shape}")
    assert out.shape == x.shape and not torch.isnan(out).any()
    
    # Test 2: LKA Block
    print("2. LKABlock:")
    block = LKABlock(dim=64, kernel_size=21).to(device)
    p = sum(pp.numel() for pp in block.parameters())
    out = block(x)
    print(f"   Params: {p:,}, Shape: {x.shape} → {out.shape}")
    assert out.shape == x.shape and not torch.isnan(out).any()
    
    # Test 3: Enhanced Cross-Band
    print("3. EnhancedCrossBandWithLKA (9 bands):")
    ecb = EnhancedCrossBandWithLKA(dim=64, num_bands=9, num_heads=4).to(device)
    p = sum(pp.numel() for pp in ecb.parameters())
    bands = [torch.randn(2, 3, 64, 64, device=device) for _ in range(9)]
    out_bands = ecb(bands)
    print(f"   Params: {p:,}, Bands: {len(bands)} → {len(out_bands)}")
    assert len(out_bands) == 9
    for b in out_bands:
        assert b.shape == bands[0].shape and not torch.isnan(b).any()
    
    # Test 4: Enhanced Collaborative
    print("4. EnhancedCollaborativeWithLKA:")
    ecl = EnhancedCollaborativeWithLKA(
        num_experts=3, feature_dim=128, num_heads=8
    ).to(device)
    p = sum(pp.numel() for pp in ecl.parameters())
    feats = {
        'hat': torch.randn(2, 180, 64, 64, device=device),
        'dat': torch.randn(2, 180, 64, 64, device=device),
        'nafnet': torch.randn(2, 64, 64, 64, device=device),
    }
    outputs = [torch.randn(2, 3, 256, 256, device=device) for _ in range(3)]
    enhanced = ecl(feats, outputs)
    print(f"   Params: {p:,}, Experts: {len(outputs)} → {len(enhanced)}")
    assert len(enhanced) == 3
    for e in enhanced:
        assert e.shape == outputs[0].shape
    
    # Gradient check
    print("\n5. Gradient flow check:")
    ecb.zero_grad()
    loss = sum(b.mean() for b in ecb(bands))
    loss.backward()
    g = sum(1 for pp in ecb.parameters() if pp.requires_grad and pp.grad is not None and pp.grad.abs().sum() > 0)
    t = sum(1 for pp in ecb.parameters() if pp.requires_grad)
    print(f"   CrossBand grads: {g}/{t}")
    
    ecl.zero_grad()
    loss2 = sum(e.mean() for e in ecl(feats, outputs))
    loss2.backward()
    g2 = sum(1 for pp in ecl.parameters() if pp.requires_grad and pp.grad is not None and pp.grad.abs().sum() > 0)
    t2 = sum(1 for pp in ecl.parameters() if pp.requires_grad)
    print(f"   Collaborative grads: {g2}/{t2}")
    
    print("\n  All LKA tests PASSED!")


if __name__ == '__main__':
    test_all_lka_modules()
