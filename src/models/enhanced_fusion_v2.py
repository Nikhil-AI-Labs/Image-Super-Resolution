"""
Complete Enhanced Multi-Expert Fusion Architecture - V2  (Phase 3 — Championship)
====================================================================================
NTIRE 2025 Championship Strategy — Target: 35.5 dB PSNR

Championship Architecture:
  Phase 1: Expert Processing (frozen HAT-L + DRCT-L + GRL-B + EDSR-L)
  Phase 2: Multi-Domain Frequency Decomposition  (DCT+DWT+FFT → 9 bands → 3 guidance)  +0.15 dB
  Phase 3: Cross-Band Attention + LKA (k=21)     (9-band attention + global context)   +0.20 dB
  Phase 4: Collaborative Feature Learning + LKA   (cross-expert attention + modulation)  +0.20 dB
  Phase 5: Hierarchical Multi-Resolution Fusion   (64→128→256 progressive + 70/30 blend) +0.25 dB
  Phase 6: Dynamic Expert Selection               (pixel-difficulty gating)              +0.30 dB
  Phase 7: Deep Refinement + Laplacian Edge Enh   (CNN + bilinear residual + Laplacian)  +0.10 dB

Trainable params: ~1.2M   Frozen: ~131.7M (40.85+27.58+20.20+43.09)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union

# ── Advanced module imports (Championship Architecture) ───────────────────────
from src.models.multi_domain_frequency import MultiDomainFrequencyDecomposition
from src.models.large_kernel_attention import (
    EnhancedCrossBandWithLKA, EnhancedCollaborativeWithLKA,
)
from src.models.hierarchical_fusion import HierarchicalMultiResolutionFusion
from src.models.edge_enhancement import LaplacianPyramidRefinement


# =============================================================================
# Adaptive Frequency Decomposition  (UNCHANGED from original)
# =============================================================================

class AdaptiveFrequencyDecomposition(nn.Module):
    """
    Adaptive DCT-based frequency decomposition with LEARNABLE thresholds.

    Unlike fixed 25%-50%-25% splits, this learns optimal thresholds per-image.
    Uses soft sigmoid gates for differentiable band splitting.

    Expected gain: +0.15 dB PSNR
    """

    def __init__(self, block_size: int = 8, in_channels: int = 3):
        super().__init__()
        self.block_size = block_size
        self.register_buffer('dct_matrix', self._create_dct_matrix(block_size))
        self.register_buffer('zigzag_order', self._create_zigzag_order(block_size))

        self.threshold_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_channels * 64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

        self.low_min,  self.low_max  = 0.15, 0.40
        self.high_min, self.high_max = 0.60, 0.85

    def _create_dct_matrix(self, n: int) -> torch.Tensor:
        dct = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct[k, i] = 1.0 / math.sqrt(n)
                else:
                    dct[k, i] = math.sqrt(2.0 / n) * math.cos(
                        math.pi * k * (2 * i + 1) / (2 * n)
                    )
        return dct

    def _create_zigzag_order(self, n: int) -> torch.Tensor:
        indices = torch.zeros(n, n)
        i, j = 0, 0
        for idx in range(n * n):
            indices[i, j] = idx
            if (i + j) % 2 == 0:
                if   j == n - 1: i += 1
                elif i == 0:     j += 1
                else:            i -= 1; j += 1
            else:
                if   i == n - 1: j += 1
                elif j == 0:     i += 1
                else:            i += 1; j -= 1
        return indices / (n * n - 1)

    def dct2d_block(self, x):
        return torch.matmul(torch.matmul(self.dct_matrix, x), self.dct_matrix.T)

    def idct2d_block(self, x):
        return torch.matmul(torch.matmul(self.dct_matrix.T, x), self.dct_matrix)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, H, W = x.shape
        bs = self.block_size

        thresh_raw  = self.threshold_predictor(x)
        low_thresh  = thresh_raw[:, 0:1] * (self.low_max  - self.low_min)  + self.low_min
        high_thresh = thresh_raw[:, 1:2] * (self.high_max - self.high_min) + self.high_min

        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H_pad, W_pad = x.shape

        x_blocks  = x.view(B, C, H_pad // bs, bs, W_pad // bs, bs)
        x_blocks  = x_blocks.permute(0, 1, 2, 4, 3, 5)
        dct_coeffs = self.dct2d_block(x_blocks)

        zigzag          = self.zigzag_order.view(1, 1, 1, 1, bs, bs)
        temperature     = 50.0
        low_thresh_exp  = low_thresh.view(B, 1, 1, 1, 1, 1)
        high_thresh_exp = high_thresh.view(B, 1, 1, 1, 1, 1)

        low_mask  = torch.sigmoid(temperature * (low_thresh_exp  - zigzag))
        high_mask = torch.sigmoid(temperature * (zigzag - high_thresh_exp))
        mid_mask  = torch.clamp(1.0 - low_mask - high_mask, min=0.0)

        low_blocks  = self.idct2d_block(dct_coeffs * low_mask)
        mid_blocks  = self.idct2d_block(dct_coeffs * mid_mask)
        high_blocks = self.idct2d_block(dct_coeffs * high_mask)

        def blocks_to_image(blocks):
            blocks = blocks.permute(0, 1, 2, 4, 3, 5)
            img    = blocks.contiguous().view(B, C, H_pad, W_pad)
            if pad_h > 0 or pad_w > 0:
                img = img[:, :, :H, :W]
            return img

        return (
            blocks_to_image(low_blocks),
            blocks_to_image(mid_blocks),
            blocks_to_image(high_blocks),
            (low_thresh, high_thresh),
        )


# =============================================================================
# Cross-Band Attention  (UNCHANGED from original)
# =============================================================================

class CrossBandAttention(nn.Module):
    """
    Cross-Band Communication via Multi-Head Attention.

    Each pixel location attends across its 3 frequency representations
    (low, mid, high), allowing bands to share complementary information.

    Expected gain: +0.20 dB PSNR
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.band_projectors = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, 1) for _ in range(3)
        ])
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.output_projectors = nn.ModuleList([
            nn.Conv2d(hidden_dim, in_channels, 1) for _ in range(3)
        ])
        self.band_gates = nn.Parameter(torch.ones(3))

    def forward(self, bands: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(bands) == 3
        B, C, H, W = bands[0].shape

        projected   = [proj(band) for proj, band in zip(self.band_projectors, bands)]
        stacked     = torch.stack(projected, dim=1)
        reshaped    = stacked.permute(0, 3, 4, 1, 2).reshape(B * H * W, 3, self.hidden_dim)
        attn_out, _ = self.attention(reshaped, reshaped, reshaped)
        attn_reshaped = attn_out.view(B, H, W, 3, self.hidden_dim).permute(0, 3, 4, 1, 2)

        band_weights = F.softmax(self.band_gates, dim=0)
        enhanced = []
        for i, (proj, original) in enumerate(zip(self.output_projectors, bands)):
            enhanced.append(original + band_weights[i] * proj(attn_reshaped[:, i]))
        return enhanced


# =============================================================================
# Collaborative Feature Learning  (UPDATED: 4 experts, new channel map)
# =============================================================================

class CollaborativeFeatureLearning(nn.Module):
    """
    Collaborative Feature Learning via Cross-Expert Attention.

    Experts share intermediate features so they can see what others detected:
      - HAT  (180ch) shares high-freq transformer features   with DRCT / GRL / EDSR
      - DRCT (180ch) shares dense-residual features          with HAT  / GRL / EDSR
      - GRL  (180ch) shares global/regional/local features  with HAT  / DRCT / EDSR
      - EDSR (256ch) shares pure-conv deep features          with HAT  / DRCT / GRL

    Expected gain: +0.20 dB PSNR

    Phase 3 change: expert_channels default updated from
        {'hat':180,'dat':180,'nafnet':64}  →  {'hat':180,'drct':180,'grl':180,'edsr':256}
    """

    def __init__(
        self,
        expert_channels: Optional[Dict[str, int]] = None,
        common_dim: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()

        # ── Phase 3: updated defaults ────────────────────────────────────────
        if expert_channels is None:
            expert_channels = {
                'hat':  180,
                'drct': 180,
                'grl':  180,
                'edsr': 256,
            }

        self.common_dim  = common_dim
        self.num_experts = len(expert_channels)

        # Project each expert's features to a common dimension
        self.feature_projectors = nn.ModuleDict({
            name: nn.Conv2d(ch, common_dim, 1, bias=False)
            for name, ch in expert_channels.items()
        })

        self.cross_expert_attention = nn.MultiheadAttention(
            embed_dim=common_dim, num_heads=num_heads, batch_first=True
        )
        self.feature_refine = nn.Sequential(
            nn.Conv2d(common_dim, common_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(common_dim, common_dim, 3, 1, 1),
        )
        self.modulation_head = nn.Sequential(
            nn.Conv2d(common_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        expert_features: Dict[str, torch.Tensor],
        expert_outputs:  List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Args:
            expert_features: {'hat':[B,180,H,W], 'drct':[B,180,H,W],
                               'grl':[B,180,H,W], 'edsr':[B,256,H,W]}
                              (autograd-compatible — from forward_all_with_hooks)
            expert_outputs:  List of [B,3,H*4,W*4] from each expert
        Returns:
            enhanced_outputs: List of [B,3,H*4,W*4] modulated outputs
        """
        projected = {}
        for name, feat in expert_features.items():
            if name in self.feature_projectors:
                projected[name] = self.feature_projectors[name](feat)

        if not projected:
            return expert_outputs

        first_feat             = next(iter(projected.values()))
        B, C, H, W             = first_feat.shape
        expert_names           = list(projected.keys())

        feat_stack    = torch.stack([projected[n] for n in expert_names], dim=1)
        feat_reshaped = feat_stack.permute(0, 3, 4, 1, 2).reshape(
            B * H * W, len(expert_names), self.common_dim
        )
        attn_out, _ = self.cross_expert_attention(
            feat_reshaped, feat_reshaped, feat_reshaped
        )
        attn_reshaped = attn_out.reshape(B, H, W, len(expert_names), self.common_dim)
        attn_reshaped = attn_reshaped.permute(0, 3, 4, 1, 2)

        consensus = self.feature_refine(attn_reshaped.mean(dim=1))

        modulations = []
        for i in range(len(expert_names)):
            modulations.append(
                self.modulation_head(attn_reshaped[:, i] + consensus)
            )

        enhanced_outputs = []
        for i, (output, mod_map) in enumerate(zip(expert_outputs, modulations)):
            _, _, H_hr, W_hr = output.shape
            mod_hr = F.interpolate(
                mod_map, size=(H_hr, W_hr), mode='bilinear', align_corners=False
            )
            enhanced_outputs.append(output * (1.0 + 0.2 * mod_hr))

        return enhanced_outputs


# =============================================================================
# Multi-Resolution Hierarchical Fusion  (UPDATED: num_experts default 3→4)
# =============================================================================

class MultiResolutionFusion(nn.Module):
    """
    Multi-Resolution Hierarchical Fusion.

    Fuses expert outputs at 3 resolutions:
      Level 1: 64×64   — coarse structure
      Level 2: 128×128 — textures
      Level 3: 256×256 — fine details

    Each level uses residuals from the previous level.

    Phase 3 change: num_experts default 3 → 4

    Expected gain: +0.25 dB PSNR
    """

    def __init__(self, num_experts: int = 4, base_channels: int = 32):
        super().__init__()
        self.num_experts = num_experts

        def _router():
            return nn.Sequential(
                nn.Conv2d(3, base_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, num_experts, 1),
                nn.Softmax(dim=1),
            )

        self.router_64  = _router()
        self.router_128 = _router()
        self.router_256 = _router()

        self.res_weight_128 = nn.Parameter(torch.tensor(0.5))
        self.res_weight_256 = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        lr_input:      torch.Tensor,
        expert_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            lr_input:       [B, 3, H, W]
            expert_outputs: List of [B, 3, H*4, W*4]
        Returns:
            fused: [B, 3, H*4, W*4]
        """
        experts_64  = [F.interpolate(e, size=64,  mode='bilinear', align_corners=False) for e in expert_outputs]
        experts_128 = [F.interpolate(e, size=128, mode='bilinear', align_corners=False) for e in expert_outputs]
        experts_256 = expert_outputs

        def _fuse(lr_sz, experts, router):
            lr   = F.interpolate(lr_input, size=lr_sz, mode='bilinear', align_corners=False)
            r    = router(lr).unsqueeze(2)             # [B, E, 1, H, W]
            stk  = torch.stack(experts, dim=1)         # [B, E, 3, H, W]
            return (stk * r).sum(dim=1)

        fused_64  = _fuse(64,  experts_64,  self.router_64)
        fused_128 = F.interpolate(fused_64, size=128, mode='bilinear', align_corners=False) \
                    + self.res_weight_128 * (_fuse(128, experts_128, self.router_128)
                                             - F.interpolate(fused_64, size=128, mode='bilinear', align_corners=False))
        fused_256 = F.interpolate(fused_128, size=256, mode='bilinear', align_corners=False) \
                    + self.res_weight_256 * (_fuse(256, experts_256, self.router_256)
                                             - F.interpolate(fused_128, size=256, mode='bilinear', align_corners=False))
        return fused_256


# =============================================================================
# Dynamic Expert Selector  (UPDATED: num_experts default 3→4)
# =============================================================================

class DynamicExpertSelector(nn.Module):
    """
    Dynamic Expert Selection based on per-pixel difficulty.

    Estimates difficulty at each pixel and gates all experts:
      - Easy pixels (sky, smooth):   most gate weight on EDSR
      - Medium pixels (texture):     DRCT + GRL dominant
      - Hard pixels (edges/details): HAT + DRCT + GRL all contribute

    Phase 3 change: num_experts default 3 → 4

    Expected gain: +0.30 dB PSNR
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

        self.difficulty_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_experts, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        lr_input: torch.Tensor,
        routing_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gates:      [B, num_experts, H, W]
            difficulty: [B, 1, H, W]
        """
        difficulty = self.difficulty_net(lr_input)
        raw_gates  = self.gate_net(lr_input)
        threshold  = 0.7 - 0.5 * difficulty
        gates      = torch.sigmoid(self.temperature * (raw_gates - threshold))
        gate_sum   = gates.sum(dim=1, keepdim=True) + 1e-8
        gates      = gates / gate_sum.clamp(min=0.3)
        return gates, difficulty


# =============================================================================
# Complete Enhanced Fusion SR  (MAJOR Phase 3 update)
# =============================================================================

class CompleteEnhancedFusionSR(nn.Module):
    """
    Complete Enhanced Multi-Expert Fusion — Phase 3.

    Integrates all 4 Phase 2 experts via forward_all_with_hooks() and
    routes their outputs + intermediate features through the 7-phase pipeline.

    Key Phase 3 differences from original V2:
      - No ExpertFeatureExtractor — Phase 2 expert_loader handles hooks natively
      - Phase 1 uses expert_ensemble.forward_all_with_hooks(lr_input)
        which returns (outputs_dict, features_dict) with autograd-compatible features
      - CollaborativeFeatureLearning uses 4-expert channel map
      - forward_with_precomputed() supports 10-20x faster cached training

    Frozen params:    ~131.7M  (HAT-L 40.85M + DRCT-L 27.58M + GRL-B 20.20M + EDSR-L 43.09M)
    Trainable params: ~1.2M    (fusion_dim=128, refine_channels=128, refine_depth=6, base_channels=64)
    """

    def __init__(
        self,
        expert_ensemble,
        num_experts:              int  = 4,
        fusion_dim:               int  = 128,   # Collaborative learning dim (Phase 4)
        refine_channels:          int  = 128,   # Phase 7 refinement width
        refine_depth:             int  = 6,     # Phase 7 refinement depth (num conv layers)
        base_channels:            int  = 64,    # Phase 5 hierarchical base channels
        block_size:               int  = 8,
        upscale:                  int  = 4,
        enable_dynamic_selection: bool = True,   # +0.30 dB
        enable_cross_band_attn:   bool = True,   # +0.20 dB
        enable_adaptive_bands:    bool = True,   # +0.15 dB
        enable_multi_resolution:  bool = True,   # +0.25 dB
        enable_collaborative:     bool = True,   # +0.20 dB
        enable_edge_enhance:      bool = True,   # +0.10 dB (Laplacian pyramid)
    ):
        super().__init__()

        self.expert_ensemble            = expert_ensemble
        self.num_experts                = num_experts
        self.upscale                    = upscale
        self.enable_dynamic_selection   = enable_dynamic_selection
        self.enable_cross_band_attn     = enable_cross_band_attn
        self.enable_adaptive_bands      = enable_adaptive_bands
        self.enable_multi_resolution    = enable_multi_resolution
        self.enable_collaborative       = enable_collaborative
        self.enable_edge_enhance        = enable_edge_enhance

        # Ensure experts are frozen (idempotent — Phase 2 already froze them)
        if expert_ensemble is not None:
            for param in self.expert_ensemble.parameters():
                param.requires_grad = False

        # ── Phase 2: Multi-Domain Frequency Decomposition (DCT+DWT+FFT) ───
        if enable_adaptive_bands:
            self.freq_decomp = MultiDomainFrequencyDecomposition(
                block_size=block_size, in_channels=3,
                fft_mask_size=64, enable_fusion=False,  # raw 9 bands used directly by Phase 3
            )

        # ── Phase 3: Cross-Band Attention + LKA (k=21) ────────────────────
        if enable_cross_band_attn:
            self.cross_band = EnhancedCrossBandWithLKA(
                dim=64, num_bands=9, num_heads=4, lka_kernel=21,
            )

        # ── Phase 4: Collaborative Feature Learning + LKA ─────────────────
        if enable_collaborative:
            self.collaborative = EnhancedCollaborativeWithLKA(
                num_experts=num_experts,
                feature_dim=fusion_dim,
                num_heads=8,
                lka_kernel=21,
            )

        # ── Phase 5: Hierarchical Multi-Resolution Fusion (64→128→256) ────
        if enable_multi_resolution:
            self.multi_res = HierarchicalMultiResolutionFusion(
                num_experts=num_experts, base_channels=base_channels,
            )
            # Frequency-guided routing  (routing_lr → 4 expert weights)
            # Bridges gradient: loss → freq_weights → routing_lr → cross_band → freq_decomp
            self.freq_weight_conv = nn.Sequential(
                nn.Conv2d(3, 16, 1), nn.GELU(),
                nn.Conv2d(16, num_experts, 1),
            )
        else:
            self.simple_fusion = nn.Conv2d(num_experts * 3, 3, 1)

        # ── Phase 6: Dynamic Expert Selection ─────────────────────────────
        if enable_dynamic_selection:
            self.dynamic_selector = DynamicExpertSelector(
                in_channels=3, hidden_dim=32, num_experts=num_experts,
            )

        # ── Phase 7: Deep Quality Refinement ──────────────────────────────
        refine_layers = [nn.Conv2d(3, refine_channels, 3, 1, 1), nn.GELU()]
        for _ in range(refine_depth - 2):
            refine_layers.extend([
                nn.Conv2d(refine_channels, refine_channels, 3, 1, 1),
                nn.GELU(),
            ])
        refine_layers.append(nn.Conv2d(refine_channels, 3, 3, 1, 1))
        self.refine = nn.Sequential(*refine_layers)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # ── Phase 7b: Laplacian Pyramid Edge Enhancement ──────────────────
        if enable_edge_enhance:
            self.edge_enhance = LaplacianPyramidRefinement(
                num_levels=3, channels=32, edge_strength=0.15,
            )

    # =========================================================================
    # MAIN FORWARD  — live experts
    # =========================================================================

    def forward(
        self,
        lr_input: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Full 7-phase forward pass using live expert models.

        Args:
            lr_input: [B, 3, H, W]  (H=W=64 for standard x4 SR training)
        Returns:
            sr_output: [B, 3, H*4, W*4]
        """
        B, C, H, W = lr_input.shape
        H_hr, W_hr = H * self.upscale, W * self.upscale
        intermediates = {}

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1: Expert Processing + Feature Extraction
        #
        # forward_all_with_hooks() runs all 4 experts under no_grad,
        # then returns:
        #   expert_outputs:  {'hat':[B,3,256,256], 'drct':..., 'grl':..., 'edsr':...}
        #   expert_features: {'hat':[B,180,H,W], 'drct':[B,180,H,W],
        #                     'grl':[B,180,H,W],  'edsr':[B,256,H,W]}
        #
        # Features are AUTOGRAD-COMPATIBLE (cloned outside inference_mode
        # in expert_loader.py — Phase 2 guarantee).
        # ─────────────────────────────────────────────────────────────────────
        expert_outputs, expert_features = \
            self.expert_ensemble.forward_all_with_hooks(lr_input)

        expert_names      = list(expert_outputs.keys())
        expert_output_list = [expert_outputs[n] for n in expert_names]

        if return_intermediates:
            intermediates['expert_outputs']  = expert_outputs
            intermediates['expert_features'] = expert_features

        return self._run_pipeline(
            lr_input, expert_output_list, expert_features,
            H_hr, W_hr, intermediates, return_intermediates
        )

    # =========================================================================
    # CACHED FORWARD  — pre-computed expert outputs (10-20× faster training)
    # =========================================================================

    def forward_with_precomputed(
        self,
        lr_input:      torch.Tensor,
        expert_imgs:   Dict[str, torch.Tensor],
        expert_feats:  Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass using PRE-COMPUTED expert outputs.

        Skips Phase 1 (expert inference) entirely — expert_imgs and
        expert_feats are loaded from disk by CachedSRDataset.

        Args:
            lr_input:     [B, 3, H, W]
            expert_imgs:  {'hat':[B,3,H*4,W*4], 'drct':..., 'grl':..., 'edsr':...}
            expert_feats: {'hat':[B,180,H,W], 'drct':..., 'grl':..., 'edsr':[B,256,H,W]}
                          (may be None or partial — collaborative phase degrades gracefully)
        Returns:
            sr_output: [B, 3, H*4, W*4]
        """
        B, C, H, W = lr_input.shape
        H_hr, W_hr = H * self.upscale, W * self.upscale

        expert_names      = list(expert_imgs.keys())
        expert_output_list = [expert_imgs[n] for n in expert_names]
        expert_features   = expert_feats if expert_feats is not None else {}

        return self._run_pipeline(
            lr_input, expert_output_list, expert_features,
            H_hr, W_hr, {}, False
        )

    # =========================================================================
    # SHARED PIPELINE  — Phases 2-7 (used by both forward paths)
    # =========================================================================

    def _run_pipeline(
        self,
        lr_input:          torch.Tensor,
        expert_output_list: List[torch.Tensor],
        expert_features:   Dict[str, torch.Tensor],
        H_hr: int,
        W_hr: int,
        intermediates: Dict,
        return_intermediates: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:

        expert_names = ['hat', 'drct', 'grl', 'edsr'][:self.num_experts]
        routing_lr = lr_input

        # ── Phase 2: Multi-Domain Frequency Decomposition (DCT+DWT+FFT) ───
        if self.enable_adaptive_bands:
            fused_guidance_bands, raw_9_bands = self.freq_decomp(
                lr_input, return_raw_bands=True,
            )
            if return_intermediates:
                intermediates['guidance_bands'] = fused_guidance_bands
                intermediates['raw_9_bands']    = raw_9_bands
        else:
            raw_9_bands = None
            fused_guidance_bands = None

        # ── Phase 3: Cross-Band Attention + LKA (k=21) ────────────────────
        if self.enable_cross_band_attn and raw_9_bands is not None:
            enhanced_9_bands = self.cross_band(raw_9_bands)
            # Reconstruct routing_lr from first 3 DCT enhanced bands
            # (DCT is orthogonal: low+mid+high ≈ lr_input + cross-band info)
            # Gradient path: loss → freq_weight_conv(routing_lr) → Σbands → cross_band → freq_decomp
            routing_lr = enhanced_9_bands[0] + enhanced_9_bands[1] + enhanced_9_bands[2]
            if return_intermediates:
                intermediates['enhanced_9_bands'] = enhanced_9_bands
                intermediates['routing_lr']       = routing_lr
        else:
            enhanced_9_bands = raw_9_bands

        # ── Phase 4: Collaborative Feature Learning + LKA ─────────────────
        if self.enable_collaborative and expert_features:
            enhanced_outputs = self.collaborative(expert_features, expert_output_list)
            if return_intermediates:
                intermediates['collaborative_outputs'] = enhanced_outputs
        else:
            enhanced_outputs = expert_output_list

        # ── Phase 5: Hierarchical Multi-Resolution Fusion (64→128→256) ────
        if self.enable_multi_resolution:
            # 5a. Hierarchical fusion (takes dict)
            expert_dict = {name: out for name, out in zip(expert_names, enhanced_outputs)}
            hierarchical_fused = self.multi_res(expert_dict)

            # 5b. Frequency-guided fusion (uses routing_lr → bridges gradient)
            routing_lr_hr = F.interpolate(
                routing_lr, size=(H_hr, W_hr),
                mode='bilinear', align_corners=False,
            )
            freq_logits = self.freq_weight_conv(routing_lr_hr)       # [B, 4, H_hr, W_hr]
            freq_weights = F.softmax(freq_logits, dim=1)
            freq_fused = sum(
                out * freq_weights[:, i:i+1]
                for i, out in enumerate(enhanced_outputs)
            )

            # 5c. Blend: 70% hierarchical + 30% frequency-guided
            fused = hierarchical_fused * 0.7 + freq_fused * 0.3
        else:
            concat = torch.cat(enhanced_outputs, dim=1)
            fused  = self.simple_fusion(concat)

        if return_intermediates:
            intermediates['fused_before_dynamic'] = fused.clone()

        # ── Phase 6: Dynamic Expert Selection ─────────────────────────────
        if self.enable_dynamic_selection:
            gates, difficulty = self.dynamic_selector(routing_lr)
            gates_hr = F.interpolate(
                gates, size=(H_hr, W_hr), mode='bilinear', align_corners=False,
            )

            gated_outputs = [
                output * gates_hr[:, i:i+1]
                for i, output in enumerate(enhanced_outputs)
            ]
            gated_stack   = torch.stack(gated_outputs, dim=0).sum(dim=0)
            gate_sum      = gates_hr.sum(dim=1, keepdim=True) + 1e-8
            dynamic_fused = gated_stack / gate_sum

            difficulty_hr = F.interpolate(
                difficulty, size=(H_hr, W_hr), mode='bilinear', align_corners=False,
            )
            blend_weight = 0.3 + 0.4 * difficulty_hr
            fused = (1 - blend_weight) * fused + blend_weight * dynamic_fused

            if return_intermediates:
                intermediates['gates']      = gates
                intermediates['difficulty'] = difficulty

        # ── Phase 7: Deep CNN Refinement ───────────────────────────────────
        fused = fused + 0.1 * self.refine(fused)

        # ── Phase 7b: Laplacian Pyramid Edge Enhancement ──────────────────
        if self.enable_edge_enhance:
            fused = self.edge_enhance(fused)

        # ── Residual connection (bilinear upscale of original LR) ─────────
        bilinear = F.interpolate(
            lr_input, size=(H_hr, W_hr), mode='bilinear', align_corners=False,
        )
        final_sr = (fused + self.residual_scale * bilinear).clamp(0, 1)

        if return_intermediates:
            return final_sr, intermediates
        return final_sr

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        if self.expert_ensemble is not None:
            return sum(p.numel() for p in self.expert_ensemble.parameters())
        return 0

    def get_improvement_status(self) -> Dict[str, bool]:
        return {
            'dynamic_expert_selection':  self.enable_dynamic_selection,
            'cross_band_attention':      self.enable_cross_band_attn,
            'adaptive_frequency_bands':  self.enable_adaptive_bands,
            'multi_resolution_fusion':   self.enable_multi_resolution,
            'collaborative_learning':    self.enable_collaborative,
            'edge_enhancement':          self.enable_edge_enhance,
        }

    def __repr__(self) -> str:
        return (
            f"CompleteEnhancedFusionSR("
            f"num_experts={self.num_experts}, "
            f"trainable={self.get_trainable_params():,}, "
            f"frozen={self.get_frozen_params():,})"
        )


# =============================================================================
# Factory Function  (UPDATED: num_experts default 3→4)
# =============================================================================

def create_enhanced_fusion(
    expert_ensemble,
    config: Optional[Dict] = None,
) -> CompleteEnhancedFusionSR:
    """
    Create CompleteEnhancedFusionSR with optional config overrides.

    Args:
        expert_ensemble: Loaded ExpertEnsemble (Phase 2)
        config: Optional dict — keys override defaults below

    Returns:
        CompleteEnhancedFusionSR instance
    """
    default_config = {
        'num_experts':              4,
        'fusion_dim':               128,    # ~1.2M params target
        'refine_channels':          128,
        'refine_depth':             6,
        'base_channels':            64,
        'block_size':               8,
        'upscale':                  4,
        'enable_dynamic_selection': True,
        'enable_cross_band_attn':   True,
        'enable_adaptive_bands':    True,
        'enable_multi_resolution':  True,
        'enable_collaborative':     True,
        'enable_edge_enhance':      True,
    }
    if config:
        default_config.update(config)
    return CompleteEnhancedFusionSR(
        expert_ensemble=expert_ensemble,
        **default_config
    )


# =============================================================================
# Phase 3 Verification Test
# =============================================================================

def test_phase3():
    """
    Championship Architecture verification.

    Tests advanced modules + 7-phase pipeline + gradient flow.
    """
    import traceback

    print("\n" + "=" * 65)
    print("  Championship Architecture Verification")
    print("  CompleteEnhancedFusionSR  (4 experts, advanced modules)")
    print("=" * 65)

    # ── Mock Expert Ensemble  ──────────────────────────────────────────
    class MockExpertEnsemble(nn.Module):
        def __init__(self):
            super().__init__()
            self.hat  = nn.Identity()
            self.drct = nn.Identity()
            self.grl  = nn.Identity()
            self.edsr = nn.Identity()

        def forward_all_with_hooks(self, x):
            B, C, H, W = x.shape
            H_hr = H * 4
            with torch.no_grad():
                outputs = {
                    'hat':  F.interpolate(x, H_hr, mode='bilinear', align_corners=False),
                    'drct': F.interpolate(x, H_hr, mode='bilinear', align_corners=False),
                    'grl':  F.interpolate(x, H_hr, mode='bilinear', align_corners=False),
                    'edsr': F.interpolate(x, H_hr, mode='bilinear', align_corners=False),
                }
            features = {
                'hat':  torch.randn(B, 180, H, W, requires_grad=False),
                'drct': torch.randn(B, 180, H, W, requires_grad=False),
                'grl':  torch.randn(B, 180, H, W, requires_grad=False),
                'edsr': torch.randn(B, 256, H, W, requires_grad=False),
            }
            return outputs, features

    mock_ensemble = MockExpertEnsemble()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(1, 3, 64, 64).to(device)
    all_pass = True

    # ── 1. Multi-Domain Frequency Decomposition (9 bands) ─────────────────
    print("\n[1] MultiDomainFrequencyDecomposition (DCT+DWT+FFT)")
    try:
        mdf = MultiDomainFrequencyDecomposition(block_size=8).to(device)
        fused_bands, raw_bands = mdf(x, return_raw_bands=True)
        ok = len(fused_bands) == 3 and len(raw_bands) == 9
        ok = ok and all(b.shape == x.shape for b in fused_bands)
        all_pass = all_pass and ok
        p = sum(pp.numel() for pp in mdf.parameters())
        print(f"    3 guidance bands + 9 raw bands.  Params: {p:,}  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); all_pass = False

    # ── 2. EnhancedCrossBandWithLKA (9 bands) ──────────────────────────
    print("\n[2] EnhancedCrossBandWithLKA (dim=64, 9 bands, k=21)")
    try:
        ecb = EnhancedCrossBandWithLKA(dim=64, num_bands=9).to(device)
        enh_bands = ecb(raw_bands)
        ok = len(enh_bands) == 9 and all(b.shape == x.shape for b in enh_bands)
        all_pass = all_pass and ok
        p = sum(pp.numel() for pp in ecb.parameters())
        print(f"    9 enhanced bands.  Params: {p:,}  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); all_pass = False

    # ── 3. EnhancedCollaborativeWithLKA (4 experts) ─────────────────────
    print("\n[3] EnhancedCollaborativeWithLKA (4 experts, dim=128)")
    try:
        ecl = EnhancedCollaborativeWithLKA(
            num_experts=4, feature_dim=128
        ).to(device)
        feats = {
            'hat':  torch.randn(1, 180, 64, 64).to(device),
            'drct': torch.randn(1, 180, 64, 64).to(device),
            'grl':  torch.randn(1, 180, 64, 64).to(device),
            'edsr': torch.randn(1, 256, 64, 64).to(device),
        }
        outs = [torch.randn(1, 3, 256, 256).to(device)] * 4
        enh = ecl(feats, outs)
        ok = len(enh) == 4 and all(e.shape == (1, 3, 256, 256) for e in enh)
        all_pass = all_pass and ok
        p = sum(pp.numel() for pp in ecl.parameters())
        print(f"    {len(enh)} enhanced outputs.  Params: {p:,}  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); all_pass = False

    # ── 4. HierarchicalMultiResolutionFusion (dict input) ─────────────────
    print("\n[4] HierarchicalMultiResolutionFusion (base_channels=64)")
    try:
        hmrf = HierarchicalMultiResolutionFusion(
            num_experts=4, base_channels=64
        ).to(device)
        expert_dict = {
            'hat':  torch.randn(1, 3, 256, 256).to(device),
            'drct': torch.randn(1, 3, 256, 256).to(device),
            'grl':  torch.randn(1, 3, 256, 256).to(device),
            'edsr': torch.randn(1, 3, 256, 256).to(device),
        }
        fuse = hmrf(expert_dict)
        ok = fuse.shape == (1, 3, 256, 256)
        all_pass = all_pass and ok
        p = sum(pp.numel() for pp in hmrf.parameters())
        print(f"    fused: {fuse.shape}  Params: {p:,}  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); all_pass = False

    # ── 5. LaplacianPyramidRefinement ──────────────────────────────────
    print("\n[5] LaplacianPyramidRefinement (3 levels, ch=32)")
    try:
        lpr = LaplacianPyramidRefinement(
            num_levels=3, channels=32, edge_strength=0.15
        ).to(device)
        sr_in = torch.randn(1, 3, 256, 256).to(device)
        sr_out = lpr(sr_in)
        ok = sr_out.shape == (1, 3, 256, 256)
        all_pass = all_pass and ok
        p = sum(pp.numel() for pp in lpr.parameters())
        print(f"    output: {sr_out.shape}  Params: {p:,}  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); all_pass = False

    # ── 6. DynamicExpertSelector (unchanged) ───────────────────────────
    print("\n[6] DynamicExpertSelector (num_experts=4)")
    des = DynamicExpertSelector(num_experts=4).to(device)
    gates, diff = des(x)
    ok = gates.shape == (1, 4, 64, 64) and diff.shape == (1, 1, 64, 64)
    all_pass = all_pass and ok
    print(f"    gates: {gates.shape}  difficulty: {diff.shape}  {'PASS' if ok else 'FAIL'}")

    # ── 7. Full model instantiation ────────────────────────────────────
    print("\n[7] CompleteEnhancedFusionSR — Championship Architecture")
    try:
        model = CompleteEnhancedFusionSR(
            expert_ensemble=mock_ensemble,
            num_experts=4,
            fusion_dim=128,
            refine_channels=128,
            refine_depth=6,
            base_channels=64,
        ).to(device)
        trainable = model.get_trainable_params()
        frozen    = model.get_frozen_params()
        print(f"    trainable: {trainable:,}  frozen: {frozen:,}  PASS")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); return False

    # ── 8. Full forward (live path) ────────────────────────────────────
    print("\n[8] forward() — 7-phase live path")
    try:
        with torch.no_grad():
            sr, ints = model(x, return_intermediates=True)
        ok = sr.shape == (1, 3, 256, 256) and 0.0 <= sr.min() and sr.max() <= 1.0
        all_pass = all_pass and ok
        print(f"    output: {sr.shape}  range: [{sr.min().item():.4f}, {sr.max().item():.4f}]")
        print(f"    intermediates: {list(ints.keys())}")
        print(f"    {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); return False

    # ── 9. Cached forward path ────────────────────────────────────────
    print("\n[9] forward_with_precomputed() — cached path")
    try:
        expert_imgs = {n: torch.randn(1, 3, 256, 256).to(device) for n in ['hat','drct','grl','edsr']}
        expert_feats = {
            'hat':  torch.randn(1, 180, 64, 64).to(device),
            'drct': torch.randn(1, 180, 64, 64).to(device),
            'grl':  torch.randn(1, 180, 64, 64).to(device),
            'edsr': torch.randn(1, 256, 64, 64).to(device),
        }
        with torch.no_grad():
            sr_cached = model.forward_with_precomputed(x, expert_imgs, expert_feats)
        ok = sr_cached.shape == (1, 3, 256, 256)
        all_pass = all_pass and ok
        print(f"    output: {sr_cached.shape}  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); return False

    # ── 10. Backward pass + gradient count ──────────────────────────────
    print("\n[10] Backward pass through fusion layers")
    try:
        model.train()
        expert_imgs_t  = {n: torch.randn(1, 3, 256, 256).to(device) for n in ['hat','drct','grl','edsr']}
        expert_feats_t = {
            'hat':  torch.randn(1, 180, 64, 64, requires_grad=True).to(device),
            'drct': torch.randn(1, 180, 64, 64, requires_grad=True).to(device),
            'grl':  torch.randn(1, 180, 64, 64, requires_grad=True).to(device),
            'edsr': torch.randn(1, 256, 64, 64, requires_grad=True).to(device),
        }
        sr_t = model.forward_with_precomputed(x, expert_imgs_t, expert_feats_t)
        loss = sr_t.mean()
        loss.backward()
        grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        ok = grads == total_trainable
        all_pass = all_pass and ok
        print(f"    {grads}/{total_trainable} params received gradients  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"    FAIL: {e}"); traceback.print_exc(); return False

    # ── 11. Improvement status ─────────────────────────────────────────
    print("\n[11] Improvement Status:")
    for name, enabled in model.get_improvement_status().items():
        print(f"    {'ON' if enabled else 'OFF'} {name}")

    print(f"\n{'='*65}")
    if all_pass:
        print(f"  Championship Architecture: ALL PASS")
    else:
        print(f"  Championship Architecture: SOME TESTS FAILED")
    print(f"{'='*65}\n")
    return all_pass


if __name__ == '__main__':
    test_phase3()
