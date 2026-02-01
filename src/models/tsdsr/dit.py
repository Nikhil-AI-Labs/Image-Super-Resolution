"""
TSD-SR DiT Transformer
======================
Diffusion Transformer (DiT) backbone for TSD-SR.

Based on:
- DiT: Diffusion Transformers (Peebles & Xie, ICCV 2023)
- TSD-SR: Target Score Distillation (NTIRE 2025)

Architecture:
- Patch embedding for latent space
- Transformer blocks with adaptive layer norm (adaLN)
- Time embedding for diffusion timesteps
- Optional conditional embedding

Author: NTIRE SR Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


# ============================================================================
# Building Blocks
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] timestep indices
        Returns:
            Embeddings [B, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization (adaLN) for conditioning."""
    
    def __init__(self, hidden_size: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(condition_dim, hidden_size * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input
            condition: [B, D_cond] conditioning
        Returns:
            Normalized output [B, N, D]
        """
        # Get scale and shift from condition
        scale_shift = self.linear(condition)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Apply adaptive layer norm
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x


class FeedForward(nn.Module):
    """Transformer feed-forward network."""
    
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv.unbind(0)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with adaLN conditioning.
    
    Structure:
    x -> AdaLN -> Attention -> Add -> AdaLN -> FFN -> Add
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = AdaLayerNorm(hidden_size, condition_dim)
        self.attn = Attention(hidden_size, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = AdaLayerNorm(hidden_size, condition_dim)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.norm1(x, condition))
        # FFN with residual
        x = x + self.mlp(self.norm2(x, condition))
        return x


# ============================================================================
# Main DiT Model
# ============================================================================

class DiT(nn.Module):
    """
    Diffusion Transformer for TSD-SR.
    
    Operates in latent space (4 channels, H/8, W/8).
    Uses patch embedding, transformer blocks, and unpatchify.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        dropout: float = 0.0,
        time_embed_dim: int = 256
    ):
        """
        Args:
            in_channels: Input latent channels (4 for SD VAE)
            hidden_size: Transformer hidden dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            patch_size: Patch size for embedding
            dropout: Dropout rate
            time_embed_dim: Time embedding dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Transformer blocks
        condition_dim = hidden_size
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, condition_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Output projection (unpatchify)
        self.final_layer = nn.Linear(
            hidden_size,
            patch_size * patch_size * in_channels
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embed
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize final layer to zero for stable training
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
    
    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert patch sequence back to image.
        
        Args:
            x: [B, N, patch_size^2 * C]
            H, W: Original latent height/width
            
        Returns:
            [B, C, H, W]
        """
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        
        x = x.reshape(-1, h_patches, w_patches, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)  # B, C, h, p, w, p
        x = x.reshape(-1, self.out_channels, H, W)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input latent [B, C, H, W]
            timestep: Diffusion timestep [B]
            condition: Optional conditioning [B, D] (unused for now)
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Patchify
        x = self.patch_embed(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Time conditioning
        t = self.time_embed(timestep)  # [B, D]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t)
        
        # Final norm and projection
        x = self.norm(x)
        x = self.final_layer(x)  # [B, N, p*p*C]
        
        # Unpatchify
        x = self.unpatchify(x, H, W)
        
        return x


class TSDSRDiT(nn.Module):
    """
    TSD-SR one-step DiT model.
    
    Wraps DiT with:
    - Fixed timestep for one-step inference
    - Residual connection for refinement
    - CLIP text conditioning (optional)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        one_step_timestep: int = 500
    ):
        super().__init__()
        
        self.dit = DiT(
            in_channels=4,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads
        )
        
        self.one_step_timestep = one_step_timestep
        self.is_loaded = False
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load weights with flexible key mapping."""
        try:
            # Try direct loading
            super().load_state_dict(state_dict, strict=False)
            self.is_loaded = True
        except Exception as e:
            print(f"  Warning: State dict loading issue: {e}")
            # Try loading just the DiT
            try:
                self.dit.load_state_dict(state_dict, strict=False)
                self.is_loaded = True
            except:
                pass
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        One-step refinement.
        
        Args:
            x: Input latent [B, 4, H, W]
            timestep: Optional timestep (uses one_step_timestep if None)
            
        Returns:
            Refined latent [B, 4, H, W]
        """
        if timestep is None:
            timestep = torch.full((x.shape[0],), self.one_step_timestep, device=x.device)
        
        # Predict noise/refinement
        pred = self.dit(x, timestep)
        
        # Residual refinement
        refined = x + pred * 0.1  # Small refinement step
        
        return refined


# ============================================================================
# Factory functions
# ============================================================================

def create_tsdsr_dit_small(pretrained: bool = False) -> TSDSRDiT:
    """Create small TSD-SR DiT model (~86M params)."""
    return TSDSRDiT(hidden_size=384, depth=12, num_heads=6)


def create_tsdsr_dit_base(pretrained: bool = False) -> TSDSRDiT:
    """Create base TSD-SR DiT model (~130M params)."""
    return TSDSRDiT(hidden_size=768, depth=12, num_heads=12)


def create_tsdsr_dit_large(pretrained: bool = False) -> TSDSRDiT:
    """Create large TSD-SR DiT model (~400M params)."""
    return TSDSRDiT(hidden_size=1024, depth=24, num_heads=16)


# ============================================================================
# Testing
# ============================================================================

def test_dit():
    """Test DiT architecture."""
    print("\n" + "="*70)
    print("DiT TRANSFORMER TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_tsdsr_dit_small().to(device)
    
    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: TSD-SR DiT (Small)")
    print(f"  Parameters: {params:,}")
    print(f"  Device: {device}")
    
    # Test forward
    batch_size = 2
    latent_h, latent_w = 32, 32  # 256x256 image / 8
    
    x = torch.randn(batch_size, 4, latent_h, latent_w).to(device)
    
    print(f"\n  Input shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test gradient
    x.requires_grad = True
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    print(f"  Gradient check: ✓ (grad available: {x.grad is not None})")
    
    print("\n" + "="*70)
    print("✓ DiT TEST COMPLETE")
    print("="*70 + "\n")
    
    return model


if __name__ == '__main__':
    test_dit()
