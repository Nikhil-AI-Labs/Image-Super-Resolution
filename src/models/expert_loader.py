"""
Expert Loader Module
====================
Loads HAT, MambaIR, NAFNet as frozen experts for the multi-expert fusion pipeline.
Based on NTIRE 2025 winning strategies (Samsung 1st Track A, SNUCV 1st Track B).

This module handles:
1. Model initialization for each expert architecture
2. Pretrained weight loading with flexible checkpoint handling
3. Freezing all parameters (experts are not trained!)
4. Padding/window handling for window-based attention models
5. Batch inference support

Usage:
------
    from src.models.expert_loader import ExpertEnsemble
    
    # Initialize and load all experts
    ensemble = ExpertEnsemble(device='cuda')
    ensemble.load_all_experts()
    
    # Run inference
    lr_image = torch.randn(1, 3, 256, 256).cuda()
    expert_outputs = ensemble.forward_all(lr_image)
    # expert_outputs = [hat_sr, mambair_sr, nafnet_sr], each [1, 3, 1024, 1024]

Author: NTIRE SR Team
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings


# ============================================================================
# Utility Functions
# ============================================================================

def pad_to_window_size(
    x: torch.Tensor, 
    window_size: int, 
    scale: int = 4
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Pad input tensor to be divisible by window_size.
    
    Required for window-based attention models like HAT.
    
    Args:
        x: Input tensor [B, C, H, W]
        window_size: Window size for attention
        scale: Upscaling factor
        
    Returns:
        Tuple of (padded_tensor, original_size, padded_size)
    """
    _, _, h, w = x.shape
    
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    
    if pad_h == 0 and pad_w == 0:
        return x, (h, w), (h, w)
    
    padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    return padded_x, (h, w), (h + pad_h, w + pad_w)


def crop_to_size(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Crop tensor back to target size after inference."""
    return x[:, :, :target_h, :target_w]


def load_checkpoint_flexible(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = False
) -> Tuple[nn.Module, Dict]:
    """
    Flexibly load checkpoint handling different formats.
    
    Handles:
    - Direct state_dict
    - state_dict under 'state_dict' key
    - state_dict under 'params' or 'params_ema' keys (BasicSR format)
    - Module prefix removal ('module.' from DDP)
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        strict: Whether to require exact key matching
        
    Returns:
        Tuple of (model, info_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict from various formats
    if 'params_ema' in ckpt:
        state_dict = ckpt['params_ema']
    elif 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = ckpt
    
    # Remove 'module.' prefix from DDP training
    state_dict = OrderedDict(
        (k.replace('module.', ''), v) for k, v in state_dict.items()
    )
    
    # Load into model
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key in state_dict:
        if key in model_state:
            if state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: shape mismatch")
        else:
            skipped_keys.append(f"{key}: not in model")
    
    model.load_state_dict(model_state, strict=False)
    
    info = {
        'loaded': len(loaded_keys),
        'skipped': len(skipped_keys),
        'total': len(model_state),
        'skipped_keys': skipped_keys[:5] if skipped_keys else []
    }
    
    return model, info


# ============================================================================
# ExpertEnsemble - Main Class
# ============================================================================

class ExpertEnsemble(nn.Module):
    """
    Multi-Expert Ensemble for Super-Resolution.
    
    Manages HAT, MambaIR, and NAFNet experts as frozen feature extractors.
    Based on NTIRE 2025 winning approaches.
    
    Attributes:
        hat: HAT-L model (Samsung 1st place Track A)
        mambair: MambaIR model (SNUCV 1st place Track B)
        nafnet: NAFNet-SR model (Samsung's partner)
        upscale: Upscaling factor (default 4)
        window_size: Window size for HAT (default 16)
    """
    
    def __init__(
        self,
        upscale: int = 4,
        window_size: int = 16,
        device: Union[str, torch.device] = 'cuda',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize ExpertEnsemble.
        
        Args:
            upscale: Upscaling factor
            window_size: Window size for HAT
            device: Device to load models on
            checkpoint_dir: Directory containing pretrained weights
        """
        super().__init__()
        
        self.upscale = upscale
        self.window_size = window_size
        self.device = torch.device(device)
        
        # Default checkpoint directory
        if checkpoint_dir is None:
            # Try to find project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            checkpoint_dir = project_root / 'checkpoints' / 'pretrained_weights'
        
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Expert models (initialized as None)
        self.hat = None
        self.mambair = None
        self.nafnet = None
        
        # Track which experts are loaded
        self._experts_loaded = {
            'hat': False,
            'mambair': False,
            'nafnet': False
        }
    
    def _setup_basicsr_mocks(self):
        """Setup basicsr mocks for HAT import."""
        import types
        from timm.models.layers import to_2tuple, trunc_normal_
        
        class MockRegistry:
            def __init__(self):
                self._obj_map = {}
            
            def register(self, name=None):
                def decorator(cls):
                    return cls
                return decorator
            
            def get(self, name):
                return self._obj_map.get(name)
        
        # Create mock modules
        if 'basicsr' not in sys.modules:
            sys.modules['basicsr'] = types.ModuleType('basicsr')
        
        if 'basicsr.utils' not in sys.modules:
            sys.modules['basicsr.utils'] = types.ModuleType('basicsr.utils')
        
        if 'basicsr.utils.registry' not in sys.modules:
            registry_module = types.ModuleType('basicsr.utils.registry')
            registry_module.ARCH_REGISTRY = MockRegistry()
            sys.modules['basicsr.utils.registry'] = registry_module
        else:
            sys.modules['basicsr.utils.registry'].ARCH_REGISTRY = MockRegistry()
        
        if 'basicsr.archs' not in sys.modules:
            sys.modules['basicsr.archs'] = types.ModuleType('basicsr.archs')
        
        if 'basicsr.archs.arch_util' not in sys.modules:
            arch_util = types.ModuleType('basicsr.archs.arch_util')
            arch_util.to_2tuple = to_2tuple
            arch_util.trunc_normal_ = trunc_normal_
            sys.modules['basicsr.archs.arch_util'] = arch_util
    
    def load_hat(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True
    ) -> bool:
        """
        Load HAT-L model with pretrained weights.
        
        HAT-L configuration (NTIRE 2025 winner):
        - embed_dim: 180
        - depths: [6] * 12 (12 stages, 6 blocks each)
        - num_heads: [6] * 12
        - window_size: 16
        
        Args:
            checkpoint_path: Path to HAT checkpoint
            freeze: Whether to freeze model parameters
            
        Returns:
            True if successful
        """
        try:
            # Setup mocks for HAT import
            self._setup_basicsr_mocks()
            
            # Import HAT
            from src.models.hat import create_hat_model
            
            # Create model
            self.hat = create_hat_model(
                embed_dim=180,
                depths=[6] * 12,
                num_heads=[6] * 12,
                window_size=self.window_size,
                upscale=self.upscale,
                img_range=1.0
            )
            
            # Load checkpoint if provided
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'hat' / 'HAT-L_SRx4_ImageNet-pretrain.pth'
            
            if os.path.exists(checkpoint_path):
                self.hat, info = load_checkpoint_flexible(checkpoint_path, self.hat)
                print(f"✓ HAT loaded: {info['loaded']}/{info['total']} params")
            else:
                print(f"⚠ HAT checkpoint not found: {checkpoint_path}")
                print("  Model initialized with random weights")
            
            # Freeze if requested
            if freeze:
                for param in self.hat.parameters():
                    param.requires_grad = False
                self.hat.eval()
            
            self.hat = self.hat.to(self.device)
            self._experts_loaded['hat'] = True
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load HAT: {e}")
            return False
    
    def load_mambair(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True
    ) -> bool:
        """
        Load MambaIR model with pretrained weights.
        
        NOTE: MambaIR requires mamba-ssm package with CUDA compilation.
        
        Args:
            checkpoint_path: Path to MambaIR checkpoint
            freeze: Whether to freeze model parameters
            
        Returns:
            True if successful
        """
        try:
            from src.models.mambair import create_mambair_model, MAMBA_AVAILABLE
            
            if not MAMBA_AVAILABLE:
                print("⚠ MambaIR not available (mamba-ssm not installed)")
                return False
            
            # Create model
            self.mambair = create_mambair_model(
                upscale=self.upscale,
                embed_dim=180,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                window_size=8,
                img_range=1.0
            )
            
            # Load checkpoint if provided
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'mambair' / 'MambaIR_SR4_x4.pth'
            
            if os.path.exists(checkpoint_path):
                self.mambair, info = load_checkpoint_flexible(checkpoint_path, self.mambair)
                print(f"✓ MambaIR loaded: {info['loaded']}/{info['total']} params")
            else:
                print(f"⚠ MambaIR checkpoint not found: {checkpoint_path}")
                print("  Model initialized with random weights")
            
            # Freeze if requested
            if freeze:
                for param in self.mambair.parameters():
                    param.requires_grad = False
                self.mambair.eval()
            
            self.mambair = self.mambair.to(self.device)
            self._experts_loaded['mambair'] = True
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load MambaIR: {e}")
            return False
    
    def load_nafnet(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True
    ) -> bool:
        """
        Load NAFNet-SR model with pretrained weights.
        
        NOTE: NAFNet-SIDD is a denoising model, but we adapt it for SR.
        
        Args:
            checkpoint_path: Path to NAFNet checkpoint (optional)
            freeze: Whether to freeze model parameters
            
        Returns:
            True if successful
        """
        try:
            from src.models.nafnet import create_nafnet_sr_model
            
            # Create model
            self.nafnet = create_nafnet_sr_model(
                upscale=self.upscale,
                width=64,
                middle_blk_num=12
            )
            
            # Note: NAFNet-SIDD weights are for denoising, not SR
            # We initialize with random weights for the SR variant
            # unless a specific SR checkpoint is provided
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'nafnet' / 'NAFNet-SIDD-width64.pth'
            
            if os.path.exists(checkpoint_path):
                # Try to load - may have shape mismatches since it's denoising model
                try:
                    self.nafnet, info = load_checkpoint_flexible(checkpoint_path, self.nafnet)
                    print(f"✓ NAFNet loaded: {info['loaded']}/{info['total']} params")
                except Exception as e:
                    print(f"⚠ NAFNet checkpoint incompatible (denoising → SR): {e}")
                    print("  Using random initialization for SR")
            else:
                print(f"⚠ NAFNet checkpoint not found: {checkpoint_path}")
                print("  Model initialized with random weights")
            
            # Freeze if requested
            if freeze:
                for param in self.nafnet.parameters():
                    param.requires_grad = False
                self.nafnet.eval()
            
            self.nafnet = self.nafnet.to(self.device)
            self._experts_loaded['nafnet'] = True
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load NAFNet: {e}")
            return False
    
    def load_all_experts(
        self,
        checkpoint_paths: Optional[Dict[str, str]] = None,
        freeze: bool = True
    ) -> Dict[str, bool]:
        """
        Load all available expert models.
        
        Args:
            checkpoint_paths: Dict mapping expert name to checkpoint path
            freeze: Whether to freeze model parameters
            
        Returns:
            Dict mapping expert name to success status
        """
        if checkpoint_paths is None:
            checkpoint_paths = {}
        
        results = {}
        
        print("\n" + "=" * 60)
        print("Loading Expert Models")
        print("=" * 60)
        
        # Load HAT
        results['hat'] = self.load_hat(
            checkpoint_path=checkpoint_paths.get('hat'),
            freeze=freeze
        )
        
        # Load MambaIR
        results['mambair'] = self.load_mambair(
            checkpoint_path=checkpoint_paths.get('mambair'),
            freeze=freeze
        )
        
        # Load NAFNet
        results['nafnet'] = self.load_nafnet(
            checkpoint_path=checkpoint_paths.get('nafnet'),
            freeze=freeze
        )
        
        print("=" * 60)
        loaded = sum(results.values())
        print(f"Loaded {loaded}/3 experts")
        print("=" * 60 + "\n")
        
        return results
    
    @torch.no_grad()
    def forward_hat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run HAT inference with proper window padding.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            SR image [B, 3, H*scale, W*scale]
        """
        if self.hat is None:
            raise RuntimeError("HAT not loaded. Call load_hat() first.")
        
        _, _, h, w = x.shape
        
        # Pad to window size
        x_padded, (orig_h, orig_w), (padded_h, padded_w) = pad_to_window_size(
            x, self.window_size, self.upscale
        )
        
        # Forward pass
        sr_padded = self.hat(x_padded)
        
        # Crop to target size
        target_h = h * self.upscale
        target_w = w * self.upscale
        sr = crop_to_size(sr_padded, target_h, target_w)
        
        return sr.clamp(0, 1)
    
    @torch.no_grad()
    def forward_mambair(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run MambaIR inference.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            SR image [B, 3, H*scale, W*scale]
        """
        if self.mambair is None:
            raise RuntimeError("MambaIR not loaded. Call load_mambair() first.")
        
        sr = self.mambair(x)
        return sr.clamp(0, 1)
    
    @torch.no_grad()
    def forward_nafnet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run NAFNet-SR inference.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            SR image [B, 3, H*scale, W*scale]
        """
        if self.nafnet is None:
            raise RuntimeError("NAFNet not loaded. Call load_nafnet() first.")
        
        sr = self.nafnet(x)
        return sr.clamp(0, 1)
    
    @torch.no_grad()
    def forward_all(
        self, 
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run inference on all loaded experts.
        
        Args:
            x: Input LR image [B, 3, H, W]
            return_dict: If True, return dict with expert names as keys
            
        Returns:
            List or Dict of SR outputs from each expert
        """
        outputs = {}
        
        if self._experts_loaded['hat']:
            outputs['hat'] = self.forward_hat(x)
        
        if self._experts_loaded['mambair']:
            outputs['mambair'] = self.forward_mambair(x)
        
        if self._experts_loaded['nafnet']:
            outputs['nafnet'] = self.forward_nafnet(x)
        
        if return_dict:
            return outputs
        else:
            return list(outputs.values())
    
    def get_loaded_experts(self) -> List[str]:
        """Get list of successfully loaded expert names."""
        return [name for name, loaded in self._experts_loaded.items() if loaded]
    
    def __repr__(self) -> str:
        loaded = self.get_loaded_experts()
        return f"ExpertEnsemble(experts={loaded}, upscale={self.upscale})"


# ============================================================================
# Test Function
# ============================================================================

def test_expert_loader():
    """Quick test to verify expert loader works."""
    print("\n" + "=" * 60)
    print("Testing Expert Loader")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create ensemble
    ensemble = ExpertEnsemble(device=device)
    
    # Try to load HAT and NAFNet (MambaIR may fail without mamba-ssm)
    results = ensemble.load_all_experts()
    
    print(f"\nLoaded experts: {ensemble.get_loaded_experts()}")
    
    # Test forward pass with dummy input
    if any(results.values()):
        x = torch.randn(1, 3, 64, 64).to(device)
        outputs = ensemble.forward_all(x, return_dict=True)
        
        print("\nForward pass results:")
        for name, output in outputs.items():
            print(f"  {name}: {output.shape}")
        
        print("\n✓ Expert loader test passed!")
    else:
        print("\n⚠ No experts loaded - check checkpoint paths")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_expert_loader()
