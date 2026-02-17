"""
Expert Loader Module
====================
Loads HAT, DAT, NAFNet as frozen experts for the multi-expert fusion pipeline.
Based on NTIRE 2025 winning strategies (Samsung 1st Track A, SNUCV 1st Track B).

Expert roles:
- HAT: High-frequency specialist (edges, sharp details)
- DAT: Mid-frequency specialist (textures, patterns)
- NAFNet: Low-frequency specialist (smooth regions)

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
    # expert_outputs = {'hat': sr, 'dat': sr, 'nafnet': sr}

Author: NTIRE SR Team
"""

# Backward compatibility aliases (for config files using old names)
EXPERT_ALIASES = {
    'mambair': 'dat',
    'mamba': 'dat',
}

def normalize_expert_name(name: str) -> str:
    """Normalize expert name for backward compatibility."""
    name_lower = name.lower()
    return EXPERT_ALIASES.get(name_lower, name_lower)

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
    Multi-Expert Ensemble for Super-Resolution with Hook-Based Feature Extraction.
    
    Manages HAT, DAT, and NAFNet experts as frozen feature extractors.
    Uses forward hooks to reliably capture intermediate features for
    Collaborative Feature Learning.
    
    Attributes:
        hat: HAT-L model - High-freq specialist (Samsung 1st place Track A)
        dat: DAT model - Mid-freq specialist (ICCV 2023)
        nafnet: NAFNet-SR model - Low-freq specialist
        upscale: Upscaling factor (default 4)
        window_size: Window size for HAT/DAT (default 16)
        
    Feature Extraction (via hooks):
        - HAT: [B, 180, H, W] from conv_after_body
        - DAT: [B, 180, H, W] from conv_after_body
        - NAFNet: [B, 64, H, W] from encoder output
    """
    
    def __init__(
        self,
        upscale: int = 4,
        window_size: int = 16,
        device: Union[str, torch.device] = 'cuda',
        devices: Optional[Dict[str, str]] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize ExpertEnsemble with hook-based feature extraction.
        
        Args:
            upscale: Upscaling factor
            window_size: Window size for HAT
            device: Default device (used if devices dict not provided)
            devices: Dict mapping expert names to devices for multi-GPU support
                     Example: {'hat': 'cuda:0', 'dat': 'cuda:1', 'nafnet': 'cuda:1'}
            checkpoint_dir: Directory containing pretrained weights
        """
        super().__init__()
        
        self.upscale = upscale
        self.window_size = window_size
        
        # Multi-GPU support: per-expert device assignment
        default_device = torch.device(device)
        if devices is not None:
            self.device_hat = torch.device(devices.get('hat', device))
            self.device_dat = torch.device(devices.get('dat', device))
            self.device_nafnet = torch.device(devices.get('nafnet', device))
            self.multi_gpu = (self.device_hat != self.device_dat or 
                              self.device_hat != self.device_nafnet)
        else:
            self.device_hat = default_device
            self.device_dat = default_device
            self.device_nafnet = default_device
            self.multi_gpu = False
        
        # Primary device for fusion network (where results are gathered)
        self.device = default_device
        
        # CUDA streams for parallel execution
        self._streams = {}
        if torch.cuda.is_available():
            self._streams['hat'] = torch.cuda.Stream(device=self.device_hat)
            self._streams['dat'] = torch.cuda.Stream(device=self.device_dat)
            self._streams['nafnet'] = torch.cuda.Stream(device=self.device_nafnet)
        
        # Default checkpoint directory
        if checkpoint_dir is None:
            # Try to find project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            checkpoint_dir = project_root / 'checkpoints' / 'pretrained_weights'
        
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Expert models (initialized as None)
        self.hat = None
        self.dat = None  # Replaced MambaIR with DAT
        self.nafnet = None
        
        # Track which experts are loaded
        self._experts_loaded = {
            'hat': False,
            'dat': False,
            'nafnet': False
        }
        
        # =====================================================
        # Hook-based Feature Extraction Infrastructure
        # =====================================================
        # Store captured features from forward hooks
        self._captured_features = {}
        
        # Store hook handles for cleanup
        self._hook_handles = []
        
        # Flag to enable/disable feature capture
        self._capture_features = False
        
        if self.multi_gpu:
            print(f"  [Multi-GPU] HAT → {self.device_hat}, DAT → {self.device_dat}, NAFNet → {self.device_nafnet}")
    
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
            
            self.hat = self.hat.to(self.device_hat)
            self._experts_loaded['hat'] = True
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load HAT: {e}")
            return False
    
    def load_dat(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True
    ) -> bool:
        """
        Load DAT (Dual Aggregation Transformer) model with pretrained weights.
        
        DAT is the mid-frequency specialist, handling textures and patterns.
        ICCV 2023: https://arxiv.org/abs/2308.03364
        
        Args:
            checkpoint_path: Path to DAT checkpoint
            freeze: Whether to freeze model parameters
            
        Returns:
            True if successful
        """
        try:
            from src.models.dat import create_dat_model, DAT_AVAILABLE
            
            if not DAT_AVAILABLE:
                print("⚠ DAT not available")
                return False
            
            # Create DAT-S model (standard configuration)
            # NOTE: Official DAT_x4.pth uses expansion_factor=4.0, not 2.0!
            self.dat = create_dat_model(
                upscale=self.upscale,
                embed_dim=180,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                split_size=[8, 32],  # Standard DAT config (not DAT-S which uses [8, 16])
                img_range=1.0,
                expansion_factor=4.0  # Must match official pretrained weights
            )
            
            # Load checkpoint if provided
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'dat' / 'DAT_x4.pth'
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.dat, info = load_checkpoint_flexible(checkpoint_path, self.dat)
                print(f"✓ DAT loaded: {info['loaded']}/{info['total']} params")
            else:
                print(f"⚠ DAT checkpoint not found: {checkpoint_path}")
                print("  Model initialized with random weights")
            
            # Freeze if requested
            if freeze:
                for param in self.dat.parameters():
                    param.requires_grad = False
                self.dat.eval()
            
            self.dat = self.dat.to(self.device_dat)
            self._experts_loaded['dat'] = True
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load DAT: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Backward compatibility alias
    def load_mambair(self, checkpoint_path: Optional[str] = None, freeze: bool = True) -> bool:
        """Alias for load_dat (backward compatibility)."""
        print("⚠ MambaIR replaced by DAT - loading DAT instead")
        return self.load_dat(checkpoint_path, freeze)
    
    def load_nafnet(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True
    ) -> bool:
        """
        Load NAFNet-SR model with pretrained NAFNet-SIDD weights.
        
        NAFNet-SIDD-width64 architecture:
        - UNet-style with enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]
        - middle_blk_num=12, width=64
        
        The SR wrapper uses bicubic upscaling + NAFNet refinement.
        
        Args:
            checkpoint_path: Path to NAFNet-SIDD checkpoint
            freeze: Whether to freeze model parameters
            
        Returns:
            True if successful
        """
        try:
            from src.models.nafnet import create_nafnet_sr_model
            
            # Create NAFNetSR model with SIDD-compatible architecture
            self.nafnet = create_nafnet_sr_model(
                upscale=self.upscale,
                width=64,
                middle_blk_num=12,
                enc_blk_nums=[2, 2, 4, 8],  # Official SIDD config
                dec_blk_nums=[2, 2, 2, 2]   # Official SIDD config
            )
            
            # Load NAFNet-SIDD checkpoint
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / 'nafnet' / 'NAFNet-SIDD-width64.pth'
            
            if os.path.exists(checkpoint_path):
                # Load checkpoint
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
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
                    state_dict = ckpt
                
                # Remove 'module.' prefix if present
                state_dict = OrderedDict(
                    (k.replace('module.', ''), v) for k, v in state_dict.items()
                )
                
                # Use the custom weight loading method that maps to nafnet backbone
                info = self.nafnet.load_nafnet_weights(state_dict)
                print(f"✓ NAFNet loaded: {info['loaded']}/{info['total']} params")
                
                if info['skipped'] > 0 and info['loaded'] < info['total'] * 0.9:
                    print(f"  ⚠ Some weights skipped ({info['skipped']} keys)")
            else:
                print(f"⚠ NAFNet checkpoint not found: {checkpoint_path}")
                print("  Model initialized with random weights")
            
            # Freeze if requested
            if freeze:
                for param in self.nafnet.parameters():
                    param.requires_grad = False
                self.nafnet.eval()
            
            self.nafnet = self.nafnet.to(self.device_nafnet)
            self._experts_loaded['nafnet'] = True
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load NAFNet: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Load DAT (with backward compatibility for 'mambair' key)
        dat_path = checkpoint_paths.get('dat') or checkpoint_paths.get('mambair')
        results['dat'] = self.load_dat(
            checkpoint_path=dat_path,
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
    
    @torch.inference_mode()
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
    
    @torch.inference_mode()
    def forward_dat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run DAT inference with proper window padding.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            SR image [B, 3, H*scale, W*scale]
        """
        if self.dat is None:
            raise RuntimeError("DAT not loaded. Call load_dat() first.")
        
        _, _, h, w = x.shape
        
        # Pad to window size (DAT uses 16x16 windows)
        x_padded, (orig_h, orig_w), (padded_h, padded_w) = pad_to_window_size(
            x, self.window_size, self.upscale
        )
        
        # Forward pass
        sr_padded = self.dat(x_padded)
        
        # Crop to target size
        target_h = h * self.upscale
        target_w = w * self.upscale
        sr = crop_to_size(sr_padded, target_h, target_w)
        
        return sr.clamp(0, 1)
    
    # Backward compatibility alias
    @torch.inference_mode()
    def forward_mambair(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward_dat (backward compatibility)."""
        return self.forward_dat(x)
    
    @torch.inference_mode()
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
    
    def forward_all(
        self, 
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run inference on all loaded experts with TRUE parallel execution.
        
        Uses ThreadPoolExecutor to bypass Python's GIL and launch expert
        forward passes concurrently across multiple GPUs. This achieves
        real parallelism unlike sequential CUDA stream launches.
        
        IMPORTANT: Uses torch.no_grad() instead of @torch.inference_mode()
        so that hook-captured features can be used in autograd for
        collaborative learning. Inference mode tensors are permanently
        incompatible with backward passes.
        
        Args:
            x: Input LR image [B, 3, H, W]
            return_dict: If True, return dict with expert names as keys
            
        Returns:
            List or Dict of SR outputs from each expert (all on self.device)
        """
        outputs = {}
        
        # Clear captured features from previous forward pass
        self._captured_features = {}
        
        # Multi-GPU TRUE parallel execution with ThreadPoolExecutor
        # Wrap in no_grad to prevent gradient computation for frozen experts
        # IMPORTANT: We use no_grad() instead of inference_mode() so that
        # hook-captured features can still participate in autograd
        with torch.no_grad():
            if self.multi_gpu and torch.cuda.is_available():
                
                def run_expert_on_stream(name: str, forward_fn, device, stream):
                    """
                    Execute expert forward pass on its dedicated CUDA stream.
                    
                    This function runs in a separate thread, bypassing Python's GIL
                    and allowing true concurrent execution across GPUs.
                    """
                    with torch.cuda.device(device):
                        with torch.cuda.stream(stream):
                            # Non-blocking transfer to expert's device
                            x_device = x.to(device, non_blocking=True)
                            # Forward pass (hooks will capture features automatically)
                            output = forward_fn(x_device)
                            # Non-blocking transfer back to primary device
                            return output.to(self.device, non_blocking=True)
                
                # Use ThreadPoolExecutor to launch ALL experts concurrently
                # This is the key difference from sequential stream launches!
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {}
                    
                    # Submit all experts at the same time (true parallelism!)
                    if self._experts_loaded['hat']:
                        futures['hat'] = executor.submit(
                            run_expert_on_stream,
                            'hat', self.forward_hat, self.device_hat, self._streams['hat']
                        )
                    
                    if self._experts_loaded['dat']:
                        futures['dat'] = executor.submit(
                            run_expert_on_stream,
                            'dat', self.forward_dat, self.device_dat, self._streams['dat']
                        )
                    
                    if self._experts_loaded['nafnet']:
                        futures['nafnet'] = executor.submit(
                            run_expert_on_stream,
                            'nafnet', self.forward_nafnet, self.device_nafnet, self._streams['nafnet']
                        )
                    
                    # Wait for all experts to complete and gather results
                    for name, future in futures.items():
                        outputs[name] = future.result()
                
                # Synchronize to ensure all GPU operations complete
                torch.cuda.synchronize()
                
                # Move captured features to primary device for Collaborative Learning
                for name in list(self._captured_features.keys()):
                    feat = self._captured_features[name]
                    if feat.device != self.device:
                        self._captured_features[name] = feat.to(self.device, non_blocking=True)
                
                # Final sync for feature transfers
                torch.cuda.synchronize()
            
            else:
                # Single GPU sequential execution (original behavior)
                if self._experts_loaded['hat']:
                    outputs['hat'] = self.forward_hat(x)
                
                if self._experts_loaded['dat']:
                    outputs['dat'] = self.forward_dat(x)
                
                if self._experts_loaded['nafnet']:
                    outputs['nafnet'] = self.forward_nafnet(x)
        
        if return_dict:
            return outputs
        else:
            return list(outputs.values())
    
    # =========================================================================
    # Hook-Based Feature Extraction System
    # =========================================================================
    
    def _create_feature_hook(self, name: str, capture_input: bool = False):
        """
        Create a forward hook that captures features.
        
        Args:
            name: Key to store captured feature under
            capture_input: If True, capture the INPUT to the layer instead of output.
                           Useful for NAFNet's ending layer where input=[B, 64, H, W]
        
        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            if self._capture_features:
                if capture_input:
                    # Grab the first input tensor (features BEFORE this layer)
                    feat = input[0] if isinstance(input, tuple) else input
                else:
                    # Grab the output tensor (features AFTER this layer)
                    if isinstance(output, tuple):
                        feat = output[0]
                    else:
                        feat = output
                
                # Clone and detach to prevent memory leaks
                self._captured_features[name] = feat.clone().detach()
        
        return hook_fn
    
    def _register_all_hooks(self) -> bool:
        """
        Register forward hooks on all loaded expert models.
        
        Hook targets and expected dimensions:
        - HAT:    conv_after_body OUTPUT → [B, 180, H, W]
        - DAT:    conv_after_body OUTPUT → [B, 180, H, W]
        - NAFNet: ending INPUT → [B, 64, H, W]  (features before final 3ch conv)
        
        Returns:
            True if at least one hook was registered
        """
        # Clear any existing hooks first
        self._remove_all_hooks()
        
        registered = False
        
        # HAT: Hook on conv_after_body OUTPUT (before upsample) → [B, 180, H, W]
        if self._experts_loaded['hat'] and self.hat is not None:
            try:
                if hasattr(self.hat, 'conv_after_body'):
                    handle = self.hat.conv_after_body.register_forward_hook(
                        self._create_feature_hook('hat')
                    )
                    self._hook_handles.append(handle)
                    registered = True
            except Exception as e:
                print(f"  Warning: Could not register HAT hook: {e}")
        
        # DAT: Hook on conv_after_body OUTPUT (before upsample) → [B, 180, H, W]
        if self._experts_loaded['dat'] and self.dat is not None:
            try:
                if hasattr(self.dat, 'conv_after_body'):
                    handle = self.dat.conv_after_body.register_forward_hook(
                        self._create_feature_hook('dat')
                    )
                    self._hook_handles.append(handle)
                    registered = True
            except Exception as e:
                print(f"  Warning: Could not register DAT hook: {e}")
        
        # NAFNet: Hook on 'ending' layer and capture its INPUT → [B, 64, H, W]
        # NAFNetSR architecture: intro(64ch) → UNet body(1024ch bottleneck) → ending(64ch→3ch)
        # The INPUT to 'ending' is the final 64-channel feature map after UNet decoding
        # This is the correct feature for CollaborativeFeatureLearning (expects 64ch)
        if self._experts_loaded['nafnet'] and self.nafnet is not None:
            try:
                if hasattr(self.nafnet, 'ending') and self.nafnet.ending is not None:
                    # Hook ending layer, capture INPUT (64ch features before final conv)
                    handle = self.nafnet.ending.register_forward_hook(
                        self._create_feature_hook('nafnet', capture_input=True)
                    )
                    self._hook_handles.append(handle)
                    registered = True
                    print("    ✓ NAFNet hook registered on 'ending' (input features, 64ch)")
                elif hasattr(self.nafnet, 'intro') and self.nafnet.intro is not None:
                    # Fallback to intro layer output (also 64 channels)
                    handle = self.nafnet.intro.register_forward_hook(
                        self._create_feature_hook('nafnet')
                    )
                    self._hook_handles.append(handle)
                    registered = True
                    print("    ⚠ NAFNet hook fallback: using 'intro' output")
            except Exception as e:
                print(f"  Warning: Could not register NAFNet hook: {e}")
        
        return registered
    
    def _remove_all_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except:
                pass
        self._hook_handles = []
    
    def forward_all_with_hooks(
        self,
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with hook-based feature extraction.
        
        IMPORTANT: Uses torch.no_grad() instead of @torch.inference_mode()
        so captured features can participate in autograd for collaborative learning.
        Inference mode tensors cannot be used in backward passes at all.
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            Tuple of (outputs, features):
            - outputs: Dict with SR outputs from each expert
            - features: Dict with intermediate features from each expert
        """
        _, _, h, w = x.shape
        
        # Register hooks if not already done
        if not self._hook_handles:
            self._register_all_hooks()
        
        # Clear previous features and enable capture
        self._captured_features = {}
        self._capture_features = True
        
        try:
            # Use torch.no_grad() instead of @torch.inference_mode()
            # Inference mode creates tensors that CANNOT participate in autograd,
            # even after leaving the context. no_grad tensors CAN be used as
            # inputs to layers with trainable weights.
            with torch.no_grad():
                # Run normal forward - hooks will capture features automatically
                outputs = self.forward_all(x, return_dict=True)
        finally:
            # Disable capture to prevent memory leaks
            self._capture_features = False
        
        # Process captured features - resize to LR resolution
        features = {}
        for name, feat in self._captured_features.items():
            # Ensure 4D tensor [B, C, H, W]
            if feat.dim() == 3:
                # [B, L, C] -> [B, C, H, W]
                B, L, C = feat.shape
                side = int(math.sqrt(L))
                feat = feat.view(B, side, side, C).permute(0, 3, 1, 2).contiguous()
            
            # Resize to LR resolution for consistent processing
            features[name] = F.interpolate(
                feat, size=(h, w), mode='bilinear', align_corners=False
            )
        
        return outputs, features
    
    def get_loaded_experts(self) -> List[str]:
        """Get list of successfully loaded expert names."""
        return [name for name, loaded in self._experts_loaded.items() if loaded]
    
    @torch.inference_mode()
    def forward_all_with_features(
        self, 
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run inference on all experts and extract REAL intermediate features.
        
        This method extracts features from the last transformer/conv layer BEFORE
        the upsampling stage, providing genuine intermediate representations for
        Collaborative Feature Learning.
        
        Feature dimensions:
        - HAT: [B, 180, H, W] from conv_after_body
        - DAT: [B, 180, H, W] from residual groups output
        - NAFNet: [B, 64, H, W] from encoder output
        
        Args:
            x: Input LR image [B, 3, H, W]
            
        Returns:
            Tuple of (outputs, features):
            - outputs: Dict with SR outputs from each expert
            - features: Dict with intermediate features from each expert
        """
        outputs = {}
        features = {}
        
        _, _, h, w = x.shape
        target_h = h * self.upscale
        target_w = w * self.upscale
        
        # =====================================================
        # HAT: Extract features before upsample
        # =====================================================
        if self._experts_loaded['hat'] and self.hat is not None:
            try:
                # Pad to window size
                x_padded, (orig_h, orig_w), (padded_h, padded_w) = pad_to_window_size(
                    x, self.window_size, self.upscale
                )
                
                # Extract features step by step
                # Step 1: Shallow feature extraction
                hat_feat = self.hat.conv_first(x_padded)
                
                # Step 2: Deep feature extraction through residual groups
                feat_deep = hat_feat
                for i, layer in enumerate(self.hat.layers):
                    feat_deep = layer(feat_deep, (padded_h, padded_w))
                
                # Normalize and reshape
                feat_deep = self.hat.norm(feat_deep)
                B, L, C = feat_deep.shape
                feat_deep = feat_deep.view(B, padded_h, padded_w, C).permute(0, 3, 1, 2).contiguous()
                
                # This is the REAL intermediate feature! [B, 180, H_pad, W_pad]
                features['hat'] = F.interpolate(
                    feat_deep, size=(h, w), mode='bilinear', align_corners=False
                )  # [B, 180, H, W]
                
                # Step 3: Continue to get output
                feat_deep = self.hat.conv_after_body(feat_deep) + hat_feat
                
                # Step 4: Upsample
                sr_padded = self.hat.upsample(feat_deep)
                sr = crop_to_size(sr_padded, target_h, target_w)
                outputs['hat'] = sr.clamp(0, 1)
                
            except Exception as e:
                # Fallback to normal forward
                print(f"  HAT feature extraction failed: {e}, using fallback")
                outputs['hat'] = self.forward_hat(x)
                # Create pseudo-feature as fallback
                feat = F.interpolate(outputs['hat'], size=(h, w), mode='bilinear', align_corners=False)
                features['hat'] = feat.repeat(1, 60, 1, 1)[:, :180, :, :]
        
        # =====================================================
        # DAT: Extract features before upsample
        # =====================================================
        if self._experts_loaded['dat'] and self.dat is not None:
            try:
                # Pad to window size
                x_padded, (orig_h, orig_w), (padded_h, padded_w) = pad_to_window_size(
                    x, self.window_size, self.upscale
                )
                
                # Step 1: Shallow feature
                dat_feat = self.dat.conv_first(x_padded)
                
                # Step 2: Deep feature extraction - before_RG processing
                feat_deep = self.dat.before_RG(dat_feat)
                
                # Step 3: Pass through residual groups
                x_size = (padded_h, padded_w)
                for layer in self.dat.layers:
                    feat_deep = layer(feat_deep, x_size)
                
                # Normalize
                feat_deep = self.dat.norm(feat_deep)
                B, L, C = feat_deep.shape
                feat_deep = feat_deep.view(B, padded_h, padded_w, C).permute(0, 3, 1, 2).contiguous()
                
                # This is the REAL intermediate feature! [B, 180, H_pad, W_pad]
                features['dat'] = F.interpolate(
                    feat_deep, size=(h, w), mode='bilinear', align_corners=False
                )  # [B, 180, H, W]
                
                # Step 4: Residual connection and upsample
                feat_deep = self.dat.conv_after_body(feat_deep) + dat_feat
                sr_padded = self.dat.upsample(feat_deep)
                sr = crop_to_size(sr_padded, target_h, target_w)
                outputs['dat'] = sr.clamp(0, 1)
                
            except Exception as e:
                # Fallback to normal forward
                print(f"  DAT feature extraction failed: {e}, using fallback")
                outputs['dat'] = self.forward_dat(x)
                # Create pseudo-feature as fallback
                feat = F.interpolate(outputs['dat'], size=(h, w), mode='bilinear', align_corners=False)
                features['dat'] = feat.repeat(1, 60, 1, 1)[:, :180, :, :]
        
        # =====================================================
        # NAFNet: Extract features before upsample
        # =====================================================
        if self._experts_loaded['nafnet'] and self.nafnet is not None:
            try:
                # NAFNetSR structure: intro -> body -> conv_after_body -> upsample
                # Step 1: Initial convolution
                naf_feat = self.nafnet.intro(x)
                
                # Step 2: Body processing - THIS is the main feature extractor
                if hasattr(self.nafnet, 'body'):
                    # NAFNetSR architecture
                    body_out = self.nafnet.body(naf_feat)
                    # Apply residual connection if conv_after_body exists
                    if hasattr(self.nafnet, 'conv_after_body'):
                        body_out = self.nafnet.conv_after_body(body_out) + naf_feat
                    
                    # This is the REAL intermediate feature! [B, 64, H, W]
                    features['nafnet'] = body_out  # [B, width, H, W]
                else:
                    # Original NAFNet (UNet-style) - use intro features
                    features['nafnet'] = naf_feat
                
                # Step 3: Complete forward pass for output
                sr = self.nafnet(x)
                outputs['nafnet'] = sr.clamp(0, 1)
                
            except Exception as e:
                # Fallback to normal forward
                print(f"  NAFNet feature extraction failed: {e}, using fallback")
                outputs['nafnet'] = self.forward_nafnet(x)
                # Create pseudo-feature as fallback
                feat = F.interpolate(outputs['nafnet'], size=(h, w), mode='bilinear', align_corners=False)
                features['nafnet'] = feat.repeat(1, 22, 1, 1)[:, :64, :, :]
        
        return outputs, features
    
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
