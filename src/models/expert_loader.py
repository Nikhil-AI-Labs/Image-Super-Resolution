"""
Expert Loader Module — DRCT + GRL + NAFNet
=============================================
3-Expert local ensemble: DRCT-L, GRL-B, NAFNet-64.
MambaIR features are loaded from disk (Decoupled Compute Paradigm).

Expert roles (hook-based feature extraction):
  DRCT-L  [B, 180, H, W]  conv_after_body — dense residual connected transformer
  GRL-B   [B, 180, H, W]  conv_after_body — global/regional/local representation
  NAFNet  [B, 64,  H, W]  decoder output  — nonlinear-activation-free CNN (SIDD)

MambaIR [B, 180, H, W] — NOT loaded here. Features extracted on Colab/Kaggle
via scripts/extract_mamba_features.py and loaded by CachedSRDataset.

All local experts are FROZEN — only the downstream fusion network trains.

CRITICAL: forward_all_with_hooks() uses feat.clone() OUTSIDE @inference_mode
context to convert captured tensors into autograd-compatible tensors for
backward passes through the fusion network.

Usage:
    ensemble = ExpertEnsemble(device='cuda')
    ensemble.load_all_experts()

    outputs, features = ensemble.forward_all_with_hooks(lr_image)
    # outputs  = {'drct': [B,3,H*4,W*4], 'grl': ..., 'nafnet': ...}
    # features = {'drct': [B,180,H,W], 'grl': [B,180,H,W],
    #             'nafnet': [B,64,H,W]}
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings

# ── Expert name normalization ────────────────────────────────────────────────
EXPERT_ALIASES = {
    'mambair': 'mamba',   # normalize
    'nafnet_sidd': 'nafnet',  # normalize
}

def normalize_expert_name(name: str) -> str:
    name_lower = name.lower()
    return EXPERT_ALIASES.get(name_lower, name_lower)


# ============================================================================
# Utility Functions
# ============================================================================

def pad_to_window_size(
    x: torch.Tensor,
    window_size: int,
    scale: int = 4
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (h, w), (h, w)
    padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_x, (h, w), (h + pad_h, w + pad_w)

def crop_to_size(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return x[:, :, :target_h, :target_w]

def load_checkpoint_flexible(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = False
) -> Tuple[nn.Module, Dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

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

    state_dict = OrderedDict(
        (k.replace('module.', ''), v) for k, v in state_dict.items()
    )

    model_state  = model.state_dict()
    loaded_keys  = []
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
        'loaded':       len(loaded_keys),
        'skipped':      len(skipped_keys),
        'total':        len(model_state),
        'skipped_keys': skipped_keys[:5] if skipped_keys else [],
    }
    return model, info


def _find_pretrained_dir() -> Path:
    """
    Walk up from expert_loader.py looking for a 'pretrained/' directory.
    Fallback: cwd / 'pretrained'.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / 'pretrained').exists():
            return candidate / 'pretrained'
        candidate = candidate.parent
    return Path.cwd() / 'pretrained'


# ============================================================================
# ExpertEnsemble — DRCT + GRL + NAFNet (3 local experts)
# ============================================================================

class ExpertEnsemble(nn.Module):
    """
    3-Expert Local Ensemble with hook-based feature extraction.

    Local experts (loaded and run on your GPU):
        'drct':   [B, 180, H, W] — DRCT-L conv_after_body
        'grl':    [B, 180, H, W] — GRL-B  conv_after_body
        'nafnet': [B, 64,  H, W] — NAFNet decoder output (before ending conv)

    MambaIR (180ch) is NOT loaded here — its features come from
    scripts/extract_mamba_features.py (Colab/Kaggle extraction).

    Downstream feat_proj (in fusion network) must use:
        nn.Conv2d(180, fusion_dim, 1)  for drct / grl
        nn.Conv2d(64,  fusion_dim, 1)  for nafnet
    """

    # ── window sizes per expert ──────────────────────────────────────────────
    DRCT_WINDOW = 16   # DRCT-L default window_size
    GRL_WINDOW  = 8    # GRL-B default window_size

    def __init__(
        self,
        upscale:        int                        = 4,
        window_size:    int                        = 16,   # DRCT window size
        device:         Union[str, torch.device]   = 'cuda',
        checkpoint_dir: Optional[str]              = None,
    ):
        super().__init__()
        self.upscale     = upscale
        self.window_size = window_size
        self.device      = torch.device(device)

        # ── checkpoint directory ─────────────────────────────────────────────
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = _find_pretrained_dir()

        # ── expert models (3 local) ──────────────────────────────────────────
        self.drct   = None   # DRCT-L,   27.6M,  window=16
        self.grl    = None   # GRL-B,    20.2M,  window=8
        self.nafnet = None   # NAFNet-64, ~67M,  UNet (SIDD weights)

        self._experts_loaded = {
            'drct':   False,
            'grl':    False,
            'nafnet': False,
        }

        # ── hook infrastructure ──────────────────────────────────────────────
        self._captured_features = {}
        self._hook_handles      = []
        self._capture_features  = False

        print(f"  [Single-GPU] 3 local experts (DRCT+GRL+NAFNet) → {self.device}")
        print(f"  [Decoupled]  MambaIR features loaded from disk (not in ensemble)")

    # ── basicsr mock (DRCT needs it) ──────────────────────────────────────────
    def _setup_basicsr_mocks(self):
        import types
        try:
            from timm.models.layers import to_2tuple, trunc_normal_
        except ImportError:
            from timm.layers import to_2tuple, trunc_normal_

        class MockRegistry:
            def __init__(self): self._obj_map = {}
            def register(self, name=None):
                def decorator(cls): return cls
                return decorator
            def get(self, name): return self._obj_map.get(name)

        if 'basicsr' not in sys.modules:
            sys.modules['basicsr'] = types.ModuleType('basicsr')
        if 'basicsr.utils' not in sys.modules:
            sys.modules['basicsr.utils'] = types.ModuleType('basicsr.utils')
        if 'basicsr.utils.registry' not in sys.modules:
            rm = types.ModuleType('basicsr.utils.registry')
            rm.ARCH_REGISTRY = MockRegistry()
            sys.modules['basicsr.utils.registry'] = rm
        else:
            sys.modules['basicsr.utils.registry'].ARCH_REGISTRY = MockRegistry()
        if 'basicsr.archs' not in sys.modules:
            sys.modules['basicsr.archs'] = types.ModuleType('basicsr.archs')
        if 'basicsr.archs.arch_util' not in sys.modules:
            au = types.ModuleType('basicsr.archs.arch_util')
            au.to_2tuple    = to_2tuple
            au.trunc_normal_ = trunc_normal_
            sys.modules['basicsr.archs.arch_util'] = au

    # =========================================================================
    # LOAD METHODS
    # =========================================================================

    def load_drct(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ) -> bool:
        """
        Load DRCT-L (27.6M).  window_size=16,  hook→conv_after_body [B,180,H,W]

        Checkpoint: pretrained/drct/DRCT-L_X4.pth  (485.6 MB)
        basicsr mocks: handled inside drct/__init__.py at import time.
        """
        try:
            self._setup_basicsr_mocks()
            from src.models.drct import create_drct_model, DRCT_AVAILABLE

            if not DRCT_AVAILABLE:
                print("  ✗ DRCT-L  architecture not available")
                return False

            self.drct = create_drct_model(
                upscale     = self.upscale,
                img_size    = 64,
                window_size = self.DRCT_WINDOW,   # 16
                embed_dim   = 180,
                depths      = [6] * 12,
                num_heads   = [6] * 12,
                img_range   = 1.0,
                upsampler   = 'pixelshuffle',
                resi_connection = '1conv',
            )

            if checkpoint_path is None:
                checkpoint_path = str(
                    self.checkpoint_dir / 'drct' / 'DRCT-L_X4.pth'
                )
            if os.path.exists(checkpoint_path):
                self.drct, info = load_checkpoint_flexible(checkpoint_path, self.drct)
                print(f"  ✓ DRCT-L  loaded: {info['loaded']}/{info['total']} keys")
            else:
                print(f"  ⚠ DRCT-L  checkpoint not found: {checkpoint_path}")

            if freeze:
                for p in self.drct.parameters(): p.requires_grad = False
                self.drct.eval()
            self.drct = self.drct.to(self.device)
            self._experts_loaded['drct'] = True
            return True

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ DRCT-L  failed: {e}")
            return False

    def load_grl(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ) -> bool:
        """
        Load GRL-B (20.2M).  window_size=8,  hook→conv_after_body [B,180,H,W]

        Checkpoint: pretrained/grl/GRL-B_SR_x4.pth  (81.2 MB)
        1390/1390 checkpoint keys load (13 attention buffers recomputed at init).
        fairscale mock: handled inside grl/__init__.py at import time.
        """
        try:
            from src.models.grl import create_grl_model, GRL_AVAILABLE

            if not GRL_AVAILABLE:
                print("  ✗ GRL-B   architecture not available")
                return False

            self.grl = create_grl_model(
                upscale                   = self.upscale,
                img_size                  = 64,
                window_size               = self.GRL_WINDOW,   # 8
                embed_dim                 = 180,
                img_range                 = 1.0,
                local_connection          = True,
                anchor_window_down_factor = 2,
                conv_type                 = '1conv',
                mlp_ratio                 = 2.0,
            )

            if checkpoint_path is None:
                checkpoint_path = str(
                    self.checkpoint_dir / 'grl' / 'GRL-B_SR_x4.pth'
                )
            if os.path.exists(checkpoint_path):
                self.grl, info = load_checkpoint_flexible(checkpoint_path, self.grl)
                print(f"  ✓ GRL-B   loaded: {info['loaded']}/{info['total']} keys"
                      f"  (13 attention buffers recomputed at init — expected)")
            else:
                print(f"  ⚠ GRL-B   checkpoint not found: {checkpoint_path}")

            if freeze:
                for p in self.grl.parameters(): p.requires_grad = False
                self.grl.eval()
            self.grl = self.grl.to(self.device)
            self._experts_loaded['grl'] = True
            return True

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ GRL-B   failed: {e}")
            return False

    def load_nafnet(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ) -> bool:
        """
        Load NAFNet-SIDD-width64 (~67M params).
        
        Wrapped in NAFNetSR: bicubic upscale → NAFNet refinement.
        Hook target: last decoder output → [B, 64, H_up, W_up]
        Feature is resized to LR resolution → [B, 64, H, W]
        
        Checkpoint: pretrained/nafnet/NAFNet-SIDD-width64.pth
        """
        try:
            from src.models.nafnet import NAFNetSR, create_nafnet_sr_model

            self.nafnet = create_nafnet_sr_model(
                upscale=self.upscale,
                width=64,
                middle_blk_num=12,
                enc_blk_nums=[2, 2, 4, 8],
                dec_blk_nums=[2, 2, 2, 2],
            )

            if checkpoint_path is None:
                # Try both possible filenames
                for fname in [
                    'NAFNet-SIDD-width64.pth',
                    'NAFNet_SIDD_width64.pth',
                ]:
                    p = self.checkpoint_dir / 'nafnet' / fname
                    if p.exists():
                        checkpoint_path = str(p)
                        break

            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load checkpoint once (avoid double torch.load RAM spike)
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                state = ckpt.get('params', ckpt)
                self.nafnet.load_nafnet_weights(state)
                print(f"  ✓ NAFNet  loaded: {checkpoint_path}")
            else:
                print(f"  ⚠ NAFNet  checkpoint not found (searched pretrained/nafnet/)")

            if freeze:
                for p in self.nafnet.parameters(): p.requires_grad = False
                self.nafnet.eval()
            self.nafnet = self.nafnet.to(self.device)
            self._experts_loaded['nafnet'] = True
            return True

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ NAFNet  failed: {e}")
            return False

    def load_all_experts(
        self,
        checkpoint_paths: Optional[Dict[str, str]] = None,
        freeze: bool = True,
    ) -> Dict[str, bool]:
        """Load all 3 local experts.  Returns dict of name→success."""
        if checkpoint_paths is None:
            checkpoint_paths = {}

        print("\n" + "=" * 60)
        print("  Loading Expert Models  (DRCT + GRL + NAFNet)")
        print("  MambaIR: features loaded from disk (Decoupled Compute)")
        print("=" * 60)

        results = {
            'drct':   self.load_drct(checkpoint_paths.get('drct'), freeze),
            'grl':    self.load_grl(checkpoint_paths.get('grl'), freeze),
            'nafnet': self.load_nafnet(checkpoint_paths.get('nafnet'), freeze),
        }

        print("=" * 60)
        loaded = sum(results.values())
        print(f"  Loaded {loaded}/3 local experts")
        print("=" * 60 + "\n")
        return results

    # =========================================================================
    # FORWARD METHODS — individual experts
    # =========================================================================

    @torch.inference_mode()
    def forward_drct(self, x: torch.Tensor) -> torch.Tensor:
        """
        DRCT-L inference.  Pads to window_size=16, crops output.

        DRCT has internal check_image_size() but external padding ensures
        it is a no-op, keeping behaviour consistent.
        """
        if self.drct is None: raise RuntimeError("DRCT not loaded.")
        _, _, h, w = x.shape
        xp, _, _ = pad_to_window_size(x, self.DRCT_WINDOW, self.upscale)
        sr = self.drct(xp)
        return crop_to_size(sr, h * self.upscale, w * self.upscale).clamp(0, 1)

    @torch.inference_mode()
    def forward_grl(self, x: torch.Tensor) -> torch.Tensor:
        """
        GRL-B inference.  Pads to window_size=8, crops output.

        GRL has internal check_image_size() + self-cropping:
            return x[:, :, :H*upscale, :W*upscale]
        External padding ensures H,W captured by GRL equal the padded dims,
        so GRL crops to padded SR size.  We then crop to original SR size.
        """
        if self.grl is None: raise RuntimeError("GRL not loaded.")
        _, _, h, w = x.shape
        xp, _, _ = pad_to_window_size(x, self.GRL_WINDOW, self.upscale)
        sr = self.grl(xp)
        return crop_to_size(sr, h * self.upscale, w * self.upscale).clamp(0, 1)

    @torch.inference_mode()
    def forward_nafnet(self, x: torch.Tensor) -> torch.Tensor:
        """
        NAFNet-SIDD inference.

        NAFNet-SIDD is a restoration model (denoiser), NOT an upscaler.
        NAFNetSR wraps it with bicubic upscale → NAFNet refinement.
        
        Steps:
            1. Bicubic upscale LR → HR resolution
            2. NAFNet refinement (denoising/enhancement)
            3. Output [B, 3, H*4, W*4]
        """
        if self.nafnet is None: raise RuntimeError("NAFNet not loaded.")
        return self.nafnet(x).clamp(0, 1)

    # =========================================================================
    # PARALLEL FORWARD (no hooks)
    # =========================================================================

    def forward_all(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run all 3 local experts sequentially on a single GPU.
        Uses torch.no_grad() (NOT inference_mode) so hook-captured tensors remain
        autograd-compatible when cloned outside the context.
        """
        outputs = {}
        with torch.no_grad():
            if self._experts_loaded['drct']:   outputs['drct']   = self.forward_drct(x)
            if self._experts_loaded['grl']:    outputs['grl']    = self.forward_grl(x)
            if self._experts_loaded['nafnet']: outputs['nafnet'] = self.forward_nafnet(x)

        return outputs if return_dict else list(outputs.values())

    # =========================================================================
    # HOOK INFRASTRUCTURE
    # =========================================================================

    def _create_feature_hook(self, name: str, capture_input: bool = False):
        """Returns a forward hook that stores output (or input) in _captured_features."""
        def hook_fn(module, inp, out):
            if self._capture_features:
                feat = (inp[0] if isinstance(inp, tuple) else inp) \
                       if capture_input else \
                       (out[0]  if isinstance(out, tuple) else out)
                # Store as-is (may be inference-mode tensor).
                # Caller MUST call .clone() OUTSIDE inference_mode to get a
                # regular autograd-compatible tensor.
                self._captured_features[name] = feat
        return hook_fn

    def _register_all_hooks(self) -> bool:
        """
        Register feature extraction hooks on all 3 local experts.

        Hook targets:
            DRCT:   self.drct.conv_after_body  → [B, 180, H_pad, W_pad]
            GRL:    self.grl.conv_after_body   → [B, 180, H_pad, W_pad]
            NAFNet: self.nafnet.nafnet.decoders[-1]  → capture INPUT
                    This is the feature BEFORE the final decoder block,
                    giving [B, 64, H_up, W_up] at full upscaled resolution.
                    Alternative: hook ending's INPUT for [B, 64, H_up, W_up].

        NAFNet hook strategy:
            - The NAFNet UNet has: intro → encoders → middle → decoders → ending
            - `ending` is a 1×1 conv: [B, 64, H, W] → [B, 3, H, W]
            - We hook `ending` INPUT to capture [B, 64, H, W] features
        """
        self._remove_all_hooks()
        registered = False

        # DRCT: OUTPUT of conv_after_body
        if self._experts_loaded['drct'] and self.drct is not None:
            if hasattr(self.drct, 'conv_after_body'):
                handle = self.drct.conv_after_body.register_forward_hook(
                    self._create_feature_hook('drct', capture_input=False)
                )
                self._hook_handles.append(handle)
                registered = True
            else:
                print("  ⚠ DRCT: no attribute 'conv_after_body' — hook skipped")

        # GRL: OUTPUT of conv_after_body
        if self._experts_loaded['grl'] and self.grl is not None:
            if hasattr(self.grl, 'conv_after_body'):
                handle = self.grl.conv_after_body.register_forward_hook(
                    self._create_feature_hook('grl', capture_input=False)
                )
                self._hook_handles.append(handle)
                registered = True
            else:
                print("  ⚠ GRL: no attribute 'conv_after_body' — hook skipped")

        # NAFNet: INPUT of ending (captures [B, 64, H, W] before final 1x1 conv)
        # CRITICAL: self.nafnet is a NAFNetSR wrapper — we must access the inner
        # NAFNet UNet to find the `.ending` layer reliably.
        if self._experts_loaded['nafnet'] and self.nafnet is not None:
            # Safely access the inner NAFNet model through the wrapper
            naf_core = getattr(self.nafnet, 'nafnet', self.nafnet)

            if hasattr(naf_core, 'ending'):
                handle = naf_core.ending.register_forward_hook(
                    self._create_feature_hook('nafnet', capture_input=True)
                )
                self._hook_handles.append(handle)
                registered = True
            else:
                print("  ⚠ NAFNet: no attribute 'ending' — hook skipped")

        return registered

    def _remove_all_hooks(self):
        for h in self._hook_handles:
            try: h.remove()
            except: pass
        self._hook_handles = []

    # =========================================================================
    # MAIN TRAINING-TIME FORWARD — hooks + autograd-safe features
    # =========================================================================

    def forward_all_with_hooks(
        self,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with hook-based feature extraction.
        The returned features dict is AUTOGRAD-COMPATIBLE for training the
        fusion network.

        IMPORTANT — inference_mode vs no_grad:
            Individual forward_X methods are @torch.inference_mode().
            Tensors created inside inference_mode CANNOT participate in
            autograd even after the context exits.
            FIX: after the with torch.no_grad() block, call .clone() on each
            captured tensor OUTSIDE inference_mode.  clone() outside
            inference_mode creates a normal tensor — backward passes work.

        Returns:
            outputs:  {'drct': [B,3,Hs,Ws], 'grl': ..., 'nafnet': ...}
            features: {'drct': [B,180,H,W], 'grl': [B,180,H,W],
                       'nafnet': [B,64,H,W]}
            where H, W = original LR spatial dims.
            
        Note: MambaIR features are NOT included — they come from cached
        .pt files loaded by CachedSRDataset.
        """
        _, _, h, w = x.shape

        # Register hooks once (idempotent — skip if already registered)
        if not self._hook_handles:
            self._register_all_hooks()

        self._captured_features = {}
        self._capture_features  = True

        try:
            with torch.no_grad():
                outputs = self.forward_all(x, return_dict=True)
        finally:
            self._capture_features = False

        # ── CRITICAL: clone() OUTSIDE inference_mode ─────────────────────────
        features = {}
        for name, feat in self._captured_features.items():
            cloned = feat.clone()
            # Crop transformer-based features to original LR resolution
            if name in ('drct', 'grl'):
                cloned = cloned[:, :, :h, :w].contiguous()
            elif name == 'nafnet':
                # NAFNet features are at upscaled resolution (H*4, W*4)
                # Resize to LR resolution for consistency with other experts
                cloned = F.interpolate(
                    cloned, size=(h, w), mode='bilinear', align_corners=False
                ).contiguous()
            features[name] = cloned

        return outputs, features

    # Alias for backward compat
    def forward_all_with_features(self, x: torch.Tensor):
        """Alias → forward_all_with_hooks (preferred method for training)."""
        return self.forward_all_with_hooks(x)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_loaded_experts(self) -> List[str]:
        return [n for n, loaded in self._experts_loaded.items() if loaded]

    def __repr__(self) -> str:
        return (f"ExpertEnsemble(experts={self.get_loaded_experts()}, "
                f"upscale={self.upscale}, device={self.device})")


# ============================================================================
# Verification Test
# ============================================================================

def test_expert_ensemble(checkpoint_dir: Optional[str] = None):
    """
    Expert ensemble verification.
    Checks: loading, param counts, forward shapes, hook channels,
            feature autograd compatibility, output value ranges.
    """
    import traceback

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*65}")
    print(f"  Expert Ensemble Verification — device: {device}")
    print(f"  Local: DRCT + GRL + NAFNet")
    print(f"  Remote: MambaIR (Decoupled Compute)")
    print(f"{'='*65}")

    ensemble = ExpertEnsemble(device=device, checkpoint_dir=checkpoint_dir)
    results  = ensemble.load_all_experts()

    loaded = ensemble.get_loaded_experts()
    print(f"\nLoaded: {loaded}")

    if not loaded:
        print("✗ No experts loaded — check checkpoint paths")
        return False

    # ── 1. Param count check ────────────────────────────────────────────────
    expected_params = {'drct': 27.6, 'grl': 20.2, 'nafnet': 67.0}
    model_map = {'drct': ensemble.drct, 'grl': ensemble.grl,
                 'nafnet': ensemble.nafnet}

    print(f"\n{'Expert':<8} {'Params':>8}  {'Expected':>8}  {'OK'}")
    print("-" * 40)
    all_params_ok = True
    for name in loaded:
        m = model_map[name]
        p = sum(x.numel() for x in m.parameters()) / 1e6
        exp = expected_params[name]
        ok  = abs(p - exp) < 5.0   # 5M tolerance
        all_params_ok &= ok
        print(f"  {name:<6} {p:>8.2f}M  {exp:>6.1f}M  {'✓' if ok else '✗'}")

    # ── 2. Forward + Hook test ────────────────────────────────────────────────
    x = torch.randn(1, 3, 64, 64).to(device)

    expected_hook_ch = {'drct': 180, 'grl': 180, 'nafnet': 64}

    print(f"\n{'Expert':<8} {'SR shape':<20} {'Hook shape':<22} {'OK'}")
    print("-" * 60)

    try:
        outputs, features = ensemble.forward_all_with_hooks(x)
    except Exception as e:
        print(f"✗ forward_all_with_hooks failed: {e}")
        traceback.print_exc()
        return False

    all_fwd_ok = True
    for name in loaded:
        sr   = outputs[name]
        feat = features[name]
        sr_ok   = sr.shape   == (1, 3, 256, 256)
        hook_ok = feat.shape[1] == expected_hook_ch[name]
        size_ok = feat.shape[2] == 64 and feat.shape[3] == 64  # LR resolution
        ok = sr_ok and hook_ok and size_ok
        all_fwd_ok &= ok
        print(f"  {name:<6} {str(tuple(sr.shape)):<20} {str(tuple(feat.shape)):<22} "
              f"{'✓' if ok else '✗'}")
        if not sr_ok:
            print(f"         ✗ SR shape wrong: {sr.shape} != (1,3,256,256)")
        if not hook_ok:
            print(f"         ✗ Hook ch wrong: {feat.shape[1]} != {expected_hook_ch[name]}")
        if not size_ok:
            print(f"         ✗ Feature spatial wrong: {feat.shape[2:]} != (64, 64)")

    # ── 3. Autograd compatibility check ─────────────────────────────────────
    print("\n  Autograd compatibility (features → trainable layer → backward):")
    autograd_ok = True
    for name, feat in features.items():
        try:
            in_ch = feat.shape[1]
            proj  = nn.Conv2d(in_ch, 64, 1).to(device)
            out   = proj(feat)
            loss  = out.mean()
            loss.backward()
            print(f"    {name:<6} ✓  feat→Conv2d→backward OK")
        except Exception as e:
            print(f"    {name:<6} ✗  {e}")
            autograd_ok = False

    # ── 4. Output range check ────────────────────────────────────────────────
    print("\n  Output value ranges (expect ~[0, 1] for loaded weights):")
    for name, sr in outputs.items():
        mn, mx = sr.min().item(), sr.max().item()
        in_range = 0.0 <= mn and mx <= 1.0
        print(f"    {name:<6} [{mn:.4f}, {mx:.4f}]  {'✓' if in_range else '⚠ check weights'}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    all_ok = all_params_ok and all_fwd_ok and autograd_ok
    print(f"\n{'='*65}")
    print(f"  Expert Ensemble: {'ALL PASS ✓' if all_ok else 'ISSUES FOUND ✗'}")
    print(f"{'='*65}\n")
    return all_ok


if __name__ == '__main__':
    test_expert_ensemble()
