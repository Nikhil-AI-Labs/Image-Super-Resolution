"""
Expert Loader Module — Phase 2
================================
4-Expert ensemble: HAT-L, DRCT-L, GRL-B, EDSR-L.

Expert roles (hook-based feature extraction):
  HAT-L   [B, 180, H, W]  conv_after_body — high-freq transformer (Samsung NTIRE winner)
  DRCT-L  [B, 180, H, W]  conv_after_body — dense residual connected transformer
  GRL-B   [B, 180, H, W]  conv_after_body — global/regional/local representation
  EDSR-L  [B, 256, H, W]  conv_after_body — enhanced deep SR (pure conv, no attention)

All experts are FROZEN — only the downstream fusion network trains.

CRITICAL: forward_all_with_hooks() uses feat.clone() OUTSIDE @inference_mode
context to convert captured tensors into autograd-compatible tensors for
backward passes through the fusion network.

Usage:
    ensemble = ExpertEnsemble(device='cuda')
    ensemble.load_all_experts()

    outputs, features = ensemble.forward_all_with_hooks(lr_image)
    # outputs  = {'hat': [B,3,H*4,W*4], 'drct': ..., 'grl': ..., 'edsr': ...}
    # features = {'hat': [B,180,H,W], 'drct': [B,180,H,W],
    #             'grl': [B,180,H,W],  'edsr': [B,256,H,W]}
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
from concurrent.futures import ThreadPoolExecutor
import warnings

# ── Phase 2: updated aliases ────────────────────────────────────────────────
EXPERT_ALIASES = {
    'mambair': 'dat',   # backward compat
    'mamba':   'dat',   # backward compat
    'dat':     'drct',  # dat → drct in new ensemble
    'nafnet':  'edsr',  # nafnet → edsr in new ensemble
}

def normalize_expert_name(name: str) -> str:
    name_lower = name.lower()
    return EXPERT_ALIASES.get(name_lower, name_lower)


# ============================================================================
# Utility Functions  (unchanged from Phase 1)
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
# ExpertEnsemble  — Phase 2
# ============================================================================

class ExpertEnsemble(nn.Module):
    """
    4-Expert Ensemble with hook-based feature extraction.

    Hook feature shapes (at LR resolution, ready for feat_proj):
        'hat':  [B, 180, H, W]
        'drct': [B, 180, H, W]
        'grl':  [B, 180, H, W]
        'edsr': [B, 256, H, W]

    Downstream feat_proj (in fusion network) must use:
        nn.Conv2d(180, fusion_dim, 1)  for hat / drct / grl
        nn.Conv2d(256, fusion_dim, 1)  for edsr
    """

    # ── window sizes per expert ──────────────────────────────────────────────
    DRCT_WINDOW = 16   # DRCT-L default window_size
    GRL_WINDOW  = 8    # GRL-B default window_size

    def __init__(
        self,
        upscale:        int                        = 4,
        window_size:    int                        = 16,   # HAT window size
        device:         Union[str, torch.device]   = 'cuda',
        checkpoint_dir: Optional[str]              = None,
    ):
        super().__init__()
        self.upscale     = upscale
        self.window_size = window_size   # used by HAT
        self.device      = torch.device(device)   # single GPU for all experts

        # ── checkpoint directory ─────────────────────────────────────────────
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = _find_pretrained_dir()

        # ── expert models ────────────────────────────────────────────────────
        self.hat  = None   # HAT-L,   40.8M,  window=16
        self.drct = None   # DRCT-L,  27.6M,  window=16
        self.grl  = None   # GRL-B,   20.2M,  window=8
        self.edsr = None   # EDSR-L,  43.1M,  conv-only

        self._experts_loaded = {
            'hat':  False,
            'drct': False,
            'grl':  False,
            'edsr': False,
        }

        # ── hook infrastructure ──────────────────────────────────────────────
        self._captured_features = {}
        self._hook_handles      = []
        self._capture_features  = False

        print(f"  [Single-GPU] All 4 experts → {self.device}")

    # ── basicsr mock (HAT + DRCT both need it) ───────────────────────────────
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

    def load_hat(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ) -> bool:
        """Load HAT-L (40.8M).  window_size=16,  hook→conv_after_body [B,180,H,W]"""
        try:
            self._setup_basicsr_mocks()
            from src.models.hat import create_hat_model

            self.hat = create_hat_model(
                embed_dim   = 180,
                depths      = [6] * 12,
                num_heads   = [6] * 12,
                window_size = self.window_size,
                upscale     = self.upscale,
                img_range   = 1.0,
            )

            if checkpoint_path is None:
                checkpoint_path = str(
                    self.checkpoint_dir / 'hat' / 'HAT-L_SRx4_ImageNet-pretrain.pth'
                )
            if os.path.exists(checkpoint_path):
                self.hat, info = load_checkpoint_flexible(checkpoint_path, self.hat)
                print(f"  ✓ HAT-L   loaded: {info['loaded']}/{info['total']} keys")
            else:
                print(f"  ⚠ HAT-L   checkpoint not found: {checkpoint_path}")

            if freeze:
                for p in self.hat.parameters(): p.requires_grad = False
                self.hat.eval()
            self.hat = self.hat.to(self.device)
            self._experts_loaded['hat'] = True
            return True

        except Exception as e:
            print(f"  ✗ HAT-L   failed: {e}")
            return False

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
            # drct/__init__.py sets up basicsr mocks at import time
            self._setup_basicsr_mocks()   # idempotent — safe to call again
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

    def load_edsr(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ) -> bool:
        """
        Load EDSR-L (43.1M).  NO window padding needed.  hook→conv_after_body [B,256,H,W]

        Checkpoint: pretrained/edsr/EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth  (172.4 MB)
        138/139 keys load (mean buffer initialized to DIV2K RGB mean in code).
        CRITICAL: img_range=255.0 is MANDATORY — pretrained weights expect it.
        """
        try:
            from src.models.edsr import create_edsr_model, EDSR_AVAILABLE

            if not EDSR_AVAILABLE:
                print("  ✗ EDSR-L  architecture not available")
                return False

            self.edsr = create_edsr_model(
                num_feat  = 256,
                num_block = 32,
                upscale   = self.upscale,
                res_scale = 0.1,
                img_range = 255.0,   # ← MANDATORY, pretrained weights require this
            )

            if checkpoint_path is None:
                # Try both possible filenames
                for fname in [
                    'EDSR_Lx4_f256b32_DIV2K.pth',
                    'EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth',
                ]:
                    p = self.checkpoint_dir / 'edsr' / fname
                    if p.exists():
                        checkpoint_path = str(p)
                        break

            if checkpoint_path and os.path.exists(checkpoint_path):
                self.edsr, info = load_checkpoint_flexible(checkpoint_path, self.edsr)
                print(f"  ✓ EDSR-L  loaded: {info['loaded']}/{info['total']} keys"
                      f"  (1 mean buffer initialized in code — expected)")
            else:
                print(f"  ⚠ EDSR-L  checkpoint not found (searched pretrained/edsr/)")

            if freeze:
                for p in self.edsr.parameters(): p.requires_grad = False
                self.edsr.eval()
            self.edsr = self.edsr.to(self.device)
            self._experts_loaded['edsr'] = True
            return True

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ EDSR-L  failed: {e}")
            return False

    def load_all_experts(
        self,
        checkpoint_paths: Optional[Dict[str, str]] = None,
        freeze: bool = True,
    ) -> Dict[str, bool]:
        """Load all 4 experts.  Returns dict of name→success."""
        if checkpoint_paths is None:
            checkpoint_paths = {}

        print("\n" + "=" * 60)
        print("  Loading Expert Models  (Phase 2 — 4-Expert Ensemble)")
        print("=" * 60)

        results = {
            'hat':  self.load_hat( checkpoint_paths.get('hat'),  freeze),
            'drct': self.load_drct(checkpoint_paths.get('drct'), freeze),
            'grl':  self.load_grl( checkpoint_paths.get('grl'),  freeze),
            'edsr': self.load_edsr(checkpoint_paths.get('edsr'), freeze),
        }

        print("=" * 60)
        loaded = sum(results.values())
        print(f"  Loaded {loaded}/4 experts")
        print("=" * 60 + "\n")
        return results

    # =========================================================================
    # FORWARD METHODS  — individual experts
    # =========================================================================

    @torch.inference_mode()
    def forward_hat(self, x: torch.Tensor) -> torch.Tensor:
        """HAT-L inference.  Pads to window_size=16, crops output."""
        if self.hat is None: raise RuntimeError("HAT not loaded.")
        _, _, h, w = x.shape
        xp, _, _ = pad_to_window_size(x, self.window_size, self.upscale)
        sr = self.hat(xp)
        return crop_to_size(sr, h * self.upscale, w * self.upscale).clamp(0, 1)

    @torch.inference_mode()
    def forward_drct(self, x: torch.Tensor) -> torch.Tensor:
        """
        DRCT-L inference.  Pads to window_size=16, crops output.

        DRCT has internal check_image_size() but external padding ensures
        it is a no-op, keeping behaviour consistent with HAT.
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
    def forward_edsr(self, x: torch.Tensor) -> torch.Tensor:
        """
        EDSR-L inference.  Fully convolutional — no padding needed.

        img_range=255.0 is handled internally by EDSR:
            (x - mean) * 255  →  body  →  out / 255 + mean
        Input [0,1] → Output [0,1].
        """
        if self.edsr is None: raise RuntimeError("EDSR not loaded.")
        return self.edsr(x).clamp(0, 1)

    # =========================================================================
    # PARALLEL FORWARD (no hooks)
    # =========================================================================

    def forward_all(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run all 4 experts sequentially on a single GPU.
        Uses torch.no_grad() (NOT inference_mode) so hook-captured tensors remain
        autograd-compatible when cloned outside the context.
        """
        outputs = {}
        with torch.no_grad():
            if self._experts_loaded['hat']:  outputs['hat']  = self.forward_hat(x)
            if self._experts_loaded['drct']: outputs['drct'] = self.forward_drct(x)
            if self._experts_loaded['grl']:  outputs['grl']  = self.forward_grl(x)
            if self._experts_loaded['edsr']: outputs['edsr'] = self.forward_edsr(x)

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
        Register conv_after_body OUTPUT hooks on all 4 experts.

        Hook targets:
            HAT:  self.hat.conv_after_body   → [B, 180, H_pad, W_pad]
            DRCT: self.drct.conv_after_body  → [B, 180, H_pad, W_pad]
            GRL:  self.grl.conv_after_body   → [B, 180, H_pad, W_pad]
            EDSR: self.edsr.conv_after_body  → [B, 256, H, W]

        All four use capture_input=False (OUTPUT mode).
        After forward_all_with_hooks(), caller crops features to (h, w).
        """
        self._remove_all_hooks()
        registered = False

        expert_map = [
            ('hat',  self.hat,  'conv_after_body', False),
            ('drct', self.drct, 'conv_after_body', False),
            ('grl',  self.grl,  'conv_after_body', False),
            ('edsr', self.edsr, 'conv_after_body', False),
        ]

        for name, model, attr, capture_input in expert_map:
            if not self._experts_loaded[name] or model is None:
                continue
            if not hasattr(model, attr):
                print(f"  ⚠ {name}: no attribute '{attr}' — hook skipped")
                continue
            try:
                handle = getattr(model, attr).register_forward_hook(
                    self._create_feature_hook(name, capture_input)
                )
                self._hook_handles.append(handle)
                registered = True
            except Exception as e:
                print(f"  ⚠ {name}: hook registration failed: {e}")

        return registered

    def _remove_all_hooks(self):
        for h in self._hook_handles:
            try: h.remove()
            except: pass
        self._hook_handles = []

    # =========================================================================
    # MAIN TRAINING-TIME FORWARD  — hooks + autograd-safe features
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
            outputs:  {'hat': [B,3,Hs,Ws], 'drct': ..., 'grl': ..., 'edsr': ...}
            features: {'hat': [B,180,H,W], 'drct': [B,180,H,W],
                       'grl': [B,180,H,W],  'edsr': [B,256,H,W]}
            where H, W = original LR spatial dims.
        """
        _, _, h, w = x.shape

        # Register hooks once (idempotent — skip if already registered)
        if not self._hook_handles:
            self._register_all_hooks()

        self._captured_features = {}
        self._capture_features  = True

        try:
            with torch.no_grad():
                # forward_hat/drct/grl/edsr are @inference_mode internally.
                # Hooks fire inside those contexts and store inference-mode tensors.
                outputs = self.forward_all(x, return_dict=True)
        finally:
            self._capture_features = False

        # ── CRITICAL: clone() OUTSIDE inference_mode ─────────────────────────
        # We are now outside both no_grad and inference_mode contexts.
        # feat is an inference-mode tensor stored in _captured_features.
        # feat.clone() here creates a regular tensor that CAN be an input
        # to trainable layers (feat_proj) and CAN participate in backward.
        features = {}
        for name, feat in self._captured_features.items():
            # Clone to exit inference-mode, then crop to original LR resolution.
            # Features are at padded resolution (h_pad, w_pad); crop to (h, w).
            features[name] = feat.clone()[:, :, :h, :w].contiguous()

        return outputs, features

    # Alias for backward compat with code that calls forward_all_with_features
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
# Phase 2 Verification Test
# ============================================================================

def test_phase2(checkpoint_dir: Optional[str] = None):
    """
    Complete Phase 2 verification.
    Checks: loading, param counts, forward shapes, hook channels,
            feature autograd compatibility, output value ranges.
    """
    import traceback

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*65}")
    print(f"  Phase 2 Verification  —  device: {device}")
    print(f"{'='*65}")

    ensemble = ExpertEnsemble(device=device, checkpoint_dir=checkpoint_dir)
    results  = ensemble.load_all_experts()

    loaded = ensemble.get_loaded_experts()
    print(f"\nLoaded: {loaded}")

    if not loaded:
        print("✗ No experts loaded — check checkpoint paths")
        return False

    # ── 1. Param count check ────────────────────────────────────────────────
    expected_params = {'hat': 40.8, 'drct': 27.6, 'grl': 20.2, 'edsr': 43.1}
    model_map = {'hat': ensemble.hat, 'drct': ensemble.drct,
                 'grl': ensemble.grl, 'edsr': ensemble.edsr}

    print(f"\n{'Expert':<8} {'Params':>8}  {'Expected':>8}  {'OK'}")
    print("-" * 40)
    all_params_ok = True
    for name in loaded:
        m = model_map[name]
        p = sum(x.numel() for x in m.parameters()) / 1e6
        exp = expected_params[name]
        ok  = abs(p - exp) < 2.0   # 2M tolerance
        all_params_ok &= ok
        print(f"  {name:<6} {p:>8.2f}M  {exp:>6.1f}M  {'✓' if ok else '✗'}")

    # ── 2. Forward + Hook test ────────────────────────────────────────────────
    x = torch.randn(1, 3, 64, 64).to(device)

    expected_hook_ch = {'hat': 180, 'drct': 180, 'grl': 180, 'edsr': 256}

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
        ok = sr_ok and hook_ok
        all_fwd_ok &= ok
        print(f"  {name:<6} {str(tuple(sr.shape)):<20} {str(tuple(feat.shape)):<22} "
              f"{'✓' if ok else '✗'}")
        if not sr_ok:
            print(f"         ✗ SR shape wrong: {sr.shape} != (1,3,256,256)")
        if not hook_ok:
            print(f"         ✗ Hook ch wrong: {feat.shape[1]} != {expected_hook_ch[name]}")

    # ── 3. Autograd compatibility check ─────────────────────────────────────
    print("\n  Autograd compatibility (features → trainable layer → backward):")
    autograd_ok = True
    for name, feat in features.items():
        try:
            # Simulate feat_proj: Conv2d with trainable weights
            in_ch = feat.shape[1]
            proj  = nn.Conv2d(in_ch, 64, 1).to(device)
            out   = proj(feat)          # must NOT raise "inference-mode tensor" error
            loss  = out.mean()
            loss.backward()             # must NOT raise
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
    print(f"  Phase 2: {'ALL PASS ✓' if all_ok else 'ISSUES FOUND ✗'}")
    print(f"{'='*65}\n")
    return all_ok


if __name__ == '__main__':
    test_phase2()
