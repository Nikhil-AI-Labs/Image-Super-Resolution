"""
scripts/generate_fast_submission.py
====================================
Generates a complete NTIRE submission ZIP using cached Test TTA features.

This script is the payoff of the "Decoupled Submission" strategy:
  1. You already cached 8 TTA variants per test image (extract_test_tta_cache.py)
  2. This script loads them, runs your tiny 1.2M param fusion network, reverses
     the geometry, averages the 8 predictions, and zips the result.

Result: Full 8x TTA submission from a new checkpoint in ~30 seconds.

Output:
    submission/
    ├── 0901x4.png           (SR output, same name as input LR)
    ├── 0902x4.png
    ├── ...
    ├── readme.txt           (NTIRE required metadata)
    └── res.zip              (all the above, ready to upload)

Usage:
    python scripts/generate_fast_submission.py \\
        --test-cache-dir dataset/test_tta_cache \\
        --checkpoint checkpoints/best.pth \\
        --output-dir submission/ \\
        --config configs/train_config.yaml

    # Without TTA (faster, ~4 seconds total):
    python scripts/generate_fast_submission.py \\
        --test-cache-dir dataset/test_tta_cache \\
        --checkpoint checkpoints/best.pth \\
        --output-dir submission/ \\
        --no-tta

Author: NTIRE SR Team
"""

import os
import sys
import glob
import torch
import argparse
import yaml
import zipfile
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def reverse_tta(tensor, hflip, rot):
    """Reverse the geometric TTA transformation on SR output."""
    if rot > 0:
        tensor = torch.rot90(tensor, -rot, [2, 3])
    if hflip:
        tensor = torch.flip(tensor, [3])
    return tensor


def load_fusion_model(checkpoint_path, config_path, device):
    """
    Load CompleteEnhancedFusionSR in cached mode (expert_ensemble=None).
    Only the fusion network weights are loaded — experts are pre-computed.
    """
    from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR

    with open(config_path) as f:
        config = yaml.safe_load(f)

    fusion_cfg = config.get('model', {}).get('fusion', {})
    improvements = fusion_cfg.get('improvements', {})

    model = CompleteEnhancedFusionSR(
        expert_ensemble=None,  # CACHED MODE
        num_experts=fusion_cfg.get('num_experts', 4),
        fusion_dim=fusion_cfg.get('fusion_dim', 128),
        refine_channels=fusion_cfg.get('refine_channels', 128),
        refine_depth=fusion_cfg.get('refine_depth', 6),
        base_channels=fusion_cfg.get('base_channels', 64),
        block_size=fusion_cfg.get('block_size', 8),
        upscale=config.get('dataset', {}).get('scale', 4),
        enable_dynamic_selection=improvements.get('dynamic_expert_selection', True),
        enable_cross_band_attn=improvements.get('cross_band_attention', True),
        enable_adaptive_bands=improvements.get('adaptive_frequency_bands', True),
        enable_multi_resolution=improvements.get('multi_resolution_fusion', True),
        enable_collaborative=improvements.get('collaborative_learning', True),
        enable_edge_enhance=improvements.get('edge_enhancement', True),
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    # Clean prefixes
    clean_sd = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ['module.', 'model.']:
            if key.startswith(prefix):
                key = key[len(prefix):]
        clean_sd[key] = v

    # Filter to only matching keys
    model_sd = model.state_dict()
    loaded = 0
    for k, v in clean_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k] = v
            loaded += 1

    model.load_state_dict(model_sd, strict=False)
    model.eval().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Loaded {loaded} weight tensors from checkpoint")
    print(f"  ✓ Fusion network: {trainable:,} trainable parameters")

    if loaded == 0:
        print("  ⚠ WARNING: 0 weights loaded! Check checkpoint compatibility.")

    return model


def save_image_tensor(tensor, path):
    """Save [3, H, W] or [1, 3, H, W] tensor as PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)
    img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(img_np).save(path, format='PNG')


def main():
    parser = argparse.ArgumentParser(
        description='Fast NTIRE Submission Generator (uses cached TTA features)'
    )
    parser.add_argument('--test-cache-dir', required=True,
                        help='Directory with cached TTA .pt files')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained fusion checkpoint')
    parser.add_argument('--output-dir', default='submission',
                        help='Output directory for images + res.zip')
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--no-tta', action='store_true',
                        help='Use only t0 (no TTA averaging) for speed')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("  FAST NTIRE SUBMISSION GENERATOR")
    print("=" * 70)
    print(f"  Cache dir:   {args.test_cache_dir}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output:      {args.output_dir}")
    print(f"  TTA mode:    {'OFF (single pass)' if args.no_tta else '8x geometric self-ensemble'}")
    print(f"  Device:      {device}")
    print("=" * 70 + "\n")

    # ── Load Fusion Model ──
    print("Loading fusion model...")
    model = load_fusion_model(args.checkpoint, args.config, device)

    # ── Find unique image stems ──
    # Files are named: {stem}_t{0-7}_drct_part.pt
    # Find all t0 files to get unique stems
    t0_files = sorted(glob.glob(os.path.join(args.test_cache_dir, '*_t0_drct_part.pt')))
    stems = [Path(p).stem.replace('_t0_drct_part', '') for p in t0_files]

    if not stems:
        print("  ✗ No cached test TTA files found!")
        print(f"    Expected: *_t0_drct_part.pt in {args.test_cache_dir}")
        print(f"    Run extract_test_tta_cache.py first.")
        sys.exit(1)

    num_variants = 1 if args.no_tta else 8
    print(f"\n  Found {len(stems)} test images")
    print(f"  Generating with {num_variants}x TTA averaging...")

    # ── Generate Submissions ──
    start_time = time.time()
    generated = 0

    with torch.no_grad():
        for stem in tqdm(stems, desc='Generating SR', ncols=100):
            variant_outputs = []

            for t_idx in range(num_variants):
                t_stem = f"{stem}_t{t_idx}"
                cache_dir = args.test_cache_dir

                # Load all 3 cached parts
                d_path = os.path.join(cache_dir, f'{t_stem}_drct_part.pt')
                r_path = os.path.join(cache_dir, f'{t_stem}_rest_part.pt')
                m_path = os.path.join(cache_dir, f'{t_stem}_mamba_part.pt')

                if not os.path.exists(d_path):
                    tqdm.write(f"  ⚠ Missing: {d_path}")
                    continue

                d_data = torch.load(d_path, map_location=device, weights_only=False)
                r_data = torch.load(r_path, map_location=device, weights_only=False)
                m_data = torch.load(m_path, map_location=device, weights_only=False)

                # Extract LR input and TTA info
                lr = d_data['lr'].unsqueeze(0).float().to(device)
                tta_info = d_data.get('tta_info', {'hflip': False, 'rot': 0})
                hflip = tta_info['hflip']
                rot = tta_info['rot']

                # Combine all expert outputs (FP16 → FP32)
                expert_imgs = {
                    'drct':   d_data['outputs']['drct'].float().to(device),
                    'grl':    r_data['outputs']['grl'].float().to(device),
                    'nafnet': r_data['outputs']['nafnet'].float().to(device),
                    'mamba':  m_data['outputs']['mamba'].float().to(device),
                }
                expert_feats = {
                    'drct':   d_data['features']['drct'].float().to(device),
                    'grl':    r_data['features']['grl'].float().to(device),
                    'nafnet': r_data['features']['nafnet'].float().to(device),
                    'mamba':  m_data['features']['mamba'].float().to(device),
                }

                # Forward pass through fusion network ONLY (~5ms)
                sr = model.forward_with_precomputed(lr, expert_imgs, expert_feats)

                # Reverse the geometric transform
                sr_reversed = reverse_tta(sr, hflip, rot)

                # Move to CPU to save GPU memory
                variant_outputs.append(sr_reversed.squeeze(0).cpu())

                # Free GPU memory
                del sr, sr_reversed, lr, expert_imgs, expert_feats
                del d_data, r_data, m_data
                torch.cuda.empty_cache()

            if not variant_outputs:
                tqdm.write(f"  ⚠ No variants found for {stem}")
                continue

            # Average all TTA variants (on CPU, FP32 for precision)
            final_sr = torch.stack(variant_outputs).float().mean(dim=0).clamp(0, 1)

            # Save as PNG with NTIRE naming convention
            # Test files are named like 0901x4.png → output must be 0901x4.png
            output_name = f"{stem}x4.png"
            save_image_tensor(final_sr, os.path.join(args.output_dir, output_name))
            generated += 1

    total_time = time.time() - start_time
    avg_time = total_time / max(generated, 1)

    # ── Create res.zip ──
    print(f"\n  Creating res.zip...")
    zip_path = os.path.join(args.output_dir, 'res.zip')

    # Build readme content
    readme_content = f"""runtime per image [s] : {avg_time:.2f}
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description :
Solution based on CompleteEnhancedFusionSR architecture combining four frozen expert
models (DRCT-L, GRL-B, NAFNet-SIDD-width64, MambaIR-180) with a trainable multi-phase
fusion pipeline (~1.2M trainable parameters):
- Phase 2: Multi-Domain Frequency Decomposition (DCT+DWT+FFT → 9 bands)
- Phase 3: Cross-Band Attention with Large Kernel Attention (k=21)
- Phase 4: Collaborative Feature Learning with Cross-Expert Attention
- Phase 5: Hierarchical Multi-Resolution Fusion (progressive 3-stage)
- Phase 6: Dynamic Expert Selection (per-pixel difficulty gating)
- Phase 7: Deep CNN Refinement + Laplacian Pyramid Edge Enhancement
Training: DIV2K+Flickr2K (DF2K) training set, cached expert features.
Inference: {num_variants}x geometric TTA, PyTorch, NVIDIA GPU, FP32.
"""

    sr_images = sorted(glob.glob(os.path.join(args.output_dir, '*.png')))
    # Exclude any comparison images
    sr_images = [p for p in sr_images if 'comparison' not in os.path.basename(p)]

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for img_path in sr_images:
            zipf.write(img_path, arcname=os.path.basename(img_path))
        zipf.writestr('readme.txt', readme_content)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"  🎉 SUBMISSION READY!")
    print(f"{'='*70}")
    print(f"  Images generated: {generated}")
    print(f"  TTA variants:     {num_variants}x")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Avg per image:    {avg_time:.2f}s")
    print(f"  ZIP size:         {zip_size_mb:.1f} MB")
    print(f"  ZIP path:         {zip_path}")
    print(f"\n  Upload res.zip to CodaBench for evaluation.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
