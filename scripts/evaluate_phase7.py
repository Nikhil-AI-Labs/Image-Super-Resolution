"""
Phase 7: TSD-SR Model Evaluation
=================================
Compare PSNR baseline vs TSD-SR refinement for NTIRE 2025 submission.

Models Evaluated:
1. Baseline: PSNR fusion model only (Phase 6)
2. Teacher TSD: Multi-step diffusion refinement (slow, highest quality)
3. Student TSD: One-step diffusion refinement (fast, good quality)

Track Recommendations:
- Track A (Restoration): Use baseline for highest PSNR
- Track B (Perceptual): Use student TSD for best perceptual/speed balance

Usage:
    python scripts/evaluate_phase7.py \
        --psnr_checkpoint checkpoints/best.pth \
        --teacher_model pretrained/teacher/teacher.safetensors \
        --student_model pretrained/tsdsr/transformer.safetensors \
        --vae_model pretrained/tsdsr/vae.safetensors

Author: NTIRE SR Team
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import json
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torchvision.utils as vutils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Phase 7: TSD-SR Evaluation')
    
    # Models
    parser.add_argument('--psnr_checkpoint', type=str, default=None,
                       help='PSNR model checkpoint from Phase 6')
    parser.add_argument('--teacher_model', type=str,
                       default='pretrained/teacher/teacher.safetensors',
                       help='Teacher TSD model (multi-step)')
    parser.add_argument('--student_model', type=str,
                       default='pretrained/tsdsr/transformer.safetensors',
                       help='Student TSD model (one-step)')
    parser.add_argument('--vae_model', type=str,
                       default='pretrained/tsdsr/vae.safetensors',
                       help='VAE model')
    
    # Data
    parser.add_argument('--val_hr_dir', type=str, default='data/DF2K/val_HR')
    parser.add_argument('--val_lr_dir', type=str, default='data/DF2K/val_LR')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Limit number of samples (None = all)')
    
    # Evaluation settings
    parser.add_argument('--models', nargs='+',
                       default=['baseline'],
                       choices=['baseline', 'teacher', 'student', 'all'],
                       help='Which models to evaluate')
    parser.add_argument('--skip_perceptual', action='store_true',
                       help='Skip perceptual metrics (faster)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/phase7_comparison')
    parser.add_argument('--save_images', action='store_true',
                       help='Save output images')
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


class DummyPSNRModel(nn.Module):
    """
    Dummy PSNR model for testing when no checkpoint is provided.
    
    Uses bilinear upsampling as baseline.
    """
    
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale, mode='bicubic', align_corners=False
        ).clamp(0, 1)


def load_psnr_model(
    checkpoint_path: Optional[str],
    device: torch.device
) -> nn.Module:
    """
    Load PSNR model from Phase 6 checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint (None = use dummy)
        device: Device
        
    Returns:
        Loaded model
    """
    print("\n" + "="*70)
    print("LOADING PSNR MODEL")
    print("="*70)
    
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print(f"  No checkpoint provided or not found")
        print(f"  Using dummy bicubic upsampling as baseline")
        model = DummyPSNRModel(scale=4).to(device)
        return model
    
    # Try to load actual model
    try:
        from src.models.fusion_network import FrequencyAwareFusion, MultiFusionSR
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine model type from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model (adjust based on your saved model)
        # Try FrequencyAwareFusion first
        try:
            model = FrequencyAwareFusion(
                num_experts=3,
                use_residual=True,
                use_multiscale=True
            ).to(device)
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Loaded FrequencyAwareFusion from {checkpoint_path}")
        except Exception as e:
            print(f"  Could not load as FrequencyAwareFusion: {e}")
            model = DummyPSNRModel(scale=4).to(device)
            print(f"  Using dummy model instead")
        
        if 'metrics' in checkpoint:
            print(f"  Checkpoint PSNR: {checkpoint['metrics'].get('psnr', 'N/A')}")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"  Error loading model: {e}")
        print(f"  Using dummy bicubic upsampling")
        return DummyPSNRModel(scale=4).to(device)


@torch.no_grad()
def evaluate_combination(
    psnr_model: nn.Module,
    tsd_model: Optional[nn.Module],
    model_name: str,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
    save_images: bool = False,
    output_dir: Optional[Path] = None,
    use_perceptual: bool = True
) -> Dict[str, float]:
    """
    Evaluate a model combination (PSNR baseline + optional TSD refinement).
    
    Pipeline:
    LR → PSNR Model → [Optional: TSD Refinement] → Final SR → Metrics
    
    Args:
        psnr_model: Trained PSNR model
        tsd_model: TSD refinement model (None = baseline only)
        model_name: Name identifier
        val_loader: Validation dataloader
        device: Device
        num_samples: Max samples
        save_images: Save outputs
        output_dir: Output directory
        use_perceptual: Calculate perceptual metrics
        
    Returns:
        Dictionary of metrics
    """
    from src.utils import MetricCalculator
    
    psnr_model.eval()
    if tsd_model is not None:
        tsd_model.eval()
    
    # Initialize metric calculators
    psnr_ssim = MetricCalculator(crop_border=4, test_y_channel=True)
    
    # Perceptual metrics (if available and requested)
    perceptual = None
    if use_perceptual:
        try:
            from src.utils import PerceptualEvaluator
            perceptual = PerceptualEvaluator(device=str(device))
        except Exception as e:
            print(f"  Warning: Perceptual metrics unavailable: {e}")
    
    if save_images and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'='*70}")
    if tsd_model is not None:
        print(f"  Mode: PSNR + TSD Refinement")
        info = tsd_model.get_model_info()
        print(f"  TSD Steps: {info.get('inference_steps', 'N/A')}")
    else:
        print(f"  Mode: PSNR Baseline (no refinement)")
    print(f"{'='*70}\n")
    
    sample_count = 0
    inference_times = []
    
    pbar = tqdm(val_loader, desc=f'Eval {model_name}', ncols=120)
    
    for batch in pbar:
        if num_samples is not None and sample_count >= num_samples:
            break
        
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        
        # Get filename if available
        if 'filename' in batch:
            filename = batch['filename'][0] if isinstance(batch['filename'], list) else batch['filename']
        else:
            filename = f'sample_{sample_count:04d}'
        
        start_time = time.time()
        
        # Step 1: PSNR model
        sr = psnr_model(lr).clamp(0, 1)
        
        # Step 2: TSD refinement (if applicable)
        if tsd_model is not None:
            sr = tsd_model(sr).clamp(0, 1)
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Update PSNR/SSIM
        psnr_ssim.update(sr, hr)
        
        # Update perceptual
        if perceptual is not None:
            perceptual.update(sr, hr)
        
        # Save images
        if save_images and output_dir:
            stem = Path(filename).stem if isinstance(filename, str) else f'img_{sample_count}'
            save_path = output_dir / f"{stem}_{model_name}.png"
            vutils.save_image(sr[0].cpu(), save_path)
        
        sample_count += 1
        
        # Update progress bar
        metrics = psnr_ssim.get_metrics()
        avg_time = np.mean(inference_times) if inference_times else 0
        pbar.set_postfix({
            'PSNR': f"{metrics['psnr']:.2f}",
            'Time': f"{avg_time:.3f}s"
        })
    
    # Get final metrics
    final_metrics = psnr_ssim.get_metrics()
    
    if perceptual is not None:
        perceptual_metrics = perceptual.get_average_metrics()
        final_metrics.update(perceptual_metrics)
    
    # Add timing
    final_metrics['avg_inference_time'] = np.mean(inference_times) if inference_times else 0
    final_metrics['total_samples'] = sample_count
    
    return final_metrics


def print_comparison_table(results: Dict[str, Dict]) -> Dict[str, str]:
    """
    Print comparison table and determine winners.
    
    Args:
        results: Dictionary of model_name -> {'metrics': {...}, 'total_time': ...}
        
    Returns:
        Recommendations dictionary
    """
    print("\n" + "="*70)
    print("FINAL COMPARISON - PHASE 7 RESULTS")
    print("="*70)
    
    # Determine winners
    best_psnr = max(results.items(), key=lambda x: x[1]['metrics'].get('psnr', 0))[0]
    best_perceptual = max(
        results.items(),
        key=lambda x: x[1]['metrics'].get('perceptual_score', 0)
    )[0]
    best_speed = min(
        results.items(),
        key=lambda x: x[1]['metrics'].get('avg_inference_time', float('inf'))
    )[0]
    best_lpips = min(
        results.items(),
        key=lambda x: x[1]['metrics'].get('lpips', float('inf'))
    )[0]
    
    # Print table header
    print(f"\n{'Model':<12} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Perceptual':<12} {'Speed':<10} {'Winner'}")
    print("-"*80)
    
    for name, data in results.items():
        m = data['metrics']
        
        # Determine awards
        markers = []
        if name == best_psnr:
            markers.append("🥇PSNR")
        if name == best_perceptual:
            markers.append("🏆Perceptual")
        if name == best_lpips:
            markers.append("📐LPIPS")
        if name == best_speed:
            markers.append("⚡Speed")
        
        marker_str = " " + " ".join(markers) if markers else ""
        
        print(f"{name:<12} "
              f"{m.get('psnr', 0):<8.2f} "
              f"{m.get('ssim', 0):<8.4f} "
              f"{m.get('lpips', 0):<8.4f} "
              f"{m.get('perceptual_score', 0):<12.4f} "
              f"{m.get('avg_inference_time', 0):<10.3f}s"
              f"{marker_str}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"  Track A (Restoration):  Use '{best_psnr}' (PSNR: {results[best_psnr]['metrics'].get('psnr', 0):.2f} dB)")
    print(f"  Track B (Perceptual):   Use '{best_perceptual}' (Score: {results[best_perceptual]['metrics'].get('perceptual_score', 0):.4f})")
    print(f"  Production (Speed):     Use '{best_speed}' ({results[best_speed]['metrics'].get('avg_inference_time', 0):.3f}s/img)")
    print("="*70 + "\n")
    
    return {
        'track_a': best_psnr,
        'track_b': best_perceptual,
        'production': best_speed
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Device setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("PHASE 7: TSD-SR MODEL EVALUATION")
    print("="*70)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"{'='*70}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # LOAD PSNR MODEL
    # ========================================================================
    psnr_model = load_psnr_model(args.psnr_checkpoint, device)
    
    # ========================================================================
    # LOAD TSD-SR MODELS
    # ========================================================================
    teacher_model, student_model = None, None
    
    models_to_eval = args.models
    if 'all' in models_to_eval:
        models_to_eval = ['baseline', 'teacher', 'student']
    
    if 'teacher' in models_to_eval or 'student' in models_to_eval:
        try:
            from src.models import load_tsdsr_models
            
            teacher_model, student_model = load_tsdsr_models(
                teacher_path=args.teacher_model,
                student_path=args.student_model,
                vae_path=args.vae_model,
                device=str(device)
            )
        except Exception as e:
            print(f"\n  Warning: Could not load TSD-SR models: {e}")
            print(f"  Will only evaluate baseline\n")
            models_to_eval = ['baseline']
    
    # ========================================================================
    # CREATE VALIDATION LOADER
    # ========================================================================
    print("\n" + "="*70)
    print("CREATING VALIDATION LOADER")
    print("="*70)
    
    try:
        from src.data import create_dataloaders
        
        # Use val directories for both train and val (we only need val)
        _, val_loader = create_dataloaders(
            train_hr_dir=args.val_hr_dir,
            train_lr_dir=args.val_lr_dir,
            val_hr_dir=args.val_hr_dir,
            val_lr_dir=args.val_lr_dir,
            batch_size=args.batch_size,
            num_workers=2,
            lr_patch_size=64,  # Match training size
            scale=4,
            repeat_factor=1
        )
        print(f"✓ Validation samples: {len(val_loader.dataset)}\n")
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("Creating dummy dataloader for testing...")
        
        # Create dummy dataloader for testing
        from torch.utils.data import TensorDataset, DataLoader
        
        # Random test data
        dummy_lr = torch.rand(10, 3, 64, 64)
        dummy_hr = torch.rand(10, 3, 256, 256)
        
        class DummyDataset:
            def __init__(self, lr, hr):
                self.lr = lr
                self.hr = hr
            
            def __len__(self):
                return len(self.lr)
            
            def __getitem__(self, idx):
                return {'lr': self.lr[idx], 'hr': self.hr[idx]}
        
        val_loader = DataLoader(DummyDataset(dummy_lr, dummy_hr), batch_size=1)
        print(f"✓ Created dummy validation set: 10 samples\n")
    
    # ========================================================================
    # EVALUATE ALL MODELS
    # ========================================================================
    print("="*70)
    print("EVALUATING MODELS")
    print("="*70)
    
    results = {}
    
    for model_name in models_to_eval:
        # Select TSD model
        if model_name == 'baseline':
            tsd_model = None
        elif model_name == 'teacher':
            tsd_model = teacher_model
        elif model_name == 'student':
            tsd_model = student_model
        else:
            continue
        
        # Output directory for this model
        model_output_dir = output_dir / model_name if args.save_images else None
        
        # Evaluate
        start_time = time.time()
        
        metrics = evaluate_combination(
            psnr_model=psnr_model,
            tsd_model=tsd_model,
            model_name=model_name,
            val_loader=val_loader,
            device=device,
            num_samples=args.num_samples,
            save_images=args.save_images,
            output_dir=model_output_dir,
            use_perceptual=not args.skip_perceptual
        )
        
        total_time = time.time() - start_time
        
        results[model_name] = {
            'metrics': metrics,
            'total_time': total_time
        }
        
        # Print individual results
        print(f"\n{'='*70}")
        print(f"RESULTS: {model_name.upper()}")
        print(f"{'='*70}")
        print(f"  PSNR:              {metrics.get('psnr', 0):.2f} dB")
        print(f"  SSIM:              {metrics.get('ssim', 0):.4f}")
        if not args.skip_perceptual:
            print(f"  LPIPS:             {metrics.get('lpips', 0):.4f}")
            print(f"  DISTS:             {metrics.get('dists', 0):.4f}")
            print(f"  CLIP-IQA:          {metrics.get('clipiqa', 0):.4f}")
            print(f"  MANIQA:            {metrics.get('maniqa', 0):.4f}")
            print(f"  MUSIQ:             {metrics.get('musiq', 0):.2f}")
            print(f"  NIQE:              {metrics.get('niqe', 0):.4f}")
            print(f"{'-'*70}")
            print(f"  PERCEPTUAL SCORE:  {metrics.get('perceptual_score', 0):.4f}")
        print(f"  Avg inference:     {metrics.get('avg_inference_time', 0):.3f}s/image")
        print(f"  Total time:        {total_time:.1f}s")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    if len(results) > 1:
        recommendations = print_comparison_table(results)
    else:
        recommendations = {'track_a': 'baseline', 'track_b': 'baseline', 'production': 'baseline'}
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    # Convert numpy types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        return obj
    
    results_json = convert_to_json_serializable(results)
    
    results_file = output_dir / 'phase7_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Results saved to: {results_file}")
    
    # Save summary
    summary = {
        'recommendations': recommendations,
        'models_evaluated': list(results.keys()),
        'command_line_args': vars(args)
    }
    
    summary_file = output_dir / 'phase7_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}\n")
    
    print("="*70)
    print("PHASE 7 EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n  Next steps:")
    print(f"  1. If Track A: Use '{recommendations['track_a']}' model")
    print(f"  2. If Track B: Use '{recommendations['track_b']}' model")
    print(f"  3. Generate final submissions with scripts/generate_submission.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
