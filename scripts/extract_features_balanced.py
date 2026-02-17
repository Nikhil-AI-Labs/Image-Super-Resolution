"""
scripts/extract_features_balanced.py
====================================
Balanced Multi-GPU Feature Extractor for Pre-Computing Expert Outputs.

NOW EXTRACTS BOTH TRAINING AND VALIDATION SETS!

Strategy:
  - GPU 0: Runs HAT (Heavy ~155M params, ~3-4 seconds per image)
  - GPU 1: Runs DAT + NAFNet (Medium + Light, ~2-3 seconds combined)
  
This balanced approach ensures both GPUs finish at roughly the same time.
Results are saved as separate .pt files that can be merged at load time.

Output Structure:
  dataset/DF2K/cached_features_train/   <- Training features
  dataset/DF2K/cached_features_val/     <- Validation features

Usage:
    # Full extraction (all images, both train and val)
    python scripts/extract_features_balanced.py
    
    # Test mode (5 images only)
    python scripts/extract_features_balanced.py --test-mode --num-samples 5
    
    # Train only (skip validation)
    python scripts/extract_features_balanced.py --train-only
    
    # Val only (skip training)
    python scripts/extract_features_balanced.py --val-only

Author: NTIRE SR Team
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Optional, Tuple, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract and cache expert features for fast training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run on small subset for testing')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples in test mode')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already extracted files')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Run on single GPU (slower but uses less memory)')
    parser.add_argument('--train-only', action='store_true',
                        help='Only extract training set')
    parser.add_argument('--val-only', action='store_true',
                        help='Only extract validation set')
    return parser.parse_args()


def worker_hat(
    gpu_id: int,
    indices: list,
    hr_dir: str,
    lr_dir: str,
    scale: int,
    save_dir: Path,
    resume: bool = True,
    split_name: str = "train"
):
    """
    Worker for GPU 0: Runs ONLY HAT.
    
    HAT is the heaviest model (~155M params) so it gets a dedicated GPU.
    Saves results as {filename}_hat_part.pt
    """
    device = torch.device(f'cuda:{gpu_id}')
    print(f"[GPU {gpu_id}] Initializing HAT worker for {split_name}...")
    
    # Import here to avoid CUDA context issues before fork
    from src.models.expert_loader import ExpertEnsemble
    from src.data.dataset import SRDataset
    
    # Load only HAT
    ensemble = ExpertEnsemble(device=device, upscale=4)
    ensemble.load_hat(checkpoint_path="pretrained/HAT-L_SRx4_ImageNet-pretrain.pth")
    
    # Register hooks for feature capture (HAT: conv_after_body → 180ch)
    ensemble._register_all_hooks()
    print(f"[GPU {gpu_id}] Hooks registered for HAT feature capture")
    
    # Create dataset (no augmentation for extraction)
    dataset = SRDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        lr_patch_size=64,  # Not cropping for full image extraction
        scale=scale,
        augment=False,
        repeat_factor=1
    )
    
    print(f"[GPU {gpu_id}] Starting HAT processing ({len(indices)} images, {split_name})...")
    
    skipped = 0
    processed = 0
    
    with torch.no_grad():
        pbar = tqdm(indices, position=0, desc=f"HAT {split_name} (GPU {gpu_id})", leave=True)
        
        for idx in pbar:
            sample = dataset[idx]
            filename = Path(sample['filename']).stem
            save_path = save_dir / f"{filename}_hat_part.pt"
            
            # Skip if already exists and resume mode
            if resume and save_path.exists():
                skipped += 1
                continue
            
            lr = sample['lr'].unsqueeze(0).to(device)
            
            # Enable feature capture for collaborative learning
            ensemble._capture_features = True
            ensemble._captured_features = {}
            
            # Run HAT forward
            try:
                hat_output = ensemble.forward_hat(lr)
            except Exception as e:
                print(f"\n[GPU {gpu_id}] Error processing {filename}: {e}")
                continue
            
            # Get captured features
            hat_feat = ensemble._captured_features.get('hat', None)
            
            # Prepare payload - store on CPU to save GPU memory
            payload = {
                'outputs': {
                    'hat': hat_output.cpu()
                },
                'features': {
                    'hat': hat_feat.cpu() if hat_feat is not None else torch.zeros(1, 180, 64, 64)
                },
                'lr': sample['lr'],  # Already on CPU
                'hr': sample['hr'],  # Already on CPU
                'filename': filename
            }
            
            torch.save(payload, save_path)
            processed += 1
            
            pbar.set_postfix({'processed': processed, 'skipped': skipped})
            
            # Clear GPU cache periodically
            if processed % 50 == 0:
                torch.cuda.empty_cache()
    
    print(f"[GPU {gpu_id}] HAT {split_name} complete. Processed: {processed}, Skipped: {skipped}")


def worker_others(
    gpu_id: int,
    indices: list,
    hr_dir: str,
    lr_dir: str,
    scale: int,
    save_dir: Path,
    resume: bool = True,
    split_name: str = "train"
):
    """
    Worker for GPU 1: Runs DAT + NAFNet.
    
    These two smaller models combined roughly balance HAT's workload.
    Saves results as {filename}_rest_part.pt
    """
    device = torch.device(f'cuda:{gpu_id}')
    print(f"[GPU {gpu_id}] Initializing DAT + NAFNet worker for {split_name}...")
    
    # Import here to avoid CUDA context issues before fork
    from src.models.expert_loader import ExpertEnsemble
    from src.data.dataset import SRDataset
    
    # Load DAT and NAFNet
    ensemble = ExpertEnsemble(device=device, upscale=4)
    ensemble.load_dat(checkpoint_path="pretrained/DAT_x4.pth")
    ensemble.load_nafnet(checkpoint_path="pretrained/NAFNet-SIDD-width64.pth")
    
    # Register hooks for feature capture
    # DAT: conv_after_body → 180ch, NAFNet: ending INPUT → 64ch
    ensemble._register_all_hooks()
    print(f"[GPU {gpu_id}] Hooks registered for DAT+NAFNet feature capture")
    
    # Create dataset (no augmentation for extraction)
    dataset = SRDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        lr_patch_size=64,
        scale=scale,
        augment=False,
        repeat_factor=1
    )
    
    print(f"[GPU {gpu_id}] Starting DAT+NAFNet processing ({len(indices)} images, {split_name})...")
    
    skipped = 0
    processed = 0
    
    with torch.no_grad():
        pbar = tqdm(indices, position=1, desc=f"DAT+NAFNet {split_name} (GPU {gpu_id})", leave=True)
        
        for idx in pbar:
            sample = dataset[idx]
            filename = Path(sample['filename']).stem
            save_path = save_dir / f"{filename}_rest_part.pt"
            
            # Skip if already exists and resume mode
            if resume and save_path.exists():
                skipped += 1
                continue
            
            lr = sample['lr'].unsqueeze(0).to(device)
            
            # Enable feature capture
            ensemble._capture_features = True
            ensemble._captured_features = {}
            
            # Run DAT and NAFNet forward
            try:
                dat_output = ensemble.forward_dat(lr)
                naf_output = ensemble.forward_nafnet(lr)
            except Exception as e:
                print(f"\n[GPU {gpu_id}] Error processing {filename}: {e}")
                continue
            
            # Get captured features
            dat_feat = ensemble._captured_features.get('dat', None)
            naf_feat = ensemble._captured_features.get('nafnet', None)
            
            # Prepare payload
            payload = {
                'outputs': {
                    'dat': dat_output.cpu(),
                    'nafnet': naf_output.cpu()
                },
                'features': {
                    'dat': dat_feat.cpu() if dat_feat is not None else torch.zeros(1, 180, 64, 64),
                    'nafnet': naf_feat.cpu() if naf_feat is not None else torch.zeros(1, 64, 64, 64)
                },
                'filename': filename
            }
            
            torch.save(payload, save_path)
            processed += 1
            
            pbar.set_postfix({'processed': processed, 'skipped': skipped})
            
            # Clear GPU cache periodically
            if processed % 50 == 0:
                torch.cuda.empty_cache()
    
    print(f"[GPU {gpu_id}] DAT+NAFNet {split_name} complete. Processed: {processed}, Skipped: {skipped}")


def worker_single_gpu(
    gpu_id: int,
    indices: list,
    hr_dir: str,
    lr_dir: str,
    scale: int,
    save_dir: Path,
    resume: bool = True,
    split_name: str = "train"
):
    """
    Single GPU worker: Runs all experts sequentially.
    
    Slower but uses only one GPU. Saves both parts in one go.
    """
    device = torch.device(f'cuda:{gpu_id}')
    print(f"[GPU {gpu_id}] Initializing single-GPU worker (all experts) for {split_name}...")
    
    from src.models.expert_loader import ExpertEnsemble
    from src.data.dataset import SRDataset
    
    # Load all experts
    ensemble = ExpertEnsemble(device=device, upscale=4)
    ensemble.load_hat(checkpoint_path="pretrained/HAT-L_SRx4_ImageNet-pretrain.pth")
    ensemble.load_dat(checkpoint_path="pretrained/DAT_x4.pth")
    ensemble.load_nafnet(checkpoint_path="pretrained/NAFNet-SIDD-width64.pth")
    
    # Register hooks for feature capture
    # HAT/DAT: conv_after_body → 180ch, NAFNet: ending INPUT → 64ch  
    ensemble._register_all_hooks()
    print(f"[GPU {gpu_id}] Hooks registered for all experts")
    
    # Create dataset
    dataset = SRDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        lr_patch_size=64,
        scale=scale,
        augment=False,
        repeat_factor=1
    )
    
    print(f"[GPU {gpu_id}] Starting single-GPU extraction ({len(indices)} images, {split_name})...")
    
    skipped = 0
    processed = 0
    
    with torch.no_grad():
        pbar = tqdm(indices, desc=f"All Experts {split_name}", leave=True)
        
        for idx in pbar:
            sample = dataset[idx]
            filename = Path(sample['filename']).stem
            
            hat_path = save_dir / f"{filename}_hat_part.pt"
            rest_path = save_dir / f"{filename}_rest_part.pt"
            
            # Skip if both already exist and resume mode
            if resume and hat_path.exists() and rest_path.exists():
                skipped += 1
                continue
            
            lr = sample['lr'].unsqueeze(0).to(device)
            
            # Enable feature capture
            ensemble._capture_features = True
            ensemble._captured_features = {}
            
            # Run all experts
            try:
                hat_output = ensemble.forward_hat(lr)
                hat_feat = ensemble._captured_features.get('hat', torch.zeros(1, 180, 64, 64))
                
                dat_output = ensemble.forward_dat(lr)
                dat_feat = ensemble._captured_features.get('dat', torch.zeros(1, 180, 64, 64))
                
                naf_output = ensemble.forward_nafnet(lr)
                naf_feat = ensemble._captured_features.get('nafnet', torch.zeros(1, 64, 64, 64))
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                continue
            
            # Save HAT part
            hat_payload = {
                'outputs': {'hat': hat_output.cpu()},
                'features': {'hat': hat_feat.cpu() if isinstance(hat_feat, torch.Tensor) else hat_feat},
                'lr': sample['lr'],
                'hr': sample['hr'],
                'filename': filename
            }
            torch.save(hat_payload, hat_path)
            
            # Save rest part
            rest_payload = {
                'outputs': {
                    'dat': dat_output.cpu(),
                    'nafnet': naf_output.cpu()
                },
                'features': {
                    'dat': dat_feat.cpu() if isinstance(dat_feat, torch.Tensor) else dat_feat,
                    'nafnet': naf_feat.cpu() if isinstance(naf_feat, torch.Tensor) else naf_feat
                },
                'filename': filename
            }
            torch.save(rest_payload, rest_path)
            
            processed += 1
            pbar.set_postfix({'processed': processed, 'skipped': skipped})
            
            # Clear GPU cache periodically
            if processed % 20 == 0:
                torch.cuda.empty_cache()
    
    print(f"[GPU {gpu_id}] {split_name} extraction complete. Processed: {processed}, Skipped: {skipped}")


def extract_split(
    split_name: str,
    hr_dir: str,
    lr_dir: str,
    scale: int,
    save_dir: Path,
    args,
    num_gpus: int
):
    """Extract features for a single split (train or val)."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING {split_name.upper()} SET")
    print(f"{'='*70}")
    print(f"  HR dir: {hr_dir}")
    print(f"  LR dir: {lr_dir}")
    print(f"  Save to: {save_dir}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset size
    from src.data.dataset import SRDataset
    
    dataset = SRDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        lr_patch_size=64,
        scale=scale,
        augment=False,
        repeat_factor=1
    )
    
    total_samples = len(dataset)
    
    # Determine indices to process
    if args.test_mode:
        indices = list(range(min(args.num_samples, total_samples)))
        print(f"  Test mode: {len(indices)} samples")
    else:
        indices = list(range(total_samples))
        print(f"  Full extraction: {total_samples} samples")
    
    start_time = time.time()
    
    if args.single_gpu or num_gpus < 2:
        # Single GPU mode
        print(f"\n  Running {split_name} in SINGLE GPU mode...")
        worker_single_gpu(0, indices, hr_dir, lr_dir, scale, save_dir, 
                         resume=args.resume, split_name=split_name)
    else:
        # Multi-GPU parallel extraction
        print(f"\n  Running {split_name} in MULTI-GPU parallel mode...")
        
        # Start both workers on the same indices (they save different files)
        p1 = mp.Process(
            target=worker_hat, 
            args=(0, indices, hr_dir, lr_dir, scale, save_dir, args.resume, split_name)
        )
        p2 = mp.Process(
            target=worker_others, 
            args=(1, indices, hr_dir, lr_dir, scale, save_dir, args.resume, split_name)
        )
        
        p1.start()
        p2.start()
        
        p1.join()
        p2.join()
    
    elapsed = time.time() - start_time
    
    # Verify output
    hat_files = list(save_dir.glob("*_hat_part.pt"))
    rest_files = list(save_dir.glob("*_rest_part.pt"))
    
    print(f"\n  {split_name.upper()} Results:")
    print(f"    Time: {elapsed/60:.1f} minutes")
    print(f"    HAT parts: {len(hat_files)}")
    print(f"    Rest parts: {len(rest_files)}")
    
    return len(indices), elapsed


def main():
    """Main entry point for feature extraction."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("BALANCED MULTI-GPU FEATURE EXTRACTOR")
    print("=" * 70)
    print("Extracts BOTH Training AND Validation sets")
    print("Strategy: HAT on GPU 0, DAT+NAFNet on GPU 1")
    print("=" * 70 + "\n")
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset']
    scale = dataset_config.get('scale', 4)
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"         Memory: {mem:.1f} GB")
    
    # Set multiprocessing start method (only once)
    if num_gpus >= 2 and not args.single_gpu:
        mp.set_start_method('spawn', force=True)
    
    total_images = 0
    total_time = 0
    
    # ========================================
    # EXTRACT TRAINING SET
    # ========================================
    if not args.val_only:
        train_root = dataset_config['train']['root']
        train_hr = os.path.join(train_root, dataset_config['train'].get('hr_subdir', 'train_HR'))
        train_lr = os.path.join(train_root, dataset_config['train'].get('lr_subdir', 'train_LR'))
        train_cache = Path(train_root) / 'cached_features_train'
        
        n, t = extract_split('train', train_hr, train_lr, scale, train_cache, args, num_gpus)
        total_images += n
        total_time += t
    
    # ========================================
    # EXTRACT VALIDATION SET (CRITICAL!)
    # ========================================
    if not args.train_only:
        val_root = dataset_config['val']['root']
        val_hr = os.path.join(val_root, dataset_config['val'].get('hr_subdir', 'val_HR'))
        val_lr = os.path.join(val_root, dataset_config['val'].get('lr_subdir', 'val_LR'))
        val_cache = Path(val_root) / 'cached_features_val'
        
        n, t = extract_split('val', val_hr, val_lr, scale, val_cache, args, num_gpus)
        total_images += n
        total_time += t
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("ALL EXTRACTIONS COMPLETE!")
    print("=" * 70)
    print(f"  Total images: {total_images}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    if total_images > 0:
        print(f"  Avg time per image: {total_time/total_images:.2f}s")
    
    print("\nCached directories created:")
    if not args.val_only:
        print(f"  Train: {dataset_config['train']['root']}/cached_features_train/")
    if not args.train_only:
        print(f"  Val:   {dataset_config['val']['root']}/cached_features_val/")
    
    print("\nReady for training! Run:")
    print("  python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
