"""
Super-Resolution Dataset Loader (Optimized for Pre-Generated LR-HR Pairs)
==========================================================================
Efficient dataset loading for NTIRE 2025 super-resolution training.

Features:
1. Direct LR-HR pair loading (no on-the-fly generation)
2. Intelligent filename matching (handles various naming patterns)
3. Paired augmentation pipeline (maintains LR-HR correspondence)
4. Efficient caching and batching
5. DF2K support (DIV2K + Flickr2K)
6. Multi-worker data loading
7. Memory-efficient patch extraction

Expected: 10-15% faster training vs on-the-fly LR generation!

Author: NTIRE SR Team
"""

import os
import sys
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import random
from tqdm import tqdm
import warnings

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import augmentations (handle both module and standalone execution)
try:
    from .augmentations import SRTrainAugmentation
except ImportError:
    from augmentations import SRTrainAugmentation


class SRDataset(Dataset):
    """
    Super-Resolution Dataset for pre-generated LR-HR pairs.
    
    Optimized for datasets where LR images already exist (no on-the-fly generation).
    Supports various filename patterns and directory structures.
    
    Args:
        hr_dir: Path to HR images directory
        lr_dir: Path to LR images directory
        lr_patch_size: LR patch size (64 or 96)
        scale: Upscaling factor (4)
        augment: Whether to apply augmentations
        cache_data: Whether to cache images in memory
        repeat_factor: Repeat dataset for more training samples
        validate_pairs: Check if all LR-HR pairs exist
    """
    
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        lr_patch_size: int = 64,
        scale: int = 4,
        augment: bool = True,
        cache_data: bool = False,
        repeat_factor: int = 1,
        validate_pairs: bool = True
    ):
        super().__init__()
        
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.scale = scale
        self.augment = augment
        self.cache_data = cache_data
        self.repeat_factor = repeat_factor
        
        # Verify directories exist
        if not self.hr_dir.exists():
            raise ValueError(f"HR directory not found: {hr_dir}")
        if not self.lr_dir.exists():
            raise ValueError(f"LR directory not found: {lr_dir}")
        
        # Load file paths
        print(f"\n  Loading image pairs from:")
        print(f"    HR: {hr_dir}")
        print(f"    LR: {lr_dir}")
        
        self.hr_paths = self._load_image_paths(self.hr_dir)
        self.lr_paths = self._load_image_paths(self.lr_dir)
        
        print(f"  Found: {len(self.hr_paths)} HR images, {len(self.lr_paths)} LR images")
        
        # Match LR to HR by filename
        self.paired_paths = self._match_lr_hr_paths(validate=validate_pairs)
        
        print(f"  Matched pairs: {len(self.paired_paths)}")
        
        if len(self.paired_paths) == 0:
            raise ValueError("No matching LR-HR pairs found! Check filenames.")
        
        # Initialize augmentation pipeline
        if self.augment:
            self.augmentation = SRTrainAugmentation(
                lr_patch_size=lr_patch_size,
                scale=scale,
                use_flip=True,
                use_rotation=True,
                use_color_jitter=True,
                use_cutblur=False  # Enable if needed
            )
        else:
            self.augmentation = None
        
        # Cache data if requested
        self.cache = {}
        if cache_data:
            print("  Caching dataset in memory...")
            self._cache_dataset()
    
    def _load_image_paths(self, directory: Path) -> List[Path]:
        """Load all image paths from directory."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.tif', '*.tiff']
        paths = []
        for ext in extensions:
            paths.extend(directory.glob(ext))
        return sorted(paths)
    
    def _match_lr_hr_paths(self, validate: bool = True) -> List[Tuple[Path, Path]]:
        """
        Match LR and HR paths by filename.
        
        Handles common naming patterns:
        - img_001.png (HR) ↔ img_001.png (LR)
        - 0001.png (HR) ↔ 0001x4.png (LR)
        - image001.png (HR) ↔ image001_LR.png (LR)
        - DIV2K_0001.png (HR) ↔ DIV2K_0001x4.png (LR)
        
        Args:
            validate: If True, verify all pairs exist
            
        Returns:
            List of (lr_path, hr_path) tuples
        """
        # Create HR lookup dictionary (stem → path)
        hr_dict = {p.stem: p for p in self.hr_paths}
        
        # Create LR lookup with cleaned stems
        lr_dict = {}
        lr_to_original = {}  # Map cleaned stem back to original
        
        for lr_path in self.lr_paths:
            stem = lr_path.stem
            # Remove common LR suffixes (try multiple patterns)
            clean_stem = stem
            
            # Pattern: 0001x4 -> 0001
            for scale_suffix in ['x4', 'x2', 'x3', 'x8']:
                clean_stem = clean_stem.replace(scale_suffix, '')
            
            # Pattern: 0001_LR -> 0001
            for lr_suffix in ['_LR', '_lr', 'LR', 'lr', '_bicubic', '_BICUBIC']:
                clean_stem = clean_stem.replace(lr_suffix, '')
            
            # Remove trailing underscores
            clean_stem = clean_stem.rstrip('_')
            
            lr_dict[clean_stem] = lr_path
            lr_to_original[clean_stem] = stem
        
        # Match pairs
        pairs = []
        unmatched_hr = []
        
        for hr_stem, hr_path in hr_dict.items():
            if hr_stem in lr_dict:
                pairs.append((lr_dict[hr_stem], hr_path))
            else:
                unmatched_hr.append(hr_stem)
        
        # If no matches found with cleaning, try exact matching
        if len(pairs) == 0:
            lr_stems = {p.stem: p for p in self.lr_paths}
            for hr_stem, hr_path in hr_dict.items():
                if hr_stem in lr_stems:
                    pairs.append((lr_stems[hr_stem], hr_path))
        
        # Report unmatched files
        if validate and unmatched_hr and len(pairs) > 0:
            if len(unmatched_hr) <= 10:  # Only show if not too many
                print(f"\n  Warning: {len(unmatched_hr)} HR images without matching LR:")
                for stem in unmatched_hr[:5]:
                    print(f"    - {stem}")
                if len(unmatched_hr) > 5:
                    print(f"    ... and {len(unmatched_hr) - 5} more")
        
        return pairs
    
    def _cache_dataset(self):
        """Cache entire dataset in memory (only for small datasets)."""
        for idx in tqdm(range(len(self.paired_paths)), desc="  Caching"):
            lr_path, hr_path = self.paired_paths[idx]
            lr_img = self._load_image(lr_path)
            hr_img = self._load_image(hr_path)
            self.cache[idx] = (lr_img, hr_img)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load image from disk.
        
        Returns:
            Image in RGB format, float32, range [0, 1]
        """
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def __len__(self) -> int:
        """Dataset length with repeat factor."""
        return len(self.paired_paths) * self.repeat_factor
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
            - 'lr': LR patch [3, H, W]
            - 'hr': HR patch [3, H*scale, W*scale]
            - 'filename': Image filename
        """
        # Handle repeat factor
        real_idx = idx % len(self.paired_paths)
        
        # Load from cache or disk
        if self.cache_data and real_idx in self.cache:
            lr_img, hr_img = self.cache[real_idx]
            lr_img = lr_img.copy()  # Avoid modifying cached copy
            hr_img = hr_img.copy()
        else:
            lr_path, hr_path = self.paired_paths[real_idx]
            lr_img = self._load_image(lr_path)
            hr_img = self._load_image(hr_path)
        
        # Apply augmentations
        if self.augment and self.augmentation is not None:
            lr_img, hr_img = self.augmentation(lr_img, hr_img)
        else:
            # Center crop if no augmentation
            lr_h, lr_w = lr_img.shape[:2]
            hr_h, hr_w = hr_img.shape[:2]
            
            # Verify scale relationship
            expected_hr_h = lr_h * self.scale
            expected_hr_w = lr_w * self.scale
            
            if hr_h != expected_hr_h or hr_w != expected_hr_w:
                # Resize HR to match expected dimensions
                hr_img = cv2.resize(
                    hr_img,
                    (expected_hr_w, expected_hr_h),
                    interpolation=cv2.INTER_CUBIC
                )
                hr_h, hr_w = hr_img.shape[:2]
            
            # Crop to patch size (center crop for validation)
            if lr_h >= self.lr_patch_size and lr_w >= self.lr_patch_size:
                lr_top = (lr_h - self.lr_patch_size) // 2
                lr_left = (lr_w - self.lr_patch_size) // 2
                lr_img = lr_img[
                    lr_top:lr_top + self.lr_patch_size,
                    lr_left:lr_left + self.lr_patch_size
                ]
                
                hr_top = lr_top * self.scale
                hr_left = lr_left * self.scale
                hr_img = hr_img[
                    hr_top:hr_top + self.hr_patch_size,
                    hr_left:hr_left + self.hr_patch_size
                ]
            else:
                # Resize if too small
                lr_img = cv2.resize(
                    lr_img,
                    (self.lr_patch_size, self.lr_patch_size),
                    interpolation=cv2.INTER_CUBIC
                )
                hr_img = cv2.resize(
                    hr_img,
                    (self.hr_patch_size, self.hr_patch_size),
                    interpolation=cv2.INTER_CUBIC
                )
        
        # Convert to torch tensors [C, H, W]
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).float()
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1)).float()
        
        # Ensure values in [0, 1]
        lr_tensor = lr_tensor.clamp(0, 1)
        hr_tensor = hr_tensor.clamp(0, 1)
        
        # Get filename
        _, hr_path = self.paired_paths[real_idx]
        
        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'filename': hr_path.name
        }


class DF2KDataset(SRDataset):
    """
    DF2K Dataset with pre-generated LR-HR pairs.
    
    Automatically detects directory structure patterns:
    
    Pattern 1 (Custom):
        df2k_root/
        ├── train_LR/
        ├── train_HR/
        ├── val_LR/
        └── val_HR/
    
    Pattern 2 (DIV2K Standard):
        df2k_root/
        ├── DIV2K_train_LR_bicubic/X4/
        ├── DIV2K_train_HR/
        ├── DIV2K_valid_LR_bicubic/X4/
        └── DIV2K_valid_HR/
    
    Args:
        root_dir: Path to DF2K dataset root
        split: 'train' or 'val'
        lr_patch_size: LR patch size
        scale: Upscaling factor
        augment: Enable augmentation
        cache_data: Cache in memory
        repeat_factor: Dataset repetition (20 recommended for training)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        lr_patch_size: int = 64,
        scale: int = 4,
        augment: bool = True,
        cache_data: bool = False,
        repeat_factor: int = 20
    ):
        root = Path(root_dir)
        
        # Try multiple directory naming patterns
        lr_dir, hr_dir = self._find_directories(root, split, scale)
        
        print(f"\nLoading DF2K {split} dataset...")
        
        # Initialize parent class
        super().__init__(
            hr_dir=str(hr_dir),
            lr_dir=str(lr_dir),
            lr_patch_size=lr_patch_size,
            scale=scale,
            augment=augment,
            cache_data=cache_data,
            repeat_factor=repeat_factor,
            validate_pairs=True
        )
    
    def _find_directories(
        self, 
        root: Path, 
        split: str, 
        scale: int
    ) -> Tuple[Path, Path]:
        """Find LR and HR directories using multiple patterns."""
        
        # Define patterns to try
        if split == 'train':
            patterns = [
                # Pattern 1: Custom naming
                (root / 'train_LR', root / 'train_HR'),
                # Pattern 2: DIV2K standard
                (root / 'DIV2K_train_LR_bicubic' / f'X{scale}', root / 'DIV2K_train_HR'),
                # Pattern 3: Simple
                (root / 'LR' / 'train', root / 'HR' / 'train'),
                # Pattern 4: Flat structure
                (root / 'LR_train', root / 'HR_train'),
                # Pattern 5: DF2K specific
                (root / 'DF2K_train_LR_bicubic' / f'X{scale}', root / 'DF2K_train_HR'),
            ]
        else:
            patterns = [
                # Pattern 1: Custom naming
                (root / 'val_LR', root / 'val_HR'),
                # Pattern 2: DIV2K standard
                (root / 'DIV2K_valid_LR_bicubic' / f'X{scale}', root / 'DIV2K_valid_HR'),
                # Pattern 3: Simple
                (root / 'LR' / 'val', root / 'HR' / 'val'),
                # Pattern 4: Flat structure
                (root / 'LR_val', root / 'HR_val'),
                # Pattern 5: DF2K specific
                (root / 'DF2K_valid_LR_bicubic' / f'X{scale}', root / 'DF2K_valid_HR'),
                # Pattern 6: test as val
                (root / 'test_LR', root / 'test_HR'),
            ]
        
        # Try each pattern
        for lr_dir, hr_dir in patterns:
            if lr_dir.exists() and hr_dir.exists():
                return lr_dir, hr_dir
        
        # If no pattern matched, provide helpful error
        tried = "\n".join([f"  - LR: {p[0]}, HR: {p[1]}" for p in patterns])
        raise ValueError(
            f"Could not find {split} directories. Tried patterns:\n{tried}\n\n"
            f"Please ensure your dataset follows one of these patterns."
        )


class ValidationDataset(SRDataset):
    """
    Validation dataset that returns full images (no patching).
    
    Used for PSNR/SSIM evaluation during training.
    """
    
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        scale: int = 4,
        max_size: Optional[int] = None
    ):
        """
        Args:
            hr_dir: Path to HR images
            lr_dir: Path to LR images
            scale: Upscaling factor
            max_size: Maximum image dimension (resize if larger)
        """
        self.max_size = max_size
        
        # Initialize with no augmentation and no patching
        super().__init__(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            lr_patch_size=64,  # Will be ignored
            scale=scale,
            augment=False,
            cache_data=False,
            repeat_factor=1,
            validate_pairs=True
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get full image for validation."""
        lr_path, hr_path = self.paired_paths[idx]
        
        lr_img = self._load_image(lr_path)
        hr_img = self._load_image(hr_path)
        
        # Optionally resize large images
        if self.max_size is not None:
            lr_h, lr_w = lr_img.shape[:2]
            if max(lr_h, lr_w) > self.max_size // self.scale:
                # Calculate new size
                if lr_h > lr_w:
                    new_lr_h = self.max_size // self.scale
                    new_lr_w = int(lr_w * new_lr_h / lr_h)
                else:
                    new_lr_w = self.max_size // self.scale
                    new_lr_h = int(lr_h * new_lr_w / lr_w)
                
                lr_img = cv2.resize(lr_img, (new_lr_w, new_lr_h), interpolation=cv2.INTER_CUBIC)
                
                new_hr_h = new_lr_h * self.scale
                new_hr_w = new_lr_w * self.scale
                hr_img = cv2.resize(hr_img, (new_hr_w, new_hr_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to torch tensors
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).float()
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1)).float()
        
        return {
            'lr': lr_tensor.clamp(0, 1),
            'hr': hr_tensor.clamp(0, 1),
            'filename': hr_path.name
        }


def create_dataloaders(
    train_hr_dir: str,
    train_lr_dir: str,
    val_hr_dir: Optional[str] = None,
    val_lr_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    lr_patch_size: int = 64,
    scale: int = 4,
    pin_memory: bool = True,
    repeat_factor: int = 20,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders for pre-generated LR-HR pairs.
    
    Args:
        train_hr_dir: Path to training HR images
        train_lr_dir: Path to training LR images
        val_hr_dir: Path to validation HR images (optional)
        val_lr_dir: Path to validation LR images (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        lr_patch_size: LR patch size
        scale: Upscaling factor
        pin_memory: Pin memory for faster GPU transfer
        repeat_factor: Repeat dataset for more samples
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        (train_loader, val_loader)
    """
    print("\n" + "=" * 70)
    print("CREATING DATALOADERS (PRE-GENERATED LR-HR PAIRS)")
    print("=" * 70)
    
    # Training dataset
    print("\n--- Training Dataset ---")
    train_dataset = SRDataset(
        hr_dir=train_hr_dir,
        lr_dir=train_lr_dir,
        lr_patch_size=lr_patch_size,
        scale=scale,
        augment=True,
        cache_data=False,
        repeat_factor=repeat_factor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    print(f"\nTrain DataLoader:")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples per epoch: {len(train_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Validation dataset (optional)
    val_loader = None
    if val_hr_dir is not None and val_lr_dir is not None:
        print("\n--- Validation Dataset ---")
        val_dataset = ValidationDataset(
            hr_dir=val_hr_dir,
            lr_dir=val_lr_dir,
            scale=scale,
            max_size=512  # Limit memory usage
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory
        )
        
        print(f"\nValidation DataLoader:")
        print(f"  Samples: {len(val_dataset)}")
    
    print("\n" + "=" * 70 + "\n")
    
    return train_loader, val_loader


def create_df2k_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    lr_patch_size: int = 64,
    scale: int = 4,
    pin_memory: bool = True,
    repeat_factor: int = 20
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create DF2K dataloaders.
    
    Args:
        root_dir: Path to DF2K root (contains train_LR, train_HR, etc.)
        batch_size: Training batch size
        num_workers: Number of workers
        lr_patch_size: LR patch size
        scale: Upscaling factor
        pin_memory: Pin memory
        repeat_factor: Dataset repetition
        
    Returns:
        (train_loader, val_loader)
    """
    print("\n" + "=" * 70)
    print("CREATING DF2K DATALOADERS")
    print("=" * 70)
    
    # Training dataset
    print("\n--- Training ---")
    train_dataset = DF2KDataset(
        root_dir=root_dir,
        split='train',
        lr_patch_size=lr_patch_size,
        scale=scale,
        augment=True,
        cache_data=False,
        repeat_factor=repeat_factor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    # Validation dataset
    print("\n--- Validation ---")
    val_dataset = DF2KDataset(
        root_dir=root_dir,
        split='val',
        lr_patch_size=lr_patch_size,
        scale=scale,
        augment=False,
        cache_data=False,
        repeat_factor=1
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory
    )
    
    print(f"\n✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✓ Val: {len(val_dataset)} samples")
    print("\n" + "=" * 70 + "\n")
    
    return train_loader, val_loader


# ============================================================================
# Testing Functions
# ============================================================================

def test_dataset():
    """Test dataset loading with dummy LR-HR pairs."""
    print("\n" + "=" * 70)
    print("DATASET MODULE - TEST (PRE-GENERATED LR-HR)")
    print("=" * 70)
    
    import tempfile
    import shutil
    
    # Create test directories
    temp_dir = Path(tempfile.mkdtemp())
    lr_dir = temp_dir / 'LR'
    hr_dir = temp_dir / 'HR'
    lr_dir.mkdir()
    hr_dir.mkdir()
    
    print(f"\nCreating test LR-HR pairs in: {temp_dir}")
    
    # Create matching LR-HR pairs
    for i in range(5):
        # HR image (512x512)
        hr_img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        hr_img_bgr = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(hr_dir / f'test_{i:03d}.png'), hr_img_bgr)
        
        # LR image (128x128, scale=4)
        lr_img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        lr_img_bgr = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(lr_dir / f'test_{i:03d}.png'), lr_img_bgr)
    
    print(f"  Created 5 LR-HR pairs")
    
    # Test 1: Pair matching
    print("\n--- Test 1: LR-HR Pair Matching ---")
    dataset = SRDataset(
        hr_dir=str(hr_dir),
        lr_dir=str(lr_dir),
        lr_patch_size=64,
        scale=4,
        augment=False,
        cache_data=False
    )
    print(f"  Matched pairs: {len(dataset.paired_paths)}")
    assert len(dataset.paired_paths) == 5, "Should match all 5 pairs"
    print("  [PASSED]")
    
    # Test 2: Sample loading
    print("\n--- Test 2: Sample Loading ---")
    sample = dataset[0]
    print(f"  LR: {sample['lr'].shape}")
    print(f"  HR: {sample['hr'].shape}")
    assert sample['lr'].shape == (3, 64, 64), f"LR shape incorrect: {sample['lr'].shape}"
    assert sample['hr'].shape == (3, 256, 256), f"HR shape incorrect: {sample['hr'].shape}"
    print("  [PASSED]")
    
    # Test 3: Value range
    print("\n--- Test 3: Value Range ---")
    assert sample['lr'].min() >= 0 and sample['lr'].max() <= 1, "LR values out of range"
    assert sample['hr'].min() >= 0 and sample['hr'].max() <= 1, "HR values out of range"
    print(f"  LR: [{sample['lr'].min():.3f}, {sample['lr'].max():.3f}]")
    print(f"  HR: [{sample['hr'].min():.3f}, {sample['hr'].max():.3f}]")
    print("  [PASSED]")
    
    # Test 4: DataLoader
    print("\n--- Test 4: DataLoader ---")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"  Batch LR: {batch['lr'].shape}")
    print(f"  Batch HR: {batch['hr'].shape}")
    assert batch['lr'].shape == (2, 3, 64, 64), f"Batch LR incorrect: {batch['lr'].shape}"
    assert batch['hr'].shape == (2, 3, 256, 256), f"Batch HR incorrect: {batch['hr'].shape}"
    print("  [PASSED]")
    
    # Test 5: Augmented dataset
    print("\n--- Test 5: Augmented Dataset ---")
    aug_dataset = SRDataset(
        hr_dir=str(hr_dir),
        lr_dir=str(lr_dir),
        lr_patch_size=64,
        scale=4,
        augment=True,
        repeat_factor=3
    )
    assert len(aug_dataset) == 15, f"Repeat factor not working: {len(aug_dataset)}"
    sample_aug = aug_dataset[0]
    assert sample_aug['lr'].shape == (3, 64, 64)
    print(f"  Repeated length: {len(aug_dataset)} (5 x 3)")
    print(f"  Augmented sample shape: {sample_aug['lr'].shape}")
    print("  [PASSED]")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("✓ ALL DATASET TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    test_dataset()
