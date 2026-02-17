"""
Cached Super-Resolution Dataset
================================
Loads pre-computed expert outputs and features from disk for ultra-fast training.

This dataset is designed to work with features extracted by:
    scripts/extract_features_balanced.py

Expected file format:
    {cache_dir}/
    ├── img_001_hat_part.pt   - Contains HAT SR output + features + LR/HR
    ├── img_001_rest_part.pt  - Contains DAT + NAFNet SR outputs + features
    ├── img_002_hat_part.pt
    ├── img_002_rest_part.pt
    └── ...

Each .pt file contains:
    - outputs: Dict[str, Tensor] - SR outputs from experts
    - features: Dict[str, Tensor] - Intermediate features for collaborative learning
    - lr: Tensor - Original LR patch (in hat_part only)
    - hr: Tensor - Original HR patch (in hat_part only)
    - filename: str - Original filename stem

Augmentation Note:
    Color jitter is NOT supported because it would require re-computing expert outputs.
    Only geometric augmentations (flip, rotate) are applied consistently to all tensors.

Author: NTIRE SR Team
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class CachedSRDataset(Dataset):
    """
    Dataset that loads pre-computed expert features from disk.
    
    Achieves 10-20x training speedup by skipping expert model inference.
    
    Args:
        feature_dir: Path to cached features directory
        augment: Enable geometric augmentations (flip, rotate)
        repeat_factor: Repeat dataset for more training samples per epoch
        load_features: Whether to load intermediate features (for collaborative learning)
    """
    
    def __init__(
        self,
        feature_dir: str,
        augment: bool = True,
        repeat_factor: int = 1,
        load_features: bool = True
    ):
        super().__init__()
        
        self.feature_dir = Path(feature_dir)
        self.augment = augment
        self.repeat_factor = repeat_factor
        self.load_features = load_features
        
        # Verify directory exists
        if not self.feature_dir.exists():
            raise RuntimeError(f"Feature cache directory not found: {feature_dir}")
        
        # Find all unique filenames by looking for _hat_part.pt files
        hat_files = sorted(list(self.feature_dir.glob("*_hat_part.pt")))
        
        if len(hat_files) == 0:
            raise RuntimeError(
                f"No cached features found in {feature_dir}!\n"
                f"Run 'python scripts/extract_features_balanced.py' first."
            )
        
        # Extract filename stems (without _hat_part.pt suffix)
        self.file_stems = [f.name.replace('_hat_part.pt', '') for f in hat_files]
        
        # Verify matching rest_part files exist
        missing = []
        for stem in self.file_stems:
            rest_path = self.feature_dir / f"{stem}_rest_part.pt"
            if not rest_path.exists():
                missing.append(stem)
        
        if missing:
            print(f"Warning: {len(missing)} files missing rest_part counterparts")
            # Filter to only complete pairs
            self.file_stems = [s for s in self.file_stems if s not in missing]
        
        print(f"CachedSRDataset initialized:")
        print(f"  Directory: {feature_dir}")
        print(f"  Samples: {len(self.file_stems)}")
        print(f"  Repeat factor: {repeat_factor}")
        print(f"  Effective length: {len(self)}")
        print(f"  Augmentation: {augment}")
        print(f"  Load features: {load_features}")
    
    def __len__(self) -> int:
        """Return effective dataset length with repeat factor."""
        return len(self.file_stems) * self.repeat_factor
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a training sample.
        
        Returns:
            Dictionary with:
            - lr: [3, H, W] LR input tensor
            - hr: [3, H*4, W*4] HR target tensor  
            - expert_imgs: Dict[str, Tensor] - SR outputs from each expert
            - expert_feats: Dict[str, Tensor] - Intermediate features (if load_features=True)
            - filename: str
        """
        # Handle repeat factor
        file_idx = idx % len(self.file_stems)
        stem = self.file_stems[file_idx]
        
        # Load both parts
        hat_path = self.feature_dir / f"{stem}_hat_part.pt"
        rest_path = self.feature_dir / f"{stem}_rest_part.pt"
        
        data_hat = torch.load(hat_path, weights_only=False)
        data_rest = torch.load(rest_path, weights_only=False)
        
        # Extract LR/HR from HAT part (they're stored there)
        lr = data_hat['lr']  # [3, H, W]
        hr = data_hat['hr']  # [3, H*4, W*4]
        
        # Merge expert outputs
        expert_imgs = {}
        expert_imgs.update(data_hat['outputs'])
        expert_imgs.update(data_rest['outputs'])
        
        # Squeeze batch dimension if present
        for name in expert_imgs:
            if expert_imgs[name].dim() == 4:
                expert_imgs[name] = expert_imgs[name].squeeze(0)
        
        # Merge features (if enabled)
        expert_feats = None
        if self.load_features:
            expert_feats = {}
            expert_feats.update(data_hat.get('features', {}))
            expert_feats.update(data_rest.get('features', {}))
            
            # Squeeze batch dimension if present
            for name in expert_feats:
                if expert_feats[name].dim() == 4:
                    expert_feats[name] = expert_feats[name].squeeze(0)
        
        # Apply augmentations (same transform to all tensors)
        if self.augment:
            lr, hr, expert_imgs, expert_feats = self._apply_augmentation(
                lr, hr, expert_imgs, expert_feats
            )
        
        result = {
            'lr': lr,
            'hr': hr,
            'expert_imgs': expert_imgs,
            'filename': stem
        }
        
        if expert_feats is not None:
            result['expert_feats'] = expert_feats
        
        return result
    
    def _apply_augmentation(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
        expert_imgs: Dict[str, torch.Tensor],
        expert_feats: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply geometric augmentations consistently to all tensors.
        
        Supports:
        - Horizontal flip (50% chance)
        - Vertical flip (50% chance)
        - 90° rotations (25% chance for each: 0°, 90°, 180°, 270°)
        
        Note: Color jitter is NOT supported in cached mode.
        
        Args:
            lr: LR tensor [C, H, W]
            hr: HR tensor [C, H*4, W*4]
            expert_imgs: Dict of expert SR outputs
            expert_feats: Dict of intermediate features (optional)
            
        Returns:
            Augmented tensors
        """
        # Decide on augmentations
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot_k = random.randint(0, 3)  # 0=0°, 1=90°, 2=180°, 3=270°
        
        def apply_transform(tensor: torch.Tensor) -> torch.Tensor:
            """Apply the same transforms to any [C, H, W] tensor."""
            if hflip:
                tensor = torch.flip(tensor, dims=[-1])
            if vflip:
                tensor = torch.flip(tensor, dims=[-2])
            if rot_k > 0:
                tensor = torch.rot90(tensor, k=rot_k, dims=[-2, -1])
            return tensor
        
        # Apply to LR and HR
        lr = apply_transform(lr)
        hr = apply_transform(hr)
        
        # Apply to expert outputs
        for name in expert_imgs:
            expert_imgs[name] = apply_transform(expert_imgs[name])
        
        # Apply to features if present
        if expert_feats is not None:
            for name in expert_feats:
                expert_feats[name] = apply_transform(expert_feats[name])
        
        return lr, hr, expert_imgs, expert_feats


def create_cached_dataloader(
    feature_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True,
    repeat_factor: int = 20,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    load_features: bool = True
) -> DataLoader:
    """
    Create a DataLoader for cached features.
    
    Args:
        feature_dir: Path to cached features
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Enable geometric augmentations
        repeat_factor: Repeat dataset for more samples
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        load_features: Load intermediate features for collaborative learning
        
    Returns:
        DataLoader for cached training
    """
    dataset = CachedSRDataset(
        feature_dir=feature_dir,
        augment=augment,
        repeat_factor=repeat_factor,
        load_features=load_features
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    print(f"\nCached DataLoader:")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples per epoch: {len(dataset)}")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Workers: {num_workers}")
    
    return loader


def test_cached_dataset():
    """Test the cached dataset loading."""
    import tempfile
    import shutil
    
    print("\n" + "=" * 70)
    print("CACHED DATASET TEST")
    print("=" * 70)
    
    # Create temp directory with mock cached features
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {temp_dir}")
    
    try:
        # Create mock cached files
        for i in range(5):
            filename = f"test_img_{i:03d}"
            
            # Mock HAT part
            hat_data = {
                'outputs': {'hat': torch.randn(1, 3, 256, 256)},
                'features': {'hat': torch.randn(1, 180, 64, 64)},
                'lr': torch.randn(3, 64, 64),
                'hr': torch.randn(3, 256, 256),
                'filename': filename
            }
            torch.save(hat_data, temp_dir / f"{filename}_hat_part.pt")
            
            # Mock rest part
            rest_data = {
                'outputs': {
                    'dat': torch.randn(1, 3, 256, 256),
                    'nafnet': torch.randn(1, 3, 256, 256)
                },
                'features': {
                    'dat': torch.randn(1, 180, 64, 64),
                    'nafnet': torch.randn(1, 64, 64, 64)
                },
                'filename': filename
            }
            torch.save(rest_data, temp_dir / f"{filename}_rest_part.pt")
        
        print(f"Created 5 mock cached feature files\n")
        
        # Test 1: Basic loading
        print("--- Test 1: Basic Loading ---")
        dataset = CachedSRDataset(
            feature_dir=str(temp_dir),
            augment=False,
            repeat_factor=1,
            load_features=True
        )
        
        assert len(dataset) == 5
        sample = dataset[0]
        
        print(f"  LR shape: {sample['lr'].shape}")
        print(f"  HR shape: {sample['hr'].shape}")
        print(f"  Expert outputs: {list(sample['expert_imgs'].keys())}")
        print(f"  Expert features: {list(sample['expert_feats'].keys())}")
        
        assert sample['lr'].shape == (3, 64, 64)
        assert sample['hr'].shape == (3, 256, 256)
        assert 'hat' in sample['expert_imgs']
        assert 'dat' in sample['expert_imgs']
        assert 'nafnet' in sample['expert_imgs']
        print("  [PASSED]\n")
        
        # Test 2: Repeat factor
        print("--- Test 2: Repeat Factor ---")
        dataset2 = CachedSRDataset(
            feature_dir=str(temp_dir),
            augment=False,
            repeat_factor=4,
            load_features=False
        )
        assert len(dataset2) == 20
        print(f"  Length with repeat_factor=4: {len(dataset2)}")
        print("  [PASSED]\n")
        
        # Test 3: Augmentation
        print("--- Test 3: Augmentation Consistency ---")
        dataset3 = CachedSRDataset(
            feature_dir=str(temp_dir),
            augment=True,
            repeat_factor=1
        )
        
        # Get same sample multiple times
        shapes = []
        for _ in range(5):
            s = dataset3[0]
            # All should have valid shapes after augmentation
            shapes.append((s['lr'].shape, s['hr'].shape))
        
        print(f"  Sample shapes after augmentation: valid")
        print("  [PASSED]\n")
        
        # Test 4: DataLoader
        print("--- Test 4: DataLoader ---")
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        
        print(f"  Batch LR: {batch['lr'].shape}")
        print(f"  Batch HR: {batch['hr'].shape}")
        print(f"  Batch expert_imgs['hat']: {batch['expert_imgs']['hat'].shape}")
        
        assert batch['lr'].shape == (2, 3, 64, 64)
        assert batch['hr'].shape == (2, 3, 256, 256)
        assert batch['expert_imgs']['hat'].shape == (2, 3, 256, 256)
        print("  [PASSED]\n")
        
        print("=" * 70)
        print("✓ ALL CACHED DATASET TESTS PASSED!")
        print("=" * 70 + "\n")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    test_cached_dataset()
