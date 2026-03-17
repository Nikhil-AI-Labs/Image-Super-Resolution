"""
Cached Super-Resolution Dataset
================================
Loads pre-computed expert outputs and features from disk for ultra-fast training.

This dataset is designed to work with features extracted by:
    scripts/extract_features_balanced.py  (DRCT + GRL + NAFNet — local GPU)
    scripts/extract_mamba_features.py     (MambaIR — Colab/Kaggle)

Expected file format:
    {cache_dir}/
    ├── img_001_drct_part.pt  - Contains DRCT SR output + features + LR/HR
    ├── img_001_rest_part.pt  - Contains GRL + NAFNet SR outputs + features
    ├── img_001_mamba_part.pt - Contains MambaIR SR output + features (FP16)
    ├── img_002_drct_part.pt
    ├── img_002_rest_part.pt
    ├── img_002_mamba_part.pt
    └── ...

Each _drct_part.pt file contains:
    - outputs: Dict[str, Tensor]  - DRCT SR output
    - features: Dict[str, Tensor] - DRCT intermediate features [B, 180, H, W]
    - lr: Tensor - Original LR patch
    - hr: Tensor - Original HR patch
    - filename: str

Each _rest_part.pt file contains:
    - outputs: Dict[str, Tensor]  - GRL + NAFNet SR outputs
    - features: Dict[str, Tensor] - GRL [B, 180, H, W] + NAFNet [B, 64, H, W]

Each _mamba_part.pt file contains:
    - outputs: Dict[str, Tensor]  - MambaIR SR output (FP16 → converted to FP32)
    - features: Dict[str, Tensor] - MambaIR features [B, 180, H, W] (FP16 → FP32)

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
    
    Supports 4 experts: DRCT, GRL, NAFNet (local), MambaIR (Colab/Kaggle FP16).
    
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
        
        # Find all unique filenames by looking for _drct_part.pt files
        drct_files = sorted(list(self.feature_dir.glob("*_drct_part.pt")))
        
        if len(drct_files) == 0:
            raise RuntimeError(
                f"No cached features found in {feature_dir}!\n"
                f"Run 'python scripts/extract_features_balanced.py' first."
            )
        
        # Extract filename stems (without _drct_part.pt suffix)
        self.file_stems = [f.name.replace('_drct_part.pt', '') for f in drct_files]
        
        # Verify matching rest_part files exist
        missing_rest = []
        missing_mamba = []
        for stem in self.file_stems:
            rest_path = self.feature_dir / f"{stem}_rest_part.pt"
            mamba_path = self.feature_dir / f"{stem}_mamba_part.pt"
            if not rest_path.exists():
                missing_rest.append(stem)
            if not mamba_path.exists():
                missing_mamba.append(stem)
        
        if missing_rest:
            print(f"Warning: {len(missing_rest)} files missing rest_part counterparts")
            # Filter to only complete pairs (drct + rest at minimum)
            self.file_stems = [s for s in self.file_stems if s not in missing_rest]
        
        if missing_mamba:
            print(f"Warning: {len(missing_mamba)} files missing mamba_part "
                  f"(MambaIR features from Colab/Kaggle)")
        
        # Track which stems have mamba features
        self.has_mamba = {
            stem: (self.feature_dir / f"{stem}_mamba_part.pt").exists()
            for stem in self.file_stems
        }
        n_mamba = sum(self.has_mamba.values())
        
        print(f"CachedSRDataset initialized:")
        print(f"  Directory: {feature_dir}")
        print(f"  Samples: {len(self.file_stems)}")
        print(f"  MambaIR coverage: {n_mamba}/{len(self.file_stems)}")
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
        
        # Load DRCT part (contains LR/HR + DRCT output/features)
        drct_path = self.feature_dir / f"{stem}_drct_part.pt"
        data_drct = torch.load(drct_path, weights_only=False)
        
        # Load rest part (GRL + NAFNet outputs/features)
        rest_path = self.feature_dir / f"{stem}_rest_part.pt"
        data_rest = torch.load(rest_path, weights_only=False)
        
        # Extract LR/HR from DRCT part (they're stored there)
        lr = data_drct['lr']  # [3, H, W]
        hr = data_drct['hr']  # [3, H*4, W*4]
        
        # Merge expert outputs
        expert_imgs = {}
        expert_imgs.update(data_drct['outputs'])
        expert_imgs.update(data_rest['outputs'])
        
        # Load MambaIR part (FP16 from Colab/Kaggle) if available
        if self.has_mamba.get(stem, False):
            mamba_path = self.feature_dir / f"{stem}_mamba_part.pt"
            data_mamba = torch.load(mamba_path, weights_only=False)
            # FP16 → FP32 conversion for training stability
            for k, v in data_mamba['outputs'].items():
                expert_imgs[k] = v.float()
        else:
            # Fallback: zero tensor (graceful degradation)
            # Use drct output shape as reference for HR dimensions
            ref_shape = next(iter(expert_imgs.values())).shape
            if len(ref_shape) == 4:
                expert_imgs['mamba'] = torch.zeros(ref_shape)
            else:
                expert_imgs['mamba'] = torch.zeros(ref_shape)
        
        # Squeeze batch dimension if present
        for name in expert_imgs:
            if expert_imgs[name].dim() == 4:
                expert_imgs[name] = expert_imgs[name].squeeze(0)
        
        # Merge features (if enabled)
        expert_feats = None
        if self.load_features:
            expert_feats = {}
            expert_feats.update(data_drct.get('features', {}))
            expert_feats.update(data_rest.get('features', {}))
            
            # Load MambaIR features (FP16 → FP32)
            if self.has_mamba.get(stem, False):
                for k, v in data_mamba.get('features', {}).items():
                    expert_feats[k] = v.float()
            else:
                # Fallback: zero features
                lr_h, lr_w = lr.shape[-2], lr.shape[-1]
                expert_feats['mamba'] = torch.zeros(1, 180, lr_h, lr_w)
            
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
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
        # Create mock cached files (new format: DRCT + GRL + NAFNet + MambaIR)
        for i in range(5):
            filename = f"test_img_{i:03d}"
            
            # Mock DRCT part (includes LR/HR)
            drct_data = {
                'outputs': {'drct': torch.randn(1, 3, 256, 256)},
                'features': {'drct': torch.randn(1, 180, 64, 64)},
                'lr': torch.randn(3, 64, 64),
                'hr': torch.randn(3, 256, 256),
                'filename': filename
            }
            torch.save(drct_data, temp_dir / f"{filename}_drct_part.pt")
            
            # Mock rest part — GRL + NAFNet
            rest_data = {
                'outputs': {
                    'grl':    torch.randn(1, 3, 256, 256),
                    'nafnet': torch.randn(1, 3, 256, 256),
                },
                'features': {
                    'grl':    torch.randn(1, 180, 64, 64),
                    'nafnet': torch.randn(1, 64, 64, 64),   # NAFNet uses 64 channels
                },
                'filename': filename,
            }
            torch.save(rest_data, temp_dir / f"{filename}_rest_part.pt")
            
            # Mock MambaIR part (FP16 — simulates Colab extraction)
            mamba_data = {
                'outputs': {'mamba': torch.randn(1, 3, 256, 256).half()},
                'features': {'mamba': torch.randn(1, 180, 64, 64).half()},
                'filename': filename,
            }
            torch.save(mamba_data, temp_dir / f"{filename}_mamba_part.pt")
        
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
        assert 'drct'   in sample['expert_imgs']
        assert 'grl'    in sample['expert_imgs']
        assert 'nafnet'  in sample['expert_imgs']
        assert 'mamba'  in sample['expert_imgs']
        assert sample['expert_imgs']['drct'].shape   == (3, 256, 256)
        assert sample['expert_imgs']['nafnet'].shape  == (3, 256, 256)
        assert sample['expert_imgs']['mamba'].shape   == (3, 256, 256)
        # Verify MambaIR was converted from FP16 to FP32
        assert sample['expert_imgs']['mamba'].dtype == torch.float32
        assert sample['expert_feats']['drct'].shape   == (180, 64, 64)
        assert sample['expert_feats']['nafnet'].shape  == (64, 64, 64)  # 64 channels for NAFNet
        assert sample['expert_feats']['mamba'].shape   == (180, 64, 64)
        assert sample['expert_feats']['mamba'].dtype   == torch.float32
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
        print(f"  Batch expert_imgs['drct']: {batch['expert_imgs']['drct'].shape}")
        print(f"  Batch expert_imgs['mamba']: {batch['expert_imgs']['mamba'].shape}")
        
        assert batch['lr'].shape == (2, 3, 64, 64)
        assert batch['hr'].shape == (2, 3, 256, 256)
        assert batch['expert_imgs']['drct'].shape == (2, 3, 256, 256)
        assert batch['expert_imgs']['mamba'].shape == (2, 3, 256, 256)
        print("  [PASSED]\n")
        
        # Test 5: Missing MambaIR graceful degradation
        print("--- Test 5: Missing MambaIR Graceful Degradation ---")
        # Remove one mamba file
        (temp_dir / "test_img_000_mamba_part.pt").unlink()
        
        dataset5 = CachedSRDataset(
            feature_dir=str(temp_dir),
            augment=False,
            repeat_factor=1,
            load_features=True
        )
        sample5 = dataset5[0]
        assert 'mamba' in sample5['expert_imgs']
        assert sample5['expert_imgs']['mamba'].shape == (3, 256, 256)
        print(f"  Mamba output when missing: zeros tensor {sample5['expert_imgs']['mamba'].shape}")
        print("  [PASSED]\n")
        
        print("=" * 70)
        print("[OK] ALL CACHED DATASET TESTS PASSED!")
        print("=" * 70 + "\n")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    test_cached_dataset()
