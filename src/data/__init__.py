"""
Data Module for Super-Resolution (Optimized for Pre-Generated LR-HR Pairs)
===========================================================================
Efficient dataset loading for NTIRE 2025 super-resolution training.

Available:
- SRDataset: General SR dataset with LR-HR pairs
- DF2KDataset: DF2K with automatic directory detection
- ValidationDataset: Full-image validation without patching
- CachedSRDataset: Pre-computed expert features (10-20x faster!)
- create_dataloaders: Easy loader creation
- create_df2k_dataloaders: Convenience for DF2K
- create_cached_dataloader: Cached feature DataLoader
- SRTrainAugmentation: Complete augmentation pipeline

Performance: 10-15% faster training vs on-the-fly LR generation!
Cached Mode: 10-20x faster training with pre-computed expert features!
"""

from .dataset import (
    SRDataset,
    DF2KDataset,
    ValidationDataset,
    create_dataloaders,
    create_df2k_dataloaders,
)

from .cached_dataset import (
    CachedSRDataset,
    create_cached_dataloader,
)

from .augmentations import (
    SRTrainAugmentation,
    PairedRandomCrop,
    PairedRandomFlip,
    PairedRandomRotation,
    ColorJitter,
    GaussianBlur,
    CutBlur,
)

from .frequency_decomposition import FrequencyDecomposition

__all__ = [
    # Datasets
    'SRDataset',
    'DF2KDataset',
    'ValidationDataset',
    'CachedSRDataset',
    'create_dataloaders',
    'create_df2k_dataloaders',
    'create_cached_dataloader',
    
    # Augmentations
    'SRTrainAugmentation',
    'PairedRandomCrop',
    'PairedRandomFlip',
    'PairedRandomRotation',
    'ColorJitter',
    'GaussianBlur',
    'CutBlur',
    
    # Frequency analysis
    'FrequencyDecomposition',
]

