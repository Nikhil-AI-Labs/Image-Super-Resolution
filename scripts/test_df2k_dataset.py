"""
Test loading your actual DF2K dataset with pre-generated LR-HR pairs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_df2k_dataloaders, SRDataset
import torch


def test_your_df2k_dataset():
    """Test loading your actual DF2K dataset."""
    print("\n" + "=" * 70)
    print("TESTING YOUR DF2K DATASET")
    print("=" * 70)
    
    # Update this path to your actual DF2K location
    df2k_root = "data/DF2K"  # or wherever your dataset is
    
    print(f"\nDataset location: {df2k_root}")
    
    # Check if dataset exists
    root_path = Path(df2k_root)
    if not root_path.exists():
        print(f"\n⚠ Dataset not found at: {df2k_root}")
        print("\nPlease update 'df2k_root' path in this script.")
        print("Expected structure:")
        print("  data/DF2K/")
        print("  ├── train_LR/")
        print("  ├── train_HR/")
        print("  ├── val_LR/")
        print("  └── val_HR/")
        return False
    
    # Create dataloaders
    try:
        train_loader, val_loader = create_df2k_dataloaders(
            root_dir=df2k_root,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            lr_patch_size=64,
            scale=4,
            repeat_factor=1  # Set to 1 for testing
        )
        
        # Test training batch
        print("\n--- Testing Training Batch ---")
        train_batch = next(iter(train_loader))
        print(f"  LR batch: {train_batch['lr'].shape}")
        print(f"  HR batch: {train_batch['hr'].shape}")
        print(f"  Filenames: {train_batch['filename'][:2]}")
        
        # Verify value ranges
        assert torch.all((train_batch['lr'] >= 0) & (train_batch['lr'] <= 1))
        assert torch.all((train_batch['hr'] >= 0) & (train_batch['hr'] <= 1))
        print("  ✓ Value ranges correct [0, 1]")
        
        # Verify dimensions
        B, C, H, W = train_batch['lr'].shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == 64, f"Expected LR height 64, got {H}"
        assert W == 64, f"Expected LR width 64, got {W}"
        print("  ✓ LR dimensions correct (64x64)")
        
        B, C, H, W = train_batch['hr'].shape
        assert H == 256, f"Expected HR height 256, got {H}"
        assert W == 256, f"Expected HR width 256, got {W}"
        print("  ✓ HR dimensions correct (256x256)")
        
        # Test validation batch
        print("\n--- Testing Validation Batch ---")
        val_batch = next(iter(val_loader))
        print(f"  LR batch: {val_batch['lr'].shape}")
        print(f"  HR batch: {val_batch['hr'].shape}")
        print(f"  Filename: {val_batch['filename']}")
        
        print("\n" + "=" * 70)
        print("✓ YOUR DF2K DATASET IS READY FOR TRAINING!")
        print("=" * 70 + "\n")
        
        # Print usage example
        print("Usage example:")
        print("```python")
        print("from src.data import create_df2k_dataloaders")
        print("")
        print("train_loader, val_loader = create_df2k_dataloaders(")
        print(f"    root_dir='{df2k_root}',")
        print("    batch_size=16,")
        print("    num_workers=4,")
        print("    lr_patch_size=64,")
        print("    scale=4,")
        print("    repeat_factor=20")
        print(")")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Check your dataset structure:")
        print("   data/DF2K/")
        print("   ├── train_LR/")
        print("   ├── train_HR/")
        print("   ├── val_LR/")
        print("   └── val_HR/")
        print("\n2. Update 'df2k_root' path in the script")
        print("3. Verify images exist in directories")
        return False


if __name__ == '__main__':
    test_your_df2k_dataset()
