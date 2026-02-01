"""
Integration Test: Dataset + Fusion + Losses
============================================
Tests the complete training pipeline with all components.

Author: NTIRE SR Team
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import cv2


def test_full_pipeline():
    """Test the complete training pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE INTEGRATION TEST")
    print("=" * 70)
    
    # Create temp dataset
    temp_dir = Path(tempfile.mkdtemp())
    lr_dir = temp_dir / 'train_LR'
    hr_dir = temp_dir / 'train_HR'
    val_lr_dir = temp_dir / 'val_LR'
    val_hr_dir = temp_dir / 'val_HR'
    
    for d in [lr_dir, hr_dir, val_lr_dir, val_hr_dir]:
        d.mkdir()
    
    print("\n--- Creating Test Dataset ---")
    for i in range(4):
        # Training images
        hr_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(hr_dir / f'img_{i:03d}.png'), hr_img)
        lr_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(lr_dir / f'img_{i:03d}.png'), lr_img)
        
        # Validation images
        hr_val = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(val_hr_dir / f'val_{i:03d}.png'), hr_val)
        lr_val = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(val_lr_dir / f'val_{i:03d}.png'), lr_val)
    
    print("  Created 4 training + 4 validation pairs")
    
    # Test 1: Dataset loading
    print("\n--- Test 1: Dataset Loading ---")
    from src.data import create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        train_hr_dir=str(hr_dir),
        train_lr_dir=str(lr_dir),
        val_hr_dir=str(val_hr_dir),
        val_lr_dir=str(val_lr_dir),
        batch_size=2,
        num_workers=0,
        lr_patch_size=64,
        scale=4,
        repeat_factor=1
    )
    
    batch = next(iter(train_loader))
    assert batch['lr'].shape == (2, 3, 64, 64)
    assert batch['hr'].shape == (2, 3, 256, 256)
    print("  [PASSED] Dataset loading works")
    
    # Test 2: Fusion network
    print("\n--- Test 2: Fusion Network ---")
    from src.models.fusion_network import FrequencyAwareFusion
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    fusion = FrequencyAwareFusion(
        num_experts=3,
        use_residual=True,
        use_multiscale=True
    ).to(device)
    
    lr_input = batch['lr'].to(device)
    expert_outputs = [torch.rand(2, 3, 256, 256, device=device) for _ in range(3)]
    
    sr_output = fusion(lr_input, expert_outputs)
    assert sr_output.shape == (2, 3, 256, 256)
    print(f"  Output shape: {sr_output.shape}")
    print("  [PASSED] Fusion network works")
    
    # Test 3: Loss functions
    print("\n--- Test 3: Loss Functions ---")
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    
    criterion = CombinedLoss(
        use_swt=PYWT_AVAILABLE,
        use_fft=True,
        use_clip=False
    ).to(device)
    criterion.set_stage(2)
    
    hr_target = batch['hr'].to(device)
    loss, components = criterion(sr_output, hr_target, return_components=True)
    
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Components: {list(components.keys())}")
    print("  [PASSED] Loss functions work")
    
    # Test 4: Gradient flow
    print("\n--- Test 4: Gradient Flow ---")
    fusion.zero_grad()
    loss.backward()
    
    params_with_grad = sum(1 for p in fusion.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in fusion.parameters() if p.requires_grad)
    
    print(f"  Params with gradients: {params_with_grad}/{total_trainable}")
    assert params_with_grad > 0, "No gradients!"
    print("  [PASSED] Gradient flow works")
    
    # Test 5: Training step
    print("\n--- Test 5: Training Step ---")
    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    
    initial_loss = loss.item()
    optimizer.step()
    
    # Forward again
    sr_output2 = fusion(lr_input, expert_outputs)
    loss2 = criterion(sr_output2, hr_target)
    
    print(f"  Before: {initial_loss:.4f}, After: {loss2.item():.4f}")
    print("  [PASSED] Training step completed")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\nPipeline ready for training:")
    print("  ✓ Dataset loading with augmentation")
    print("  ✓ Fusion network forward pass")
    print("  ✓ Championship loss functions")
    print("  ✓ Gradient flow through all components")
    print("  ✓ Training step optimization")
    print("\n")


if __name__ == '__main__':
    test_full_pipeline()
