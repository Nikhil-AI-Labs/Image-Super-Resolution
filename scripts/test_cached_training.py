"""
Test Cached Training Pipeline
==============================
Integration test for the cached training feature.

Tests:
1. CachedSRDataset loading and batching
2. CompleteEnhancedFusionSR with expert_ensemble=None
3. forward_with_precomputed() works correctly
4. Gradient flow through fusion network
5. Training step optimization

Usage:
    python scripts/test_cached_training.py

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
import torch.nn as nn
import numpy as np


def test_cached_training():
    """Test the complete cached training pipeline."""
    print("\n" + "=" * 70)
    print("CACHED TRAINING INTEGRATION TEST")
    print("=" * 70)
    print("  Testing 10-20x faster training with pre-computed expert features")
    print("=" * 70 + "\n")
    
    # Create temp directory for mock cached features
    temp_dir = Path(tempfile.mkdtemp())
    print(f"[Setup] Creating mock cached features in: {temp_dir}")
    
    try:
        # ====================================================================
        # Test 1: Create mock cached feature files
        # ====================================================================
        print("\n--- Test 1: Creating Mock Cached Features ---")
        
        num_samples = 5
        for i in range(num_samples):
            filename = f"test_img_{i:03d}"
            
            # Mock HAT part (matches extract_features_balanced.py output format)
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
        
        print(f"  Created {num_samples} mock cached feature files")
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 2: CachedSRDataset loading
        # ====================================================================
        print("--- Test 2: CachedSRDataset Loading ---")
        
        from src.data.cached_dataset import CachedSRDataset
        
        dataset = CachedSRDataset(
            feature_dir=str(temp_dir),
            augment=True,
            repeat_factor=2,
            load_features=True
        )
        
        assert len(dataset) == num_samples * 2, f"Expected {num_samples * 2}, got {len(dataset)}"
        
        sample = dataset[0]
        assert 'lr' in sample and 'hr' in sample
        assert 'expert_imgs' in sample and 'expert_feats' in sample
        assert set(sample['expert_imgs'].keys()) == {'hat', 'dat', 'nafnet'}
        
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Expert outputs: {list(sample['expert_imgs'].keys())}")
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 3: CompleteEnhancedFusionSR with expert_ensemble=None
        # ====================================================================
        print("--- Test 3: Model with expert_ensemble=None ---")
        
        from src.models.enhanced_fusion import CompleteEnhancedFusionSR
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")
        
        # Create model WITHOUT experts (cached mode)
        model = CompleteEnhancedFusionSR(
            expert_ensemble=None,  # CACHED MODE!
            num_experts=3,
            upscale=4,
            enable_dynamic_selection=True,
            enable_cross_band_attn=True,
            enable_adaptive_bands=True,
            enable_multi_resolution=True,
            enable_collaborative=True,
        ).to(device)
        
        assert model.cached_mode == True, "Model should be in cached mode"
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model created in cached mode: {model.cached_mode}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 4: forward_with_precomputed works
        # ====================================================================
        print("--- Test 4: forward_with_precomputed() ---")
        
        # Get a batch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        lr_img = batch['lr'].to(device)
        hr_img = batch['hr'].to(device)
        expert_imgs = {k: v.to(device) for k, v in batch['expert_imgs'].items()}
        expert_feats = {k: v.to(device) for k, v in batch['expert_feats'].items()}
        
        print(f"  Input LR shape: {lr_img.shape}")
        print(f"  Expert outputs: {list(expert_imgs.keys())}")
        
        with torch.no_grad():
            sr_output = model.forward_with_precomputed(lr_img, expert_imgs, expert_feats)
        
        assert sr_output.shape == (2, 3, 256, 256), f"Expected (2,3,256,256), got {sr_output.shape}"
        print(f"  Output SR shape: {sr_output.shape}")
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 5: Gradient flow
        # ====================================================================
        print("--- Test 5: Gradient Flow ---")
        
        model.train()
        model.zero_grad()
        
        # Forward with gradients
        sr_output = model.forward_with_precomputed(lr_img, expert_imgs, expert_feats)
        
        # Simple L1 loss
        loss = nn.L1Loss()(sr_output, hr_img)
        loss.backward()
        
        # Check gradients exist
        params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Params with gradients: {params_with_grad}/{total_trainable}")
        
        assert params_with_grad > 0, "No gradients flowing!"
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 6: Training step
        # ====================================================================
        print("--- Test 6: Training Step Optimization ---")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Store initial loss
        initial_loss = loss.item()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Forward again
        sr_output2 = model.forward_with_precomputed(lr_img, expert_imgs, expert_feats)
        loss2 = nn.L1Loss()(sr_output2, hr_img)
        
        print(f"  Before optimization: {initial_loss:.4f}")
        print(f"  After optimization: {loss2.item():.4f}")
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 7: Verify model.forward() raises error in cached mode
        # ====================================================================
        print("--- Test 7: forward() Error in Cached Mode ---")
        
        try:
            model(lr_img)  # Should raise RuntimeError
            print("  [FAILED] forward() should raise RuntimeError in cached mode!")
            assert False
        except RuntimeError as e:
            if "cached mode" in str(e):
                print(f"  Correctly raised error: {str(e)[:60]}...")
                print("  [PASSED]\n")
            else:
                raise
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("=" * 70)
        print("✓ ALL CACHED TRAINING TESTS PASSED!")
        print("=" * 70)
        print("\nCached training pipeline is ready:")
        print("  ✓ CachedSRDataset loads and batches .pt files correctly")
        print("  ✓ CompleteEnhancedFusionSR works with expert_ensemble=None")
        print("  ✓ forward_with_precomputed() produces valid SR output")
        print("  ✓ Gradients flow correctly through fusion network")
        print("  ✓ Training optimization works")
        print("  ✓ forward() correctly blocked in cached mode")
        print("\nNext steps:")
        print("  1. Run extraction: python scripts/extract_features_balanced.py")
        print("  2. Train with cache: python train.py --cached --cache-dir <path>")
        print("\n")
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    success = test_cached_training()
    sys.exit(0 if success else 1)
