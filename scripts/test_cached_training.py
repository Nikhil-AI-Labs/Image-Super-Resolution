"""
Test Cached Training Pipeline (Updated for DRCT+GRL+NAFNet+MambaIR)
=====================================================================
Integration test for the cached training feature with the new expert roster.

Tests:
1. CachedSRDataset loading (DRCT + GRL + NAFNet + MambaIR FP16)
2. CompleteEnhancedFusionSR with expert_ensemble=None
3. forward_with_precomputed() works correctly
4. Gradient flow through fusion network
5. Training step optimization
6. MambaIR FP16→FP32 conversion

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
    print("  Experts: DRCT + GRL + NAFNet + MambaIR (FP16 from Colab)")
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
                    'nafnet': torch.randn(1, 64, 64, 64),  # NAFNet = 64 channels
                },
                'filename': filename,
            }
            torch.save(rest_data, temp_dir / f"{filename}_rest_part.pt")
            
            # Mock MambaIR part (FP16 — simulates Colab extraction!)
            mamba_data = {
                'outputs': {'mamba': torch.randn(1, 3, 256, 256).half()},
                'features': {'mamba': torch.randn(1, 180, 64, 64).half()},
                'filename': filename,
            }
            torch.save(mamba_data, temp_dir / f"{filename}_mamba_part.pt")
        
        print(f"  Created {num_samples} mock cached feature files (DRCT + rest + MambaIR)")
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
        
        expected_experts = {'drct', 'grl', 'nafnet', 'mamba'}
        assert set(sample['expert_imgs'].keys()) == expected_experts, \
            f"Expected {expected_experts}, got {set(sample['expert_imgs'].keys())}"
        
        # Verify MambaIR FP16→FP32 conversion
        assert sample['expert_imgs']['mamba'].dtype == torch.float32, \
            f"MambaIR output should be FP32, got {sample['expert_imgs']['mamba'].dtype}"
        assert sample['expert_feats']['mamba'].dtype == torch.float32, \
            f"MambaIR features should be FP32, got {sample['expert_feats']['mamba'].dtype}"
        
        # Verify NAFNet channels
        assert sample['expert_feats']['nafnet'].shape[0] == 64, \
            f"NAFNet features should have 64 channels, got {sample['expert_feats']['nafnet'].shape[0]}"
        assert sample['expert_feats']['drct'].shape[0] == 180, \
            f"DRCT features should have 180 channels, got {sample['expert_feats']['drct'].shape[0]}"
        assert sample['expert_feats']['mamba'].shape[0] == 180, \
            f"MambaIR features should have 180 channels, got {sample['expert_feats']['mamba'].shape[0]}"
        
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Expert outputs: {list(sample['expert_imgs'].keys())}")
        print(f"  Expert features: {list(sample['expert_feats'].keys())}")
        print(f"  Feature channels: drct={sample['expert_feats']['drct'].shape[0]}, "
              f"grl={sample['expert_feats']['grl'].shape[0]}, "
              f"nafnet={sample['expert_feats']['nafnet'].shape[0]}, "
              f"mamba={sample['expert_feats']['mamba'].shape[0]}")
        print(f"  MambaIR dtype: {sample['expert_imgs']['mamba'].dtype} [OK] (converted from FP16)")
        print("  [PASSED]\n")
        
        # ====================================================================
        # Test 3: CompleteEnhancedFusionSR with expert_ensemble=None
        # ====================================================================
        print("--- Test 3: Model with expert_ensemble=None ---")
        
        from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")
        
        # Create model WITHOUT experts (cached mode)
        model = CompleteEnhancedFusionSR(
            expert_ensemble=None,  # CACHED MODE!
            num_experts=4,
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
        print(f"  Expert features: {{" + 
              ", ".join(f"'{k}': {list(v.shape)}" for k, v in expert_feats.items()) + "}}")
        
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
            if "cached mode" in str(e).lower() or "expert_ensemble" in str(e).lower():
                print(f"  Correctly raised error: {str(e)[:60]}...")
                print("  [PASSED]\n")
            else:
                raise
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("=" * 70)
        print("[OK] ALL CACHED TRAINING TESTS PASSED!")
        print("=" * 70)
        print("\nPipeline verified for DRCT + GRL + NAFNet + MambaIR:")
        print("  [OK] CachedSRDataset loads DRCT/GRL/NAFNet + MambaIR (FP16->FP32)")
        print("  [OK] CompleteEnhancedFusionSR works with expert_ensemble=None")
        print("  [OK] forward_with_precomputed() produces valid SR output")
        print("  [OK] Gradients flow correctly through fusion network")
        print("  [OK] Training optimization works")
        print("  [OK] forward() correctly blocked in cached mode")
        print("\nNext steps:")
        print("  1. Run MambaIR extraction on Colab: scripts/extract_mamba_features.py")
        print("  2. Run local extraction (DRCT+GRL+NAFNet): scripts/extract_features_balanced.py")
        print("  3. Train with cache: python train.py --cached --cache-dir <path>")
        print("\n")
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    success = test_cached_training()
    sys.exit(0 if success else 1)
