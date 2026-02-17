"""
Phase 5 Pre-flight Test
========================
Verifies Phase 5 training configuration before launch.

Usage:
    python scripts/test_phase5_preflight.py
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config():
    """Test Phase 5 configuration."""
    import yaml
    
    config_path = project_root / 'configs' / 'train_config.yaml'
    assert config_path.exists(), f"Config not found: {config_path}"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Single GPU
    assert config['hardware']['gpu_ids'] == [0], \
        f"Expected gpu_ids=[0], got {config['hardware']['gpu_ids']}"
    
    # Batch size (P5000 safe)
    assert config['training']['batch_size'] <= 32, \
        f"Batch size {config['training']['batch_size']} too large for P5000"
    
    # Learning rate
    assert config['training']['optimizer']['lr'] <= 2e-4, \
        f"LR too high: {config['training']['optimizer']['lr']}"
    
    # All 4 phases enabled
    fusion = config['model']['fusion']
    assert fusion.get('enable_hierarchical', False), "Phase 1 not enabled"
    assert fusion.get('enable_multi_domain_freq', False), "Phase 2 not enabled"
    assert fusion.get('enable_lka', False), "Phase 3 not enabled"
    assert fusion.get('enable_edge_enhance', False), "Phase 4 not enabled"
    
    # Loss stages
    stages = config['loss']['stages']
    assert len(stages) == 3, f"Expected 3 loss stages, got {len(stages)}"
    assert stages[0]['epochs'][0] == 0
    assert stages[-1]['epochs'][1] == 200
    
    # EMA
    assert config['training']['ema']['enabled'] is True
    
    print("  [PASS] Configuration validated")
    return config


def test_multi_stage_scheduler():
    """Test the multi-stage loss scheduler."""
    from src.training.multi_stage_scheduler import MultiStageLossScheduler
    
    stages = [
        {'epochs': [0, 80], 'stage_name': 'foundation',
         'weights': {'l1': 1.0, 'swt': 0.0}},
        {'epochs': [80, 150], 'stage_name': 'frequency',
         'weights': {'l1': 0.75, 'swt': 0.20}},
        {'epochs': [150, 200], 'stage_name': 'detail',
         'weights': {'l1': 0.60, 'swt': 0.25}},
    ]
    
    scheduler = MultiStageLossScheduler(stages)
    
    # Test stage transitions
    scheduler.step(0)
    assert scheduler.get_stage_info()['stage_name'] == 'foundation'
    
    changed = scheduler.step(80)
    assert changed, "Expected stage change at epoch 80"
    assert scheduler.get_stage_info()['stage_name'] == 'frequency'
    
    changed = scheduler.step(150)
    assert changed, "Expected stage change at epoch 150"
    assert scheduler.get_stage_info()['stage_name'] == 'detail'
    
    # Test no change mid-stage (forward within same stage)
    changed = scheduler.step(160)
    assert not changed, "No change mid-stage"
    
    print("  [PASS] MultiStageLossScheduler working correctly")


def test_model_creation():
    """Test model creation with Phase 5 config."""
    import torch
    import yaml
    
    config_path = project_root / 'configs' / 'train_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR
    
    fusion_config = config['model']['fusion']
    improvements = fusion_config.get('improvements', {})
    
    model = CompleteEnhancedFusionSR(
        expert_ensemble=None,
        num_experts=3,
        upscale=4,
        fusion_dim=fusion_config.get('fusion_dim', 64),
        num_heads=fusion_config.get('num_heads', 4),
        refine_depth=fusion_config.get('refine_depth', 4),
        refine_channels=fusion_config.get('refine_channels', 64),
        enable_hierarchical=fusion_config.get('enable_hierarchical', True),
        enable_multi_domain_freq=fusion_config.get('enable_multi_domain_freq', False),
        enable_lka=fusion_config.get('enable_lka', False),
        enable_edge_enhance=fusion_config.get('enable_edge_enhance', False),
        enable_dynamic_selection=improvements.get('dynamic_expert_selection', True),
        enable_cross_band_attn=improvements.get('cross_band_attention', True),
        enable_adaptive_bands=improvements.get('adaptive_frequency_bands', True),
        enable_multi_resolution=improvements.get('multi_resolution_fusion', True),
        enable_collaborative=improvements.get('collaborative_learning', True),
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    
    # Verify parameter count is in expected range (~1M)
    assert 900_000 < trainable < 1_500_000, \
        f"Unexpected param count: {trainable:,}"
    
    # Test forward pass
    x = torch.randn(1, 3, 64, 64)
    experts = {
        'hat': torch.randn(1, 3, 256, 256),
        'dat': torch.randn(1, 3, 256, 256),
        'nafnet': torch.randn(1, 3, 256, 256)
    }
    
    with torch.no_grad():
        out = model.forward_with_precomputed(x, experts)
    
    assert out.shape == (1, 3, 256, 256), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    
    print(f"  [PASS] Model created and forward pass OK ({trainable:,} params)")


def test_vram_estimate():
    """Estimate VRAM usage for Phase 5."""
    import torch
    
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA - skipping VRAM estimate")
        return
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Rough VRAM estimate for batch_size=24, fusion_only (cached mode):
    # Model params: ~1M * 4 bytes = 4 MB
    # Gradients + optimizer: ~4 * 4 MB = 16 MB
    # Input (24 x 3 x 64 x 64): ~1.1 MB
    # Experts (24 x 3 x 3 x 256 x 256): ~141 MB
    # Activations: ~500 MB (rough estimate)
    # Total: ~700 MB
    
    estimated_gb = 0.7
    safe = gpu_mem > estimated_gb * 2
    
    print(f"  Estimated VRAM: ~{estimated_gb:.1f} GB (cached mode)")
    print(f"  Available: {gpu_mem:.1f} GB")
    print(f"  Safety margin: {gpu_mem / estimated_gb:.1f}x")
    
    if safe:
        print(f"  [PASS] VRAM estimate safe for batch_size=24")
    else:
        print(f"  [WARN] Tight VRAM - consider reducing batch_size")


if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 5 PRE-FLIGHT TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Configuration", test_config),
        ("Multi-Stage Scheduler", test_multi_stage_scheduler),
        ("Model Creation", test_model_creation),
        ("VRAM Estimate", test_vram_estimate),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    
    if failed == 0:
        print("\n  ALL PRE-FLIGHT TESTS PASSED! Ready for Phase 5 training.")
    else:
        print(f"\n  {failed} test(s) FAILED. Fix issues before launching training.")
    
    sys.exit(0 if failed == 0 else 1)
