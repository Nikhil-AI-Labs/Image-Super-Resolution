"""
Phase 2 Dry-Run Verification Script
====================================
Tests:
1. Standalone MultiDomainFrequencyDecomposition (9 bands, shapes, gradients)
2. Full model with Phase 1+2 enabled (param count, forward pass, gradient flow)
3. YAML config compatibility
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml


def test_standalone_frequency_module():
    """Test the standalone multi-domain frequency decomposition module."""
    from src.models.multi_domain_frequency import MultiDomainFrequencyDecomposition
    
    results = []
    results.append("=" * 60)
    results.append("TEST 1: Standalone MultiDomainFrequencyDecomposition")
    results.append("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results.append(f"Device: {device}")
    
    module = MultiDomainFrequencyDecomposition(
        block_size=8,
        in_channels=3,
        fft_mask_size=64,
        enable_fusion=True
    ).to(device)
    
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    results.append(f"Parameters: {params:,}")
    
    # Breakdown
    for name, child in module.named_children():
        p = sum(pp.numel() for pp in child.parameters() if pp.requires_grad)
        if p > 0:
            results.append(f"  {name}: {p:,}")
    
    # Forward pass
    x = torch.randn(2, 3, 64, 64, device=device)
    fused_bands, raw_bands = module(x, return_raw_bands=True)
    
    results.append(f"\nInput: {x.shape}")
    results.append(f"Raw bands: {len(raw_bands)}")
    
    band_names = module.band_names
    for i, (band, name) in enumerate(zip(raw_bands, band_names)):
        has_nan = torch.isnan(band).any().item()
        results.append(f"  {name}: {band.shape} range=[{band.min():.4f}, {band.max():.4f}] nan={has_nan}")
    
    results.append(f"Fused bands: {len(fused_bands)}")
    for i, band in enumerate(fused_bands):
        has_nan = torch.isnan(band).any().item()
        results.append(f"  guidance_{i}: {band.shape} range=[{band.min():.4f}, {band.max():.4f}] nan={has_nan}")
    
    # Gradient check
    module.zero_grad()
    loss = sum(b.mean() for b in fused_bands)
    loss.backward()
    
    grads = sum(1 for p in module.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in module.parameters() if p.requires_grad)
    results.append(f"\nGradient flow: {grads}/{total} params have gradients")
    
    # Assertions
    ok = True
    if len(raw_bands) != 9:
        results.append("FAIL: Expected 9 raw bands")
        ok = False
    if len(fused_bands) != 3:
        results.append("FAIL: Expected 3 fused bands")
        ok = False
    for band in raw_bands + fused_bands:
        if band.shape != x.shape:
            results.append(f"FAIL: Shape mismatch {band.shape} vs {x.shape}")
            ok = False
        if torch.isnan(band).any():
            results.append("FAIL: NaN detected")
            ok = False
    if grads == 0:
        results.append("FAIL: No gradients flowing")
        ok = False
    
    results.append(f"\nTest 1 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


def test_full_model_integration():
    """Test the full model with Phase 1 + Phase 2 enabled."""
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR
    
    results = []
    results.append("\n" + "=" * 60)
    results.append("TEST 2: Full Model (Phase 1 + Phase 2)")
    results.append("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CompleteEnhancedFusionSR(
        expert_ensemble=None,  # Cached mode
        num_experts=3,
        num_bands=3,
        block_size=8,
        upscale=4,
        # Phase 1
        fusion_dim=64,
        num_heads=4,
        refine_depth=4,
        refine_channels=64,
        enable_hierarchical=True,
        # Phase 2
        enable_multi_domain_freq=True,
        # Improvements
        enable_dynamic_selection=True,
        enable_cross_band_attn=True,
        enable_adaptive_bands=True,
        enable_multi_resolution=True,
        enable_collaborative=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results.append(f"Total trainable params: {total_params:,}")
    
    # Module breakdown
    results.append("\nModule breakdown:")
    for name, child in model.named_children():
        p = sum(pp.numel() for pp in child.parameters() if pp.requires_grad)
        if p > 0:
            results.append(f"  {name}: {p:,}")
    # Orphan params
    child_params = sum(pp.numel() for n, c in model.named_children() for pp in c.parameters() if pp.requires_grad)
    orphan = total_params - child_params
    if orphan > 0:
        results.append(f"  (direct params): {orphan:,}")
    
    # Forward pass
    B = 2
    expert_outputs = {
        'hat': torch.randn(B, 3, 256, 256, device=device),
        'dat': torch.randn(B, 3, 256, 256, device=device),
        'nafnet': torch.randn(B, 3, 256, 256, device=device),
    }
    lr_input = torch.randn(B, 3, 64, 64, device=device)
    
    expert_features = {
        'hat': torch.randn(B, 180, 64, 64, device=device),
        'dat': torch.randn(B, 180, 64, 64, device=device),
        'nafnet': torch.randn(B, 64, 64, 64, device=device),
    }
    
    # Use forward_with_precomputed (cached mode)
    output = model.forward_with_precomputed(
        lr_input=lr_input,
        expert_outputs=expert_outputs,
        expert_features=expert_features
    )
    
    results.append(f"\nForward pass:")
    results.append(f"  Input: {lr_input.shape}")
    results.append(f"  Output: {output.shape}")
    results.append(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    results.append(f"  Has NaN: {torch.isnan(output).any().item()}")
    results.append(f"  Has Inf: {torch.isinf(output).any().item()}")
    
    # Gradient check
    model.zero_grad()
    loss = output.mean()
    loss.backward()
    
    grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    results.append(f"\nGradient flow: {grads}/{total} params ({100*grads/total:.1f}%)")
    
    # Per-module gradient check
    results.append("Per-module gradient flow:")
    for name, child in model.named_children():
        t = sum(1 for p in child.parameters() if p.requires_grad)
        g = sum(1 for p in child.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        if t > 0:
            results.append(f"  {name}: {g}/{t}")
    
    # Assertions
    ok = True
    if output.shape != (B, 3, 256, 256):
        results.append(f"FAIL: Wrong output shape {output.shape}")
        ok = False
    if torch.isnan(output).any():
        results.append("FAIL: NaN in output")
        ok = False
    if torch.isinf(output).any():
        results.append("FAIL: Inf in output")
        ok = False
    if grads == 0:
        results.append("FAIL: No gradients")
        ok = False
    
    results.append(f"\nTest 2 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


def test_yaml_config():
    """Test that YAML config has Phase 2 parameters."""
    results = []
    results.append("\n" + "=" * 60)
    results.append("TEST 3: YAML Config Compatibility")
    results.append("=" * 60)
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    fusion = config.get('model', {}).get('fusion', {})
    
    ok = True
    
    # Phase 1 checks
    checks = {
        'fusion_dim': 64,
        'num_heads': 4,
        'refine_depth': 4,
        'refine_channels': 64,
        'enable_hierarchical': True,
        'enable_multi_domain_freq': True,
    }
    
    for key, expected in checks.items():
        val = fusion.get(key)
        status = "PASS" if val == expected else "FAIL"
        results.append(f"  {key}: {val} (expected {expected}) - {status}")
        if val != expected:
            ok = False
    
    results.append(f"\nTest 3 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


if __name__ == '__main__':
    all_results = []
    all_passed = True
    
    for test_fn in [test_standalone_frequency_module, test_full_model_integration, test_yaml_config]:
        try:
            results, passed = test_fn()
            all_results.extend(results)
            all_passed = all_passed and passed
        except Exception as e:
            import traceback
            all_results.append(f"\nEXCEPTION in {test_fn.__name__}:")
            all_results.append(traceback.format_exc())
            all_passed = False
    
    all_results.append("\n" + "=" * 60)
    all_results.append(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    all_results.append("=" * 60)
    
    # Write to file (avoid encoding issues on Windows)
    output_path = os.path.join(os.path.dirname(__file__), 'phase2_results.txt')
    with open(output_path, 'w', encoding='ascii', errors='replace') as f:
        f.write('\n'.join(all_results))
    
    print(f"Results written to {output_path}")
    
    # Also print
    for line in all_results:
        print(line)
