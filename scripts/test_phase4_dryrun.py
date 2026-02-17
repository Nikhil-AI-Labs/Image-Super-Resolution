"""
Phase 4 Dry-Run Verification Script
====================================
Tests:
1. Standalone LaplacianPyramidRefinement (shapes, gradients, edge detection)
2. Full model with Phase 1+2+3+4 (forward pass, gradients, output validity)
3. YAML config compatibility
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml


def test_standalone_edge():
    """Test standalone edge enhancement modules."""
    from src.models.edge_enhancement import LaplacianPyramidRefinement

    results = []
    results.append("=" * 60)
    results.append("TEST 1: Standalone Edge Enhancement")
    results.append("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results.append(f"Device: {device}")

    ok = True

    # 1a: LaplacianPyramidRefinement
    module = LaplacianPyramidRefinement(
        num_levels=3, channels=32, edge_strength=0.15
    ).to(device)
    p = sum(pp.numel() for pp in module.parameters())
    results.append(f"\n  LaplacianPyramidRefinement: {p:,} params")

    x = torch.randn(2, 3, 256, 256, device=device).clamp(0, 1)
    out = module(x)
    results.append(f"    Shape: {x.shape} -> {out.shape}")
    results.append(f"    Range: [{out.min():.4f}, {out.max():.4f}]")
    results.append(f"    NaN: {torch.isnan(out).any().item()}")
    if out.shape != x.shape:
        results.append("    FAIL: shape mismatch"); ok = False
    if torch.isnan(out).any():
        results.append("    FAIL: NaN"); ok = False

    # 1b: Pyramid levels
    pyr, sizes = module.build_laplacian_pyramid(x)
    results.append(f"\n  Laplacian pyramid ({len(pyr)} levels):")
    for i, (lev, sz) in enumerate(zip(pyr, sizes)):
        results.append(f"    Level {i}: {lev.shape}, size={sz}, range=[{lev.min():.3f}, {lev.max():.3f}]")

    # 1c: Edge detection check
    test_img = torch.zeros(1, 3, 256, 256, device=device)
    test_img[:, :, :, :128] = 0.3
    test_img[:, :, :, 128:] = 0.7
    pyr2, _ = module.build_laplacian_pyramid(test_img)
    edge_val = pyr2[0].abs().max().item()
    results.append(f"\n  Edge detection (sharp step): max Laplacian = {edge_val:.4f}")
    if edge_val < 0.01:
        results.append("    FAIL: Edge not detected"); ok = False

    # 1d: Gradient flow
    module.zero_grad()
    loss = module(x).mean()
    loss.backward()
    g = sum(1 for pp in module.parameters() if pp.requires_grad and pp.grad is not None and pp.grad.abs().sum() > 0)
    t = sum(1 for pp in module.parameters() if pp.requires_grad)
    results.append(f"\n  Gradients: {g}/{t}")
    if g == 0:
        results.append("    FAIL: no gradients"); ok = False

    results.append(f"\nTest 1 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


def test_full_model():
    """Test full model with Phase 1+2+3+4."""
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR

    results = []
    results.append("\n" + "=" * 60)
    results.append("TEST 2: Full Model (Phase 1 + 2 + 3 + 4)")
    results.append("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CompleteEnhancedFusionSR(
        expert_ensemble=None,
        num_experts=3,
        num_bands=3,
        block_size=8,
        upscale=4,
        fusion_dim=64,
        num_heads=4,
        refine_depth=4,
        refine_channels=64,
        enable_hierarchical=True,
        enable_multi_domain_freq=True,   # Phase 2
        enable_lka=True,                 # Phase 3
        enable_edge_enhance=True,        # Phase 4
        enable_dynamic_selection=True,
        enable_cross_band_attn=True,
        enable_adaptive_bands=True,
        enable_multi_resolution=True,
        enable_collaborative=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results.append(f"Total trainable params: {total_params:,}")

    results.append("\nModule breakdown:")
    for name, child in model.named_children():
        p = sum(pp.numel() for pp in child.parameters() if pp.requires_grad)
        if p > 0:
            results.append(f"  {name}: {p:,}")
    child_params = sum(pp.numel() for n, c in model.named_children()
                       for pp in c.parameters() if pp.requires_grad)
    orphan = total_params - child_params
    if orphan > 0:
        results.append(f"  (direct params): {orphan:,}")

    # Forward pass
    B = 2
    lr_input = torch.randn(B, 3, 64, 64, device=device)
    expert_outputs = {
        'hat': torch.randn(B, 3, 256, 256, device=device),
        'dat': torch.randn(B, 3, 256, 256, device=device),
        'nafnet': torch.randn(B, 3, 256, 256, device=device),
    }
    expert_features = {
        'hat': torch.randn(B, 180, 64, 64, device=device),
        'dat': torch.randn(B, 180, 64, 64, device=device),
        'nafnet': torch.randn(B, 64, 64, 64, device=device),
    }

    output = model.forward_with_precomputed(lr_input, expert_outputs, expert_features)

    results.append(f"\nForward pass:")
    results.append(f"  Input: {lr_input.shape}")
    results.append(f"  Output: {output.shape}")
    results.append(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    results.append(f"  NaN: {torch.isnan(output).any().item()}")
    results.append(f"  Inf: {torch.isinf(output).any().item()}")

    # Gradient check
    model.zero_grad()
    loss = output.mean()
    loss.backward()

    grads = sum(1 for p in model.parameters()
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    results.append(f"\nGradient flow: {grads}/{total} ({100*grads/total:.1f}%)")

    results.append("Per-module gradient flow:")
    for name, child in model.named_children():
        t = sum(1 for p in child.parameters() if p.requires_grad)
        g = sum(1 for p in child.parameters()
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        if t > 0:
            results.append(f"  {name}: {g}/{t}")

    # Edge-specific gradient check
    edge_grads = sum(1 for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None
                     and p.grad.abs().sum() > 0
                     and ('edge' in n or 'laplacian' in n))
    results.append(f"\nEdge-related params with gradients: {edge_grads}")

    # Assertions
    ok = True
    if output.shape != (B, 3, 256, 256):
        results.append(f"FAIL: Wrong output shape {output.shape}"); ok = False
    if torch.isnan(output).any():
        results.append("FAIL: NaN in output"); ok = False
    if torch.isinf(output).any():
        results.append("FAIL: Inf in output"); ok = False
    if grads == 0:
        results.append("FAIL: No gradients"); ok = False
    if edge_grads == 0:
        results.append("FAIL: Edge modules not receiving gradients"); ok = False

    results.append(f"\nTest 2 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


def test_yaml_config():
    """Verify YAML config has Phase 4 settings."""
    results = []
    results.append("\n" + "=" * 60)
    results.append("TEST 3: YAML Config Compatibility")
    results.append("=" * 60)

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    fusion = config.get('model', {}).get('fusion', {})

    ok = True
    checks = {
        'enable_hierarchical': True,
        'enable_multi_domain_freq': True,
        'enable_lka': True,
        'enable_edge_enhance': True,
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

    for test_fn in [test_standalone_edge, test_full_model, test_yaml_config]:
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

    output_path = os.path.join(os.path.dirname(__file__), 'phase4_results.txt')
    with open(output_path, 'w', encoding='ascii', errors='replace') as f:
        f.write('\n'.join(all_results))

    print(f"Results written to {output_path}")
    for line in all_results:
        print(line)
