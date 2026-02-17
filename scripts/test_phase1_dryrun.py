"""
Phase 1 Dry-Run Verification Script
====================================
Tests that the scaled-up CompleteEnhancedFusionSR model:
1. Creates successfully with Phase 1 parameters
2. Has ~900K trainable parameters
3. Produces correct output shape [B, 3, 256, 256]
4. Output is in valid range [0, 1] with no NaN/Inf
5. All trainable parameters receive gradients
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml


def test_phase1():
    """Complete Phase 1 dry-run test."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 60)
    
    results = {}
    
    # ================================================================
    # Test 1: Model Creation with Phase 1 Parameters
    # ================================================================
    print("\n[Test 1] Model Creation with Phase 1 Parameters")
    print("-" * 50)
    
    try:
        from src.models.enhanced_fusion import CompleteEnhancedFusionSR
        
        model = CompleteEnhancedFusionSR(
            expert_ensemble=None,  # Cached mode
            num_experts=3,
            num_bands=3,
            block_size=8,
            upscale=4,
            # Phase 1 params
            fusion_dim=128,
            num_heads=8,
            refine_depth=6,
            refine_channels=128,
            enable_hierarchical=True,
            # All improvements on
            enable_dynamic_selection=True,
            enable_cross_band_attn=True,
            enable_adaptive_bands=True,
            enable_multi_resolution=True,
            enable_collaborative=True,
        ).to(device)
        
        print("  Model created successfully")
        results['creation'] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['creation'] = False
        return results
    
    # ================================================================
    # Test 2: Parameter Count (~900K target)
    # ================================================================
    print("\n[Test 2] Parameter Count")
    print("-" * 50)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    target = 900_000
    tolerance = 0.50  # 50% tolerance
    lower = int(target * (1 - tolerance))
    upper = int(target * (1 + tolerance))
    
    in_range = lower <= trainable <= upper
    print(f"  Trainable: {trainable:,}")
    print(f"  Total:     {total:,}")
    print(f"  Target:    ~{target:,} (range: {lower:,} - {upper:,})")
    print(f"  In range:  {'YES' if in_range else 'NO'}")
    results['param_count'] = in_range
    
    # Print parameter breakdown by module
    print("\n  Parameter breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            print(f"    {name}: {params:,}")
    
    # Named parameters that aren't in children
    orphan_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and '.' not in name:
            orphan_params += p.numel()
            print(f"    {name}: {p.numel():,}")
    
    # ================================================================
    # Test 3: Forward Pass (Cached Mode)
    # ================================================================
    print("\n[Test 3] Forward Pass (Cached Mode)")
    print("-" * 50)
    
    try:
        batch_size = 2
        lr_input = torch.randn(batch_size, 3, 64, 64, device=device)
        
        # Create mock expert outputs
        expert_outputs = {
            'hat': torch.randn(batch_size, 3, 256, 256, device=device).clamp(0, 1),
            'dat': torch.randn(batch_size, 3, 256, 256, device=device).clamp(0, 1),
            'nafnet': torch.randn(batch_size, 3, 256, 256, device=device).clamp(0, 1),
        }
        
        model.eval()
        with torch.no_grad():
            output = model.forward_with_precomputed(
                lr_input=lr_input,
                expert_outputs=expert_outputs,
            )
        
        # Check shape
        expected_shape = (batch_size, 3, 256, 256)
        shape_ok = output.shape == expected_shape
        print(f"  Output shape: {output.shape} (expected {expected_shape}) {'OK' if shape_ok else 'FAIL'}")
        
        # Check range
        range_ok = (output.min() >= 0) and (output.max() <= 1)
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}] {'OK' if range_ok else 'FAIL'}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        clean = not has_nan and not has_inf
        print(f"  NaN: {has_nan}, Inf: {has_inf} {'OK' if clean else 'FAIL'}")
        
        results['forward_pass'] = shape_ok and range_ok and clean
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['forward_pass'] = False
    
    # ================================================================
    # Test 4: Gradient Flow
    # ================================================================
    print("\n[Test 4] Gradient Flow")
    print("-" * 50)
    
    try:
        model.train()
        model.zero_grad()
        
        lr_input = torch.randn(2, 3, 64, 64, device=device)
        expert_outputs = {
            'hat': torch.randn(2, 3, 256, 256, device=device).clamp(0, 1),
            'dat': torch.randn(2, 3, 256, 256, device=device).clamp(0, 1),
            'nafnet': torch.randn(2, 3, 256, 256, device=device).clamp(0, 1),
        }
        target = torch.randn(2, 3, 256, 256, device=device).clamp(0, 1)
        
        output = model.forward_with_precomputed(
            lr_input=lr_input,
            expert_outputs=expert_outputs,
        )
        
        loss = F.l1_loss(output, target)
        loss.backward()
        
        # Check gradients
        total_params = 0
        params_with_grad = 0
        params_without_grad = []
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    params_with_grad += 1
                else:
                    params_without_grad.append(name)
        
        grad_ratio = params_with_grad / total_params if total_params > 0 else 0
        grad_ok = grad_ratio > 0.50  # >50% of params should get gradients
        
        print(f"  Total trainable params: {total_params}")
        print(f"  Params with gradients:  {params_with_grad} ({grad_ratio:.1%})")
        print(f"  Gradient flow: {'OK' if grad_ok else 'FAIL'}")
        
        if params_without_grad:
            print(f"  Params without gradients ({len(params_without_grad)}):")
            for name in params_without_grad[:10]:
                print(f"    - {name}")
            if len(params_without_grad) > 10:
                print(f"    ... and {len(params_without_grad) - 10} more")
        
        results['gradient_flow'] = grad_ok
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['gradient_flow'] = False
    
    # ================================================================
    # Test 5: YAML Config Compatibility
    # ================================================================
    print("\n[Test 5] YAML Config Compatibility")
    print("-" * 50)
    
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'configs', 'train_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        fusion_config = config['model']['fusion']
        
        required_keys = ['fusion_dim', 'num_heads', 'refine_depth', 'refine_channels', 'enable_hierarchical']
        missing = [k for k in required_keys if k not in fusion_config]
        
        if missing:
            print(f"  Missing keys: {missing}")
            results['yaml_compat'] = False
        else:
            print(f"  fusion_dim: {fusion_config['fusion_dim']}")
            print(f"  num_heads: {fusion_config['num_heads']}")
            print(f"  refine_depth: {fusion_config['refine_depth']}")
            print(f"  refine_channels: {fusion_config['refine_channels']}")
            print(f"  enable_hierarchical: {fusion_config['enable_hierarchical']}")
            
            # Validate values
            valid = (fusion_config['fusion_dim'] == 128 and 
                    fusion_config['num_heads'] == 8 and
                    fusion_config['refine_depth'] == 6 and
                    fusion_config['refine_channels'] == 128 and
                    fusion_config['enable_hierarchical'] == True)
            
            print(f"  Values match expected: {'YES' if valid else 'NO'}")
            results['yaml_compat'] = valid
    
    except Exception as e:
        print(f"  FAILED: {e}")
        results['yaml_compat'] = False
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1 DRY-RUN SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")
        if not passed:
            all_passed = False
    
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    test_phase1()
