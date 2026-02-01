"""
Test Training Components
========================
Test all training utilities before full training.

Run: python scripts/test_training_components.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "=" * 70)
print("TESTING PHASE 6 TRAINING COMPONENTS")
print("=" * 70)

# Track results
test_results = []

# Test 1: Metrics
print("\n" + "-" * 40)
print("Test 1: Metrics Module")
print("-" * 40)
try:
    from src.utils.metrics import test_metrics
    test_metrics()
    test_results.append(("Metrics", "PASSED"))
except Exception as e:
    print(f"  ERROR: {e}")
    test_results.append(("Metrics", f"FAILED: {e}"))

# Test 2: Checkpoint Manager
print("\n" + "-" * 40)
print("Test 2: Checkpoint Manager")
print("-" * 40)
try:
    from src.utils.checkpoint_manager import test_checkpoint_manager
    test_checkpoint_manager()
    test_results.append(("Checkpoint Manager", "PASSED"))
except Exception as e:
    print(f"  ERROR: {e}")
    test_results.append(("Checkpoint Manager", f"FAILED: {e}"))

# Test 3: Logger
print("\n" + "-" * 40)
print("Test 3: TensorBoard Logger")
print("-" * 40)
try:
    from src.utils.logger import test_logger
    test_logger()
    test_results.append(("Logger", "PASSED"))
except Exception as e:
    print(f"  ERROR: {e}")
    test_results.append(("Logger", f"FAILED: {e}"))

# Test 4: Combined Loss set_weights
print("\n" + "-" * 40)
print("Test 4: CombinedLoss set_weights")
print("-" * 40)
try:
    import torch
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    
    criterion = CombinedLoss(use_swt=PYWT_AVAILABLE, use_fft=True)
    
    # Test set_weights
    criterion.set_weights({'l1': 0.8, 'swt': 0.2, 'vgg': 0.0})
    assert criterion.weights['l1'] == 0.8
    assert criterion.weights['swt'] == 0.2
    print(f"  set_weights: OK")
    print(f"  Active weights: {criterion.get_active_weights()}")
    
    # Test forward with new weights
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    loss, components = criterion(pred, target, return_components=True)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Components: {list(components.keys())}")
    
    test_results.append(("CombinedLoss", "PASSED"))
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    test_results.append(("CombinedLoss", f"FAILED: {e}"))

# Test 5: Config Loading
print("\n" + "-" * 40)
print("Test 5: Config Loading")
print("-" * 40)
try:
    import yaml
    config_path = project_root / 'configs' / 'train_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Epochs: {config['training']['total_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Loss stages: {len(config['loss']['stages'])}")
    
    test_results.append(("Config Loading", "PASSED"))
except Exception as e:
    print(f"  ERROR: {e}")
    test_results.append(("Config Loading", f"FAILED: {e}"))

# Test 6: Module Imports
print("\n" + "-" * 40)
print("Test 6: Module Imports")
print("-" * 40)
try:
    from src.utils import (
        MetricCalculator,
        CheckpointManager,
        TensorBoardLogger,
        EMAModel,
        calculate_psnr,
        calculate_ssim
    )
    from src.data import create_dataloaders, SRDataset
    from src.losses import CombinedLoss
    from src.models.fusion_network import FrequencyAwareFusion
    
    print("  All imports successful!")
    test_results.append(("Module Imports", "PASSED"))
except Exception as e:
    print(f"  ERROR: {e}")
    test_results.append(("Module Imports", f"FAILED: {e}"))

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

all_passed = True
for name, result in test_results:
    status = "✓" if result == "PASSED" else "✗"
    print(f"  {status} {name}: {result}")
    if result != "PASSED":
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✓ ALL TRAINING COMPONENT TESTS PASSED!")
    print("  Ready to start training: python train.py --config configs/train_config.yaml")
else:
    print("✗ SOME TESTS FAILED - Please fix before training")
print("=" * 70 + "\n")
