# ============================================================================
# PHASE 5 CHAMPIONSHIP TRAINING LAUNCHER (SINGLE GPU - WINDOWS)
# ============================================================================
# Hardware: Single P5000 GPU (16GB VRAM)
# Estimated time: 3-4 days
# Target: 35.3-35.5 dB PSNR
# ============================================================================

$ErrorActionPreference = "Stop"

Write-Host "=" * 80
Write-Host "  CHAMPIONSHIP SR - PHASE 5 TRAINING (SINGLE GPU)" -ForegroundColor Cyan
Write-Host "=" * 80
Write-Host ""
Write-Host "Hardware Configuration:"
Write-Host "  GPU: Single NVIDIA P5000 (16GB VRAM)"
Write-Host "  Batch Size: 24"
Write-Host "  Workers: 4"
Write-Host ""
Write-Host "Model Configuration:"
Write-Host "  Phase 1: Scale + Hierarchical Fusion"
Write-Host "  Phase 2: Multi-Domain Frequency (DCT+DWT+FFT)"
Write-Host "  Phase 3: Large Kernel Attention (21x21)"
Write-Host "  Phase 4: Laplacian Edge Enhancement"
Write-Host "  Parameters: ~1.02M trainable"
Write-Host ""
Write-Host "Training Configuration:"
Write-Host "  Epochs: 200"
Write-Host "  Learning Rate: 1e-4"
Write-Host "  Stages: 3 (Foundation -> Frequency -> Detail)"
Write-Host "  Target: 35.3-35.5 dB PSNR"
Write-Host ""
Write-Host "=" * 80

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = "$($PWD.Path);$env:PYTHONPATH"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

Write-Host ""
Write-Host "Environment configured:" -ForegroundColor Green
Write-Host "  CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
Write-Host ""

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
Write-Host "Running pre-flight checks..." -ForegroundColor Yellow
Write-Host ""

# Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "  [OK] Python: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Python not found!" -ForegroundColor Red
    exit 1
}

# Check CUDA
try {
    python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'"
    $torchVer = python -c "import torch; print(torch.__version__)"
    $cudaVer = python -c "import torch; print(torch.version.cuda)"
    Write-Host "  [OK] PyTorch: $torchVer" -ForegroundColor Green
    Write-Host "  [OK] CUDA: $cudaVer" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] CUDA not available!" -ForegroundColor Red
    exit 1
}

# Check GPU
$gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
$gpuMem = python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}')"
Write-Host "  [OK] GPU: $gpuName ($gpuMem GB)" -ForegroundColor Green

# Check pretrained models
$modelsOK = $true
@(
    "pretrained/HAT-L_SRx4_ImageNet-pretrain.pth",
    "pretrained/DAT_x4.pth",
    "pretrained/NAFNet-SIDD-width64.pth"
) | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "  [OK] Found: $_" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Missing: $_" -ForegroundColor Yellow
        $modelsOK = $false
    }
}

# Check config
if (Test-Path "configs/train_config.yaml") {
    Write-Host "  [OK] Config file found" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Config file missing!" -ForegroundColor Red
    exit 1
}

# Check required packages
try {
    python -c "import yaml, numpy, torch, torchvision, PIL"
    Write-Host "  [OK] All required packages installed" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Missing required packages!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 80
Write-Host "  ALL PRE-FLIGHT CHECKS PASSED!" -ForegroundColor Green
Write-Host "=" * 80
Write-Host ""

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================
Write-Host "Creating output directories..."
New-Item -ItemType Directory -Force -Path "logs/phase5_single_gpu" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints/phase5_single_gpu" | Out-Null
New-Item -ItemType Directory -Force -Path "results/phase5_single_gpu" | Out-Null
Write-Host "  [OK] Directories created" -ForegroundColor Green
Write-Host ""

# ============================================================================
# LAUNCH TRAINING
# ============================================================================
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/phase5_single_gpu/training_$timestamp.log"

Write-Host "=" * 80
Write-Host "  LAUNCHING TRAINING" -ForegroundColor Cyan
Write-Host "=" * 80
Write-Host ""
Write-Host "Log file: $logFile"
Write-Host ""
Write-Host "Monitor training:"
Write-Host "  Live monitor: python scripts/monitor_training.py"
Write-Host "  TensorBoard:  tensorboard --logdir logs/phase5_single_gpu"
Write-Host "  GPU usage:    nvidia-smi -l 1"
Write-Host ""
Write-Host "Training started at: $(Get-Date)"
Write-Host ""

# Launch training with logging
python train.py `
    --config configs/train_config.yaml `
    --cached `
    --gpu 0 `
    2>&1 | Tee-Object -FilePath $logFile

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=" * 80
if ($exitCode -eq 0) {
    Write-Host "  TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
} else {
    Write-Host "  TRAINING FAILED (Exit code: $exitCode)" -ForegroundColor Red
}
Write-Host "=" * 80
Write-Host ""
Write-Host "Training finished at: $(Get-Date)"
Write-Host ""
Write-Host "Results:"
Write-Host "  Logs: logs/phase5_single_gpu/"
Write-Host "  Checkpoints: checkpoints/phase5_single_gpu/"
Write-Host ""

exit $exitCode
