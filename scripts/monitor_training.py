"""
Real-time Training Monitor
===========================
Displays live training progress, metrics, and completion estimates.

Usage:
    python scripts/monitor_training.py
    python scripts/monitor_training.py --log logs/phase5_single_gpu/training.log
    python scripts/monitor_training.py --refresh 60
"""

import time
import sys
import os
from pathlib import Path
import re
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TrainingMonitor:
    """Monitor training progress in real-time."""
    
    def __init__(self, log_file=None, checkpoint_dir=None):
        if log_file is None:
            log_dir = project_root / 'logs' / 'phase5_single_gpu'
            if log_dir.exists():
                log_files = sorted(
                    log_dir.glob('training_*.log'),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if log_files:
                    log_file = log_files[0]
        
        self.log_file = Path(log_file) if log_file else None
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir 
            else project_root / 'checkpoints' / 'phase5_single_gpu'
        )
        self.start_time = None
        self.last_update = None
    
    def parse_latest_log(self):
        """Parse latest training metrics from log."""
        if not self.log_file or not self.log_file.exists():
            return None
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()[-200:]
            
            metrics = {}
            
            for line in reversed(lines):
                if 'Epoch' in line and not metrics.get('epoch'):
                    match = re.search(r'Epoch\s*\[?(\d+)[/\]]', line)
                    if not match:
                        match = re.search(r'Epoch\s+(\d+)', line)
                    if match:
                        metrics['epoch'] = int(match.group(1))
                
                if 'loss' in line.lower() and not metrics.get('loss'):
                    match = re.search(r'[Ll]oss[:\s=]+([0-9.]+)', line)
                    if match:
                        metrics['loss'] = float(match.group(1))
                
                if 'PSNR' in line and not metrics.get('psnr'):
                    match = re.search(r'PSNR[:\s]+([0-9.]+)', line)
                    if match:
                        metrics['psnr'] = float(match.group(1))
                
                if 'SSIM' in line and not metrics.get('ssim'):
                    match = re.search(r'SSIM[:\s]+([0-9.]+)', line)
                    if match:
                        metrics['ssim'] = float(match.group(1))
                
                if 'LR' in line and not metrics.get('lr'):
                    match = re.search(r'LR[:\s]+([0-9.e\-]+)', line)
                    if match:
                        try:
                            metrics['lr'] = float(match.group(1))
                        except ValueError:
                            pass
                
                if 'stage' in line.lower() and not metrics.get('stage'):
                    match = re.search(r'\[(\w+_?\w*)\]', line)
                    if match:
                        metrics['stage'] = match.group(1)
                
                if len(metrics) >= 5:
                    break
            
            return metrics if metrics else None
        except Exception:
            return None
    
    def get_best_checkpoint_info(self):
        """Get info about best checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        
        try:
            checkpoints = list(self.checkpoint_dir.glob('best_*.pth'))
            if not checkpoints:
                checkpoints = list(self.checkpoint_dir.glob('*.pth'))
            if not checkpoints:
                return None
            
            best = max(checkpoints, key=lambda x: x.stat().st_mtime)
            psnr_match = re.search(r'([0-9]+\.[0-9]+)', best.name)
            
            return {
                'path': best,
                'psnr': float(psnr_match.group(1)) if psnr_match else 0.0,
                'modified': datetime.fromtimestamp(best.stat().st_mtime)
            }
        except Exception:
            return None
    
    def estimate_completion(self, current_epoch, total_epochs, elapsed_time):
        """Estimate training completion time."""
        if current_epoch <= 0:
            return None
        epochs_remaining = total_epochs - current_epoch
        time_per_epoch = elapsed_time / current_epoch
        eta_seconds = epochs_remaining * time_per_epoch
        return timedelta(seconds=int(eta_seconds))
    
    def display_dashboard(self):
        """Display real-time training dashboard."""
        # Clear screen (cross-platform)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("  CHAMPIONSHIP SR - PHASE 5 TRAINING MONITOR (SINGLE P5000)")
        print("=" * 80)
        print()
        
        metrics = self.parse_latest_log()
        
        if metrics:
            epoch = metrics.get('epoch', '?')
            total = 200
            progress_pct = (epoch / total * 100) if isinstance(epoch, int) else 0
            
            # Progress bar
            bar_len = 40
            filled = int(bar_len * progress_pct / 100) if isinstance(epoch, int) else 0
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"  [{bar}] {progress_pct:.1f}%")
            print()
            
            print(f"  📊 Current Metrics:")
            print(f"     Epoch:    {epoch}/{total}")
            if 'stage' in metrics:
                print(f"     Stage:    {metrics['stage']}")
            if 'loss' in metrics:
                print(f"     Loss:     {metrics['loss']:.6f}")
            if 'psnr' in metrics:
                print(f"     PSNR:     {metrics['psnr']:.2f} dB")
            if 'ssim' in metrics:
                print(f"     SSIM:     {metrics['ssim']:.4f}")
            if 'lr' in metrics:
                print(f"     LR:       {metrics['lr']:.2e}")
            print()
            
            if self.start_time and isinstance(epoch, int) and epoch > 0:
                elapsed = time.time() - self.start_time
                eta = self.estimate_completion(epoch, total, elapsed)
                
                print(f"  ⏱️  Time:")
                print(f"     Elapsed:     {timedelta(seconds=int(elapsed))}")
                if eta:
                    print(f"     ETA:         {eta}")
                    completion_time = datetime.now() + eta
                    print(f"     Complete at: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("  ⏳ Waiting for training to start...")
            if self.log_file:
                print(f"     Log file: {self.log_file}")
                if self.log_file.exists():
                    print(f"     Log size: {self.log_file.stat().st_size / 1024:.1f} KB")
                else:
                    print(f"     Log file not yet created")
            print()
        
        best = self.get_best_checkpoint_info()
        if best:
            print(f"  🏆 Best Checkpoint:")
            print(f"     PSNR: {best['psnr']:.2f} dB")
            print(f"     File: {best['path'].name}")
            print(f"     Time: {best['modified'].strftime('%Y-%m-%d %H:%M')}")
            print()
        
        print("  🎯 Target Milestones:")
        milestones = [
            (20,  31.0, "Early Progress"),
            (80,  33.2, "Stage 1→2 (Foundation → Frequency)"),
            (150, 34.5, "Stage 2→3 (Frequency → Detail)"),
            (200, 35.5, "CHAMPIONSHIP TARGET 🏆"),
        ]
        
        current_epoch = metrics.get('epoch', 0) if metrics else 0
        for m_epoch, m_psnr, desc in milestones:
            if isinstance(current_epoch, int):
                status = "✅" if current_epoch >= m_epoch else "⏳"
            else:
                status = "⏳"
            print(f"     {status} Epoch {m_epoch:3d}: {m_psnr:.1f} dB — {desc}")
        
        print()
        print("=" * 80)
        self.last_update = datetime.now()
        print(f"  Last updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.log_file and self.log_file.exists():
            mod_time = datetime.fromtimestamp(self.log_file.stat().st_mtime)
            seconds_ago = (datetime.now() - mod_time).total_seconds()
            warning = "  ⚠️  POSSIBLE STALL!" if seconds_ago > 300 else ""
            print(f"  Log modified: {int(seconds_ago)}s ago{warning}")
        
        print("  Press Ctrl+C to exit")
        print("=" * 80)
    
    def run(self, refresh_interval=30):
        """Run monitoring loop."""
        self.start_time = time.time()
        
        print("Starting training monitor...")
        print(f"Refresh interval: {refresh_interval}s")
        if self.log_file:
            print(f"Log file: {self.log_file}")
        print()
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n\nMonitor stopped after {timedelta(seconds=int(time.time() - self.start_time))}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to training log file')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to checkpoint directory')
    parser.add_argument('--refresh', type=int, default=30,
                        help='Refresh interval in seconds')
    
    args = parser.parse_args()
    monitor = TrainingMonitor(args.log, args.checkpoint_dir)
    monitor.run(refresh_interval=args.refresh)
