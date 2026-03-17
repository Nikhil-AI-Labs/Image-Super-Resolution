"""
Kaggle Inference — Fixed (no TTA, direct inference)
====================================================
Fixed:
  1. Monkey-patches sys.modules to avoid diffusers/peft/transformers crash
  2. Uses subprocess-based multi-GPU (avoids mp.spawn pickle error in notebooks)

Usage: Copy each CELL block into a separate Kaggle notebook cell.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║ CELL 1 — Install dependencies + Mamba wheels                     ║
# ╚══════════════════════════════════════════════════════════════════╝
import subprocess, sys, os

CODE = '/kaggle/input/datasets/nikhilpathaksvnit/sr-championship-code'

CAUSAL_WHL = f'{CODE}/mamba_wheels/causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl'
MAMBA_WHL  = f'{CODE}/mamba_wheels/mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl'

req = f'{CODE}/requirements.txt'
if os.path.exists(req):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', req], check=True)
    print("  ✅ requirements.txt installed")

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', CAUSAL_WHL], check=True)
print("  ✅ causal_conv1d installed")

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', MAMBA_WHL], check=True)
print("  ✅ mamba_ssm installed")

import mamba_ssm
print(f"  ✅ import mamba_ssm OK  (v{mamba_ssm.__version__})")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ CELL 2 — Paths                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
import torch, glob, os

CODE       = '/kaggle/input/datasets/nikhilpathaksvnit/sr-championship-code'
PRETRAINED = f'{CODE}/pretrained'

DRCT_W     = f'{PRETRAINED}/drct/DRCT-L_X4.pth'
GRL_W      = f'{PRETRAINED}/grl/GRL-B_SR_x4.pth'
NAFNET_W   = f'{PRETRAINED}/nafnet/NAFNet-SIDD-width64.pth'
MAMBA_W    = f'{PRETRAINED}/mambair/MambaIR_x4.pth'

CHECKPOINT = (f'{CODE}/checkpoints/phase3_single_gpu/'
              f'championship_sr_drct_grl_nafnet_mamba/'
              f'best_epoch0040_psnr30.79.pth')

CONFIG     = f'{CODE}/configs/train_config.yaml'
TEST_LR_DIR= f'{CODE}/DIV2K_test_LR_bicubic/X4'

SUBMIT_DIR = '/tmp/submit_direct'
FINAL_ZIP  = '/kaggle/working/res.zip'

num_gpus   = torch.cuda.device_count()
test_imgs  = sorted(glob.glob(f'{TEST_LR_DIR}/*.png') +
                    glob.glob(f'{TEST_LR_DIR}/*.jpg'))

os.makedirs(SUBMIT_DIR, exist_ok=True)

print(f"  GPUs      : {num_gpus}")
print(f"  Images    : {len(test_imgs)}")
print(f"  Output    : {SUBMIT_DIR}")
print(f"  Final ZIP : {FINAL_ZIP}")
assert len(test_imgs) > 0, "❌  No test images found"
print("  ✅ Ready")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ CELL 3 — Write standalone GPU worker script to /tmp               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  mp.spawn can NOT pickle functions defined in Jupyter __main__.   ║
# ║  Solution: write a standalone .py script and launch via subprocess║
# ╚══════════════════════════════════════════════════════════════════╝
import os, json

WORKER_SCRIPT = '/tmp/sr_gpu_worker.py'

worker_code = r'''#!/usr/bin/env python3
"""
Standalone GPU worker — launched via subprocess from Kaggle notebook.
Usage: python sr_gpu_worker.py <rank> <num_gpus> <args.json>
"""
import sys, os, types, json, time

# ── Read CLI args ─────────────────────────────────────────────────────
rank     = int(sys.argv[1])
num_gpus = int(sys.argv[2])
with open(sys.argv[3]) as f:
    args = json.load(f)

code_dir    = args['code_dir']
config_path = args['config_path']
drct_w      = args['drct_w']
grl_w       = args['grl_w']
nafnet_w    = args['nafnet_w']
mamba_w     = args['mamba_w']
checkpoint  = args['checkpoint']
submit_dir  = args['submit_dir']
img_paths   = args['img_paths']

# ── Monkey-patch: prevent tsdsr_wrapper / complete_sr_pipeline imports ─
for mod_name in ['src.models.tsdsr_wrapper', 'src.models.complete_sr_pipeline']:
    fake = types.ModuleType(mod_name)
    # Add dummy attributes __init__.py expects
    for attr in ['TSDSRInference', 'VAEWrapper', 'CompleteSRPipeline']:
        setattr(fake, attr, type('Dummy', (), {}))
    for attr in ['load_tsdsr_models', 'create_tsdsr_refinement_pipeline',
                 'create_complete_pipeline', 'create_training_pipeline',
                 'create_inference_pipeline']:
        setattr(fake, attr, lambda *a, **kw: None)
    sys.modules[mod_name] = fake

sys.path.insert(0, code_dir)

# ── Imports ───────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import cv2, yaml, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

device   = torch.device(f'cuda:{rank}')
my_paths = img_paths[rank::num_gpus]
is_main  = (rank == 0)

print(f"[GPU {rank}] {len(my_paths)} images assigned")


def pad16(t):
    _, _, h, w = t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (h, w)


def unpad(t, orig_h, orig_w, scale=4):
    return t[:, :, :orig_h * scale, :orig_w * scale]


def load_lr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load: {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def save_png(tensor, path):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
           ).round().astype(np.uint8)
    Image.fromarray(arr).save(path, format='PNG')


# ── Load config ───────────────────────────────────────────────────────
with open(config_path) as f:
    cfg = yaml.safe_load(f)

fusion_cfg   = cfg.get('model', {}).get('fusion', {})
improvements = fusion_cfg.get('improvements', {})
scale        = cfg.get('dataset', {}).get('scale', 4)

# ── Load 3 local experts (DRCT, GRL, NAFNet) ─────────────────────────
from src.models.expert_loader import ExpertEnsemble

ckpts = {'drct': drct_w, 'grl': grl_w, 'nafnet': nafnet_w}
ensemble = ExpertEnsemble(device=device, upscale=scale)
ensemble.load_all_experts(checkpoint_paths=ckpts, freeze=True)
ensemble._register_all_hooks()
print(f"[GPU {rank}] ✅ DRCT + GRL + NAFNet loaded")

# ── Load MambaIR ──────────────────────────────────────────────────────
from src.models.mambair.mambair_arch import MambaIR

mamba = MambaIR(
    upscale=scale, in_chans=3, img_size=64, window_size=16,
    compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
    overlap_ratio=0.5, img_range=1.0,
    depths=(6, 6, 6, 6, 6, 6), embed_dim=180, mlp_ratio=2.0,
    drop_path_rate=0.1, upsampler='pixelshuffle',
    resi_connection='1conv',
)
state = torch.load(mamba_w, map_location='cpu', weights_only=False)
sd    = state.get('params', state.get('state_dict', state.get('model', state)))
sd    = {k.replace('module.', ''): v for k, v in sd.items()}
mamba.load_state_dict(sd, strict=False)
mamba.eval().to(device)

mamba_feat_cache = {}
if hasattr(mamba, 'conv_after_body'):
    mamba.conv_after_body.register_forward_hook(
        lambda m, i, o: mamba_feat_cache.update({'feat': o.detach()})
    )
print(f"[GPU {rank}] ✅ MambaIR loaded")

# ── Load fusion model (cached / headless mode) ───────────────────────
from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR

fusion = CompleteEnhancedFusionSR(
    expert_ensemble=None,
    num_experts    =fusion_cfg.get('num_experts',    4),
    fusion_dim     =fusion_cfg.get('fusion_dim',   128),
    refine_channels=fusion_cfg.get('refine_channels',128),
    refine_depth   =fusion_cfg.get('refine_depth',   6),
    base_channels  =fusion_cfg.get('base_channels', 64),
    block_size     =fusion_cfg.get('block_size',     8),
    upscale        =scale,
    enable_dynamic_selection =improvements.get('dynamic_expert_selection', True),
    enable_cross_band_attn   =improvements.get('cross_band_attention',     True),
    enable_adaptive_bands    =improvements.get('adaptive_frequency_bands', True),
    enable_multi_resolution  =improvements.get('multi_resolution_fusion',  True),
    enable_collaborative     =improvements.get('collaborative_learning',   True),
    enable_edge_enhance      =improvements.get('edge_enhancement',         True),
)

ckpt     = torch.load(checkpoint, map_location='cpu', weights_only=False)
raw_sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
clean_sd = {}
for k, v in raw_sd.items():
    key = k
    for pfx in ['module.', 'model.']:
        if key.startswith(pfx):
            key = key[len(pfx):]
    clean_sd[key] = v

model_sd = fusion.state_dict()
loaded   = sum(1 for k, v in clean_sd.items()
               if k in model_sd and v.shape == model_sd[k].shape)
model_sd.update({k: v for k, v in clean_sd.items()
                 if k in model_sd and v.shape == model_sd[k].shape})
fusion.load_state_dict(model_sd, strict=False)
fusion.eval().to(device)
print(f"[GPU {rank}] ✅ Fusion loaded ({loaded} tensors)")

# ── Inference loop ────────────────────────────────────────────────────
processed, errors = 0, 0
iter_ = tqdm(my_paths, desc=f'GPU {rank}', ncols=90) if is_main else my_paths

with torch.no_grad():
    for lr_path in iter_:
        out_name = Path(lr_path).name
        out_path = os.path.join(submit_dir, out_name)

        if os.path.exists(out_path):
            processed += 1
            continue

        try:
            lr_cpu         = load_lr(lr_path)
            lr_padded, (oh, ow) = pad16(lr_cpu.to(device))

            # DRCT
            ensemble._captured_features = {}
            ensemble._capture_features  = True
            drct_sr  = ensemble.forward_drct(lr_padded)
            drct_feat= ensemble._captured_features.get('drct',
                           torch.zeros(1, 180, oh, ow, device=device))
            ensemble._capture_features = False
            drct_sr  = unpad(drct_sr, oh, ow, scale).cpu().float()
            drct_feat= drct_feat[:, :, :oh, :ow].cpu().float()
            torch.cuda.empty_cache()

            # GRL
            ensemble._captured_features = {}
            ensemble._capture_features  = True
            grl_sr   = ensemble.forward_grl(lr_padded)
            grl_feat = ensemble._captured_features.get('grl',
                           torch.zeros(1, 180, oh, ow, device=device))
            ensemble._capture_features = False
            grl_sr   = unpad(grl_sr, oh, ow, scale).cpu().float()
            grl_feat = grl_feat[:, :, :oh, :ow].cpu().float()
            torch.cuda.empty_cache()

            # NAFNet
            ensemble._captured_features = {}
            ensemble._capture_features  = True
            naf_sr   = ensemble.forward_nafnet(lr_padded)
            naf_feat = ensemble._captured_features.get('nafnet',
                           torch.zeros(1, 64, oh*scale, ow*scale, device=device))
            ensemble._capture_features = False
            naf_sr   = unpad(naf_sr, oh, ow, scale).cpu().float()
            naf_feat = F.interpolate(
                naf_feat, size=(oh, ow),
                mode='bilinear', align_corners=False
            ).cpu().float()
            torch.cuda.empty_cache()

            # MambaIR
            mamba_feat_cache.clear()
            with torch.amp.autocast('cuda'):
                mamba_sr = mamba(lr_padded).clamp(0, 1)
            mamba_sr  = unpad(mamba_sr, oh, ow, scale).cpu().float()
            mamba_feat= mamba_feat_cache.get(
                'feat', torch.zeros(1, 180, oh, ow, device=device)
            )
            mamba_feat= mamba_feat[:, :, :oh, :ow].cpu().float()
            torch.cuda.empty_cache()

            # Fusion
            lr_in = lr_padded[:, :, :oh, :ow].cpu()
            expert_imgs = {
                'drct':   drct_sr.to(device),
                'grl':    grl_sr.to(device),
                'nafnet': naf_sr.to(device),
                'mamba':  mamba_sr.to(device),
            }
            expert_feats = {
                'drct':   drct_feat.to(device),
                'grl':    grl_feat.to(device),
                'nafnet': naf_feat.to(device),
                'mamba':  mamba_feat.to(device),
            }

            sr_out = fusion.forward_with_precomputed(
                lr_in.to(device), expert_imgs, expert_feats
            )

            save_png(sr_out, out_path)
            processed += 1

            del (lr_padded, lr_in,
                 drct_sr, drct_feat, grl_sr, grl_feat,
                 naf_sr, naf_feat, mamba_sr, mamba_feat,
                 expert_imgs, expert_feats, sr_out)
            torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            msg = f"[GPU {rank}] ⚠  {Path(lr_path).name}: {e}"
            if is_main:
                tqdm.write(msg)
            else:
                print(msg, file=sys.stderr)

print(f"[GPU {rank}] Done — processed={processed}, errors={errors}")
'''

with open(WORKER_SCRIPT, 'w') as f:
    f.write(worker_code)
print(f"  ✅ Worker script written to {WORKER_SCRIPT}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║ CELL 4 — Launch subprocess workers (multi-GPU)                   ║
# ╚══════════════════════════════════════════════════════════════════╝
import subprocess, sys, time, json, glob

ARGS_FILE = '/tmp/sr_worker_args.json'
LOG_DIR   = '/tmp/sr_infer'
os.makedirs(LOG_DIR, exist_ok=True)

# Write shared args to JSON (image list etc.)
worker_args = {
    'code_dir':    CODE,
    'config_path': CONFIG,
    'drct_w':      DRCT_W,
    'grl_w':       GRL_W,
    'nafnet_w':    NAFNET_W,
    'mamba_w':     MAMBA_W,
    'checkpoint':  CHECKPOINT,
    'submit_dir':  SUBMIT_DIR,
    'img_paths':   test_imgs,
}
with open(ARGS_FILE, 'w') as f:
    json.dump(worker_args, f)

print("=" * 68)
print("  DIRECT INFERENCE  (no TTA, subprocess per GPU)")
print("=" * 68)
print(f"  Images    : {len(test_imgs)}")
print(f"  GPUs      : {num_gpus}  (~{len(test_imgs)//max(num_gpus,1)} images each)")
print(f"  Output    : {SUBMIT_DIR}")
print("=" * 68 + "\n")

t0 = time.time()

# Launch one subprocess per GPU
procs = []
for gpu_id in range(num_gpus):
    log_path = f'{LOG_DIR}/gpu{gpu_id}.log'
    log_f = open(log_path, 'w')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cmd = [sys.executable, WORKER_SCRIPT, '0', '1', ARGS_FILE]
    # rank=0, num_gpus=1 inside each subprocess because CUDA_VISIBLE_DEVICES
    # makes only 1 GPU visible. But we shard by giving different image lists.
    # Actually — let's pass the real rank/num_gpus and let the worker shard.
    cmd = [sys.executable, WORKER_SCRIPT, str(gpu_id), str(num_gpus), ARGS_FILE]
    # Don't set CUDA_VISIBLE_DEVICES — let the worker use cuda:{rank}
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    procs.append((gpu_id, p, log_f, log_path))
    print(f"  Launching GPU {gpu_id} → log: {log_path}")

print(f"\n  All {num_gpus} workers launched — waiting for completion ...\n")

# Poll for progress with a tqdm bar visible in the notebook
from tqdm.auto import tqdm as tqdm_auto

pbar = tqdm_auto(total=len(test_imgs), desc='SR Inference', unit='img', ncols=90)
prev_done = 0

while True:
    # Check if all workers finished
    all_done = all(p.poll() is not None for _, p, _, _ in procs)
    # Count completed images
    cur_done = len(glob.glob(f'{SUBMIT_DIR}/*.png'))
    if cur_done > prev_done:
        pbar.update(cur_done - prev_done)
        prev_done = cur_done
    if all_done:
        # Final update
        cur_done = len(glob.glob(f'{SUBMIT_DIR}/*.png'))
        if cur_done > prev_done:
            pbar.update(cur_done - prev_done)
        break
    time.sleep(5)

pbar.close()

# Close log file handles
for _, _, log_f, _ in procs:
    log_f.close()

elapsed = time.time() - t0
sr_done = sorted(glob.glob(f'{SUBMIT_DIR}/*.png'))

# Print results and last lines of logs if errors
print(f"\n{'='*68}")
print(f"  All workers finished")
print(f"{'='*68}")

any_error = False
for gpu_id, p, _, log_path in procs:
    status = '✅' if p.returncode == 0 else '❌'
    if p.returncode != 0:
        any_error = True
    print(f"\n  [GPU {gpu_id}]  exit={p.returncode}  {status}")
    if p.returncode != 0:
        print(f"  ── last 30 lines of {log_path} ──")
        with open(log_path) as lf:
            lines = lf.readlines()
            for line in lines[-30:]:
                print(f"   {line.rstrip()}")

print(f"\n  Time      : {elapsed/60:.1f} min  ({elapsed/max(len(sr_done),1):.1f}s per image)")
print(f"  Generated : {len(sr_done)} / {len(test_imgs)} images")
print(f"{'='*68}")

if len(sr_done) == 0:
    raise RuntimeError("❌  Zero images generated — check logs above")

print("\n  ✅ Inference complete — proceed to Cell 5 (build res.zip)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║ CELL 5 — Size verification + build res.zip                       ║
# ╚══════════════════════════════════════════════════════════════════╝
import zipfile, numpy as np
from PIL import Image

sr_done = sorted(glob.glob(f'{SUBMIT_DIR}/*.png'))

print("  Output size verification (all images):")
bad = []
for p in sr_done:
    stem   = os.path.splitext(os.path.basename(p))[0]
    lr_p   = os.path.join(TEST_LR_DIR, f'{stem}.png')
    sr_img = Image.open(p)
    sw, sh = sr_img.size
    if os.path.exists(lr_p):
        lw, lh = Image.open(lr_p).size
        if sw != lw * 4 or sh != lh * 4:
            bad.append(f"{stem}: got {sw}×{sh}, expected {lw*4}×{lh*4}")

if bad:
    print(f"  ⚠  {len(bad)} size mismatches:")
    for b in bad[:5]:
        print(f"     {b}")
else:
    print(f"  ✅ All {len(sr_done)} images have correct 4× dimensions")

# ── readme.txt ───────────────────────────────────────────────────────
runtime_per = elapsed / max(len(sr_done), 1)
readme = f"""runtime per image [s] : {runtime_per:.2f}
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description :
Championship SR — CompleteEnhancedFusionSR (direct inference, no TTA)
  Experts (frozen ~131M params): DRCT-L, GRL-B, NAFNet-SIDD-w64, MambaIR-180
  Fusion  (trainable ~1.2M params): 7-phase frequency-guided fusion
  Phase 2 : Multi-Domain Frequency Decomposition (DCT+DWT+FFT, 9 bands)
  Phase 3 : Cross-Band Attention + Large Kernel Attention (k=21)
  Phase 4 : Collaborative Feature Learning (cross-expert, 8 heads)
  Phase 5 : Hierarchical Multi-Resolution Fusion (64→128→256)
  Phase 6 : Dynamic Expert Selection (per-pixel difficulty gating)
  Phase 7 : Deep CNN Refinement (6-layer 128ch) + Laplacian Edge Enhancement
Training   : DF2K (DIV2K+Flickr2K), AdamW lr=1e-4, 3-stage loss curriculum
Checkpoint : {os.path.basename(CHECKPOINT)}
Scale      : 4x  |  Inference: FP32, single pass, NVIDIA GPU
"""

readme_path = f'{SUBMIT_DIR}/readme.txt'
with open(readme_path, 'w') as f:
    f.write(readme)

# ── Build res.zip (FLAT — NTIRE requirement: no subdirectories) ───────
print(f"\n  Building {FINAL_ZIP} ...")
if os.path.exists(FINAL_ZIP):
    os.remove(FINAL_ZIP)

with zipfile.ZipFile(FINAL_ZIP, 'w', zipfile.ZIP_STORED) as zf:
    for img in sr_done:
        zf.write(img, arcname=os.path.basename(img))
    zf.write(readme_path, arcname='readme.txt')

# ── Verify zip ────────────────────────────────────────────────────────
with zipfile.ZipFile(FINAL_ZIP, 'r') as zf:
    names   = zf.namelist()
    folders = [n for n in names if '/' in n]

zip_mb = os.path.getsize(FINAL_ZIP) / 1_048_576

print(f"\n{'='*68}")
print(f"  {'✅' if not folders else '❌  HAS SUBDIRS — FIX arcname'}")
print(f"  Path     : {FINAL_ZIP}")
print(f"  Size     : {zip_mb:.1f} MB")
print(f"  Images   : {len(sr_done)} / {len(test_imgs)} expected")
print(f"  Flat     : {'✅' if not folders else '❌'}")
print(f"  readme   : {'✅' if 'readme.txt' in names else '❌'}")
print(f"  Sample   : {names[:4]}")
print(f"{'='*68}")
print(f"\n  🎉 DONE — download /kaggle/working/res.zip and upload to CodaBench")
