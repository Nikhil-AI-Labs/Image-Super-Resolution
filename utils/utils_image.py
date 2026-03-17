"""
Image utility functions for NTIRE 2026 Image SR x4.
Based on the official NTIRE2026_ImageSR_x4 baseline by Zheng Chen.
"""

import os
import math
import numpy as np
import torch
import cv2
from datetime import datetime


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


# =========================================================================
# Directory utilities
# =========================================================================
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_image_paths(dataroot):
    paths = None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


# =========================================================================
# Image I/O
# =========================================================================
def read_img(path):
    """Read image as float32 HWC BGR [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def imread_uint(path, n_channels=3):
    """Read image as uint8 HxWxC (RGB or Gray)."""
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imsave(img, img_path):
    """Save uint8 image (RGB or Gray) to file."""
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]  # RGB → BGR
    cv2.imwrite(img_path, img)


# =========================================================================
# Type conversions
# =========================================================================
def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def uint2tensor4(img, data_range=1.0):
    """uint8 HxWxC → float32 [1,C,H,W]."""
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(
        np.ascontiguousarray(img)
    ).permute(2, 0, 1).float().div(255. / data_range).unsqueeze(0)


def uint2tensor3(img):
    """uint8 HxWxC → float32 [C,H,W]."""
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(
        np.ascontiguousarray(img)
    ).permute(2, 0, 1).float().div(255.)


def tensor2uint(img, data_range=1.0):
    """float32 tensor → uint8 numpy HxWxC."""
    img = img.data.squeeze().float().clamp_(0, 1 * data_range).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0 / data_range).round())


def single2tensor4(img):
    return torch.from_numpy(
        np.ascontiguousarray(img)
    ).permute(2, 0, 1).float().unsqueeze(0)


def single2tensor3(img):
    return torch.from_numpy(
        np.ascontiguousarray(img)
    ).permute(2, 0, 1).float()


# =========================================================================
# PSNR / SSIM (for eval.py)
# =========================================================================
def cal_psnr_ssim(output_image_path, target_image_path, crop_border=4, test_y_channel=True):
    """Calculate PSNR and SSIM between two image files."""
    output_img = imread_uint(output_image_path, n_channels=3)
    target_img = imread_uint(target_image_path, n_channels=3)

    # Ensure same size
    h = min(output_img.shape[0], target_img.shape[0])
    w = min(output_img.shape[1], target_img.shape[1])
    output_img = output_img[:h, :w, :]
    target_img = target_img[:h, :w, :]

    # Crop border
    if crop_border > 0:
        output_img = output_img[crop_border:-crop_border, crop_border:-crop_border, :]
        target_img = target_img[crop_border:-crop_border, crop_border:-crop_border, :]

    # Convert to Y channel
    if test_y_channel:
        output_y = cv2.cvtColor(output_img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float64)
        target_y = cv2.cvtColor(target_img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float64)
    else:
        output_y = output_img.astype(np.float64)
        target_y = target_img.astype(np.float64)

    # PSNR
    mse = np.mean((output_y - target_y) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim
        if output_y.ndim == 2:
            ssim_val = ssim(output_y, target_y, data_range=255)
        else:
            ssim_val = ssim(output_y, target_y, data_range=255, channel_axis=2)
    except ImportError:
        ssim_val = 0.0

    return psnr, ssim_val
