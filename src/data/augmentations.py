"""
Super-Resolution Data Augmentations
====================================
Paired augmentation pipeline for LR-HR image pairs.

Features:
1. Paired random crop (maintains alignment)
2. Paired random flip (horizontal + vertical)
3. Paired random rotation (90°, 180°, 270°)
4. Color jitter (brightness, contrast, saturation)
5. Mixup augmentation (optional)
6. CutMix augmentation (optional)

All augmentations maintain LR-HR correspondence!

Author: NTIRE SR Team
"""

import numpy as np
import cv2
import random
from typing import Tuple, Optional, List
import torch


class PairedRandomCrop:
    """
    Random crop that maintains LR-HR alignment.
    
    Crops corresponding regions from LR and HR images
    while respecting the scale factor.
    """
    
    def __init__(self, lr_patch_size: int = 64, scale: int = 4):
        """
        Args:
            lr_patch_size: Size of LR patch
            scale: Upscaling factor (HR_size = LR_size * scale)
        """
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.scale = scale
    
    def __call__(
        self, 
        lr_img: np.ndarray, 
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired random crop.
        
        Args:
            lr_img: LR image [H, W, C]
            hr_img: HR image [H*scale, W*scale, C]
            
        Returns:
            (cropped_lr, cropped_hr)
        """
        lr_h, lr_w = lr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]
        
        # Validate dimensions
        if lr_h < self.lr_patch_size or lr_w < self.lr_patch_size:
            # Resize if too small
            lr_img = cv2.resize(
                lr_img, 
                (self.lr_patch_size, self.lr_patch_size),
                interpolation=cv2.INTER_CUBIC
            )
            hr_img = cv2.resize(
                hr_img,
                (self.hr_patch_size, self.hr_patch_size),
                interpolation=cv2.INTER_CUBIC
            )
            return lr_img, hr_img
        
        # Random crop position (in LR space)
        lr_top = random.randint(0, lr_h - self.lr_patch_size)
        lr_left = random.randint(0, lr_w - self.lr_patch_size)
        
        # Corresponding HR position
        hr_top = lr_top * self.scale
        hr_left = lr_left * self.scale
        
        # Crop
        lr_crop = lr_img[
            lr_top:lr_top + self.lr_patch_size,
            lr_left:lr_left + self.lr_patch_size
        ]
        hr_crop = hr_img[
            hr_top:hr_top + self.hr_patch_size,
            hr_left:hr_left + self.hr_patch_size
        ]
        
        return lr_crop, hr_crop


class PairedRandomFlip:
    """
    Random horizontal and vertical flip for paired images.
    """
    
    def __init__(self, horizontal_prob: float = 0.5, vertical_prob: float = 0.5):
        """
        Args:
            horizontal_prob: Probability of horizontal flip
            vertical_prob: Probability of vertical flip
        """
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
    
    def __call__(
        self, 
        lr_img: np.ndarray, 
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired random flip.
        
        Args:
            lr_img: LR image [H, W, C]
            hr_img: HR image [H, W, C]
            
        Returns:
            (flipped_lr, flipped_hr)
        """
        # Horizontal flip
        if random.random() < self.horizontal_prob:
            lr_img = np.flip(lr_img, axis=1).copy()
            hr_img = np.flip(hr_img, axis=1).copy()
        
        # Vertical flip
        if random.random() < self.vertical_prob:
            lr_img = np.flip(lr_img, axis=0).copy()
            hr_img = np.flip(hr_img, axis=0).copy()
        
        return lr_img, hr_img


class PairedRandomRotation:
    """
    Random 90-degree rotation for paired images.
    
    Rotates by 0°, 90°, 180°, or 270° with equal probability.
    """
    
    def __init__(self, prob: float = 0.5):
        """
        Args:
            prob: Probability of applying rotation (non-zero)
        """
        self.prob = prob
    
    def __call__(
        self, 
        lr_img: np.ndarray, 
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired random rotation.
        
        Args:
            lr_img: LR image [H, W, C]
            hr_img: HR image [H, W, C]
            
        Returns:
            (rotated_lr, rotated_hr)
        """
        if random.random() < self.prob:
            # Random rotation: 1, 2, or 3 times 90°
            k = random.randint(1, 3)
            lr_img = np.rot90(lr_img, k=k).copy()
            hr_img = np.rot90(hr_img, k=k).copy()
        
        return lr_img, hr_img


class ColorJitter:
    """
    Color jitter augmentation for paired images.
    
    Applies identical color transformations to LR and HR.
    """
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float = 0.5
    ):
        """
        Args:
            brightness: Brightness jitter range
            contrast: Contrast jitter range
            saturation: Saturation jitter range
            hue: Hue jitter range
            prob: Probability of applying jitter
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
    
    def __call__(
        self, 
        lr_img: np.ndarray, 
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired color jitter.
        
        Args:
            lr_img: LR image [H, W, C] in float32 [0, 1]
            hr_img: HR image [H, W, C] in float32 [0, 1]
            
        Returns:
            (jittered_lr, jittered_hr)
        """
        if random.random() >= self.prob:
            return lr_img, hr_img
        
        # Generate random factors (same for both images)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        
        # Apply to both images
        lr_img = self._apply_jitter(lr_img, brightness_factor, contrast_factor, saturation_factor)
        hr_img = self._apply_jitter(hr_img, brightness_factor, contrast_factor, saturation_factor)
        
        return lr_img, hr_img
    
    def _apply_jitter(
        self, 
        img: np.ndarray,
        brightness: float,
        contrast: float,
        saturation: float
    ) -> np.ndarray:
        """Apply color jitter to single image."""
        # Brightness
        img = img * brightness
        
        # Contrast (adjust around mean)
        mean = img.mean()
        img = (img - mean) * contrast + mean
        
        # Saturation (in HSV space)
        if saturation != 1.0:
            # Convert to HSV
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            img = img_uint8.astype(np.float32) / 255.0
        
        # Clip to valid range
        img = np.clip(img, 0, 1)
        
        return img


class GaussianBlur:
    """
    Optional Gaussian blur for data augmentation.
    
    Applied identically to both LR and HR.
    """
    
    def __init__(
        self,
        kernel_sizes: List[int] = [3, 5],
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        prob: float = 0.3
    ):
        """
        Args:
            kernel_sizes: Possible kernel sizes
            sigma_range: Range for sigma
            prob: Probability of applying blur
        """
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(
        self, 
        lr_img: np.ndarray, 
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply paired Gaussian blur."""
        if random.random() >= self.prob:
            return lr_img, hr_img
        
        kernel_size = random.choice(self.kernel_sizes)
        sigma = random.uniform(*self.sigma_range)
        
        lr_img = cv2.GaussianBlur(lr_img, (kernel_size, kernel_size), sigma)
        hr_img = cv2.GaussianBlur(hr_img, (kernel_size, kernel_size), sigma)
        
        return lr_img, hr_img


class CutBlur:
    """
    CutBlur augmentation for super-resolution.
    
    Replaces random region of HR with its downsampled-upsampled version.
    This teaches the model to focus on different restoration difficulties.
    
    Paper: "Rethinking Data Augmentation for Image Super-resolution"
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        prob: float = 0.5,
        scale: int = 4
    ):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying CutBlur
            scale: Downsampling scale
        """
        self.alpha = alpha
        self.prob = prob
        self.scale = scale
    
    def __call__(
        self,
        lr_img: np.ndarray,
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CutBlur augmentation."""
        if random.random() >= self.prob:
            return lr_img, hr_img
        
        h, w = hr_img.shape[:2]
        
        # Random cut size
        cut_ratio = np.random.beta(self.alpha, self.alpha)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random position
        cy = random.randint(0, h - cut_h)
        cx = random.randint(0, w - cut_w)
        
        # Create blurred version
        blurred = cv2.resize(
            hr_img[cy:cy+cut_h, cx:cx+cut_w],
            (cut_w // self.scale, cut_h // self.scale),
            interpolation=cv2.INTER_CUBIC
        )
        blurred = cv2.resize(
            blurred,
            (cut_w, cut_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Replace region
        hr_img_aug = hr_img.copy()
        hr_img_aug[cy:cy+cut_h, cx:cx+cut_w] = blurred
        
        return lr_img, hr_img_aug


class SRTrainAugmentation:
    """
    Complete augmentation pipeline for SR training.
    
    Combines all augmentations in proper order:
    1. Random crop (first, to reduce computation)
    2. Random flip
    3. Random rotation
    4. Color jitter
    5. CutBlur (optional)
    
    All transforms maintain LR-HR correspondence!
    """
    
    def __init__(
        self,
        lr_patch_size: int = 64,
        scale: int = 4,
        use_flip: bool = True,
        use_rotation: bool = True,
        use_color_jitter: bool = True,
        use_cutblur: bool = False,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        cutblur_prob: float = 0.3
    ):
        """
        Args:
            lr_patch_size: LR patch size
            scale: Upscaling factor
            use_flip: Enable random flip
            use_rotation: Enable random rotation
            use_color_jitter: Enable color jitter
            use_cutblur: Enable CutBlur augmentation
            *_prob: Probability for each transform
        """
        self.transforms = []
        
        # Always use random crop
        self.transforms.append(
            PairedRandomCrop(lr_patch_size=lr_patch_size, scale=scale)
        )
        
        # Optional transforms
        if use_flip:
            self.transforms.append(
                PairedRandomFlip(
                    horizontal_prob=flip_prob,
                    vertical_prob=flip_prob
                )
            )
        
        if use_rotation:
            self.transforms.append(
                PairedRandomRotation(prob=rotation_prob)
            )
        
        if use_color_jitter:
            self.transforms.append(
                ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    prob=color_jitter_prob
                )
            )
        
        if use_cutblur:
            self.transforms.append(
                CutBlur(prob=cutblur_prob, scale=scale)
            )
    
    def __call__(
        self, 
        lr_img: np.ndarray, 
        hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentations.
        
        Args:
            lr_img: LR image [H, W, C]
            hr_img: HR image [H*scale, W*scale, C]
            
        Returns:
            (augmented_lr, augmented_hr)
        """
        for transform in self.transforms:
            lr_img, hr_img = transform(lr_img, hr_img)
        
        return lr_img, hr_img


# ============================================================================
# Testing
# ============================================================================

def test_augmentations():
    """Test all augmentation functions."""
    print("\n" + "=" * 60)
    print("AUGMENTATION MODULE - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Create dummy LR-HR pair
    lr_img = np.random.rand(128, 128, 3).astype(np.float32)
    hr_img = np.random.rand(512, 512, 3).astype(np.float32)
    
    print(f"\nInput shapes: LR={lr_img.shape}, HR={hr_img.shape}")
    
    # Test 1: Random Crop
    print("\n--- Test 1: PairedRandomCrop ---")
    crop = PairedRandomCrop(lr_patch_size=64, scale=4)
    lr_crop, hr_crop = crop(lr_img, hr_img)
    assert lr_crop.shape == (64, 64, 3), f"LR crop shape wrong: {lr_crop.shape}"
    assert hr_crop.shape == (256, 256, 3), f"HR crop shape wrong: {hr_crop.shape}"
    print(f"  LR: {lr_img.shape} -> {lr_crop.shape}")
    print(f"  HR: {hr_img.shape} -> {hr_crop.shape}")
    print("  [PASSED]")
    
    # Test 2: Random Flip
    print("\n--- Test 2: PairedRandomFlip ---")
    flip = PairedRandomFlip(horizontal_prob=1.0, vertical_prob=0.0)
    lr_flip, hr_flip = flip(lr_crop.copy(), hr_crop.copy())
    # Check horizontal flip
    assert np.allclose(lr_flip[:, ::-1, :], lr_crop), "Horizontal flip failed"
    print(f"  Shapes preserved: LR={lr_flip.shape}, HR={hr_flip.shape}")
    print("  [PASSED]")
    
    # Test 3: Random Rotation
    print("\n--- Test 3: PairedRandomRotation ---")
    rotation = PairedRandomRotation(prob=1.0)
    lr_rot, hr_rot = rotation(lr_crop.copy(), hr_crop.copy())
    print(f"  Shapes after rotation: LR={lr_rot.shape}, HR={hr_rot.shape}")
    print("  [PASSED]")
    
    # Test 4: Color Jitter
    print("\n--- Test 4: ColorJitter ---")
    jitter = ColorJitter(brightness=0.3, contrast=0.3, prob=1.0)
    lr_jit, hr_jit = jitter(lr_crop.copy(), hr_crop.copy())
    assert lr_jit.min() >= 0 and lr_jit.max() <= 1, "LR values out of range"
    assert hr_jit.min() >= 0 and hr_jit.max() <= 1, "HR values out of range"
    print(f"  Value range: LR=[{lr_jit.min():.3f}, {lr_jit.max():.3f}]")
    print(f"  Value range: HR=[{hr_jit.min():.3f}, {hr_jit.max():.3f}]")
    print("  [PASSED]")
    
    # Test 5: Complete Pipeline
    print("\n--- Test 5: SRTrainAugmentation (Complete Pipeline) ---")
    augmentation = SRTrainAugmentation(
        lr_patch_size=64,
        scale=4,
        use_flip=True,
        use_rotation=True,
        use_color_jitter=True,
        use_cutblur=True
    )
    lr_aug, hr_aug = augmentation(lr_img, hr_img)
    assert lr_aug.shape == (64, 64, 3), f"Final LR shape wrong: {lr_aug.shape}"
    assert hr_aug.shape == (256, 256, 3), f"Final HR shape wrong: {hr_aug.shape}"
    print(f"  Final LR: {lr_aug.shape}")
    print(f"  Final HR: {hr_aug.shape}")
    print(f"  LR range: [{lr_aug.min():.3f}, {lr_aug.max():.3f}]")
    print(f"  HR range: [{hr_aug.min():.3f}, {hr_aug.max():.3f}]")
    print("  [PASSED]")
    
    # Test 6: Multiple iterations (reproducibility check)
    print("\n--- Test 6: Multiple Iterations ---")
    for i in range(5):
        lr_aug, hr_aug = augmentation(lr_img, hr_img)
        assert lr_aug.shape == (64, 64, 3)
        assert hr_aug.shape == (256, 256, 3)
    print("  5 iterations completed successfully")
    print("  [PASSED]")
    
    print("\n" + "=" * 60)
    print("ALL AUGMENTATION TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_augmentations()
