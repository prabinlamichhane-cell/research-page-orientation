"""
Shared preprocessing pipeline for PP-LCNet_x1_0_doc_ori.
Must be used identically across ALL runtime experiments (PaddlePaddle, ONNX, Optimum).

Pipeline sourced from models/PP-LCNet_x1_0_doc_ori_infer/config.json:
  1. ResizeImage(resize_short=256)  — resize so shorter side = 256, keep aspect ratio
  2. CropImage(size=224)            — center crop to 224×224
  3. NormalizeImage(mean, std)      — ImageNet stats, scale=1/255
  4. ToCHWImage                     — HWC → CHW
  + expand_dims(0)                  — add batch dim → (1, 3, 224, 224)
"""

from pathlib import Path

import cv2
import numpy as np


LABEL_TO_DEGREES = {0: 0, 1: 90, 2: 180, 3: 270}
DEGREES_TO_LABEL = {v: k for k, v in LABEL_TO_DEGREES.items()}

RESIZE_SHORT = 256
CROP_SIZE    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_short(img: np.ndarray, size: int) -> np.ndarray:
    """Resize so the shorter side equals `size`, keeping aspect ratio."""
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Center crop to size×size."""
    h, w = img.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """
    Load an image and return a preprocessed (1, 3, 224, 224) float32 array.
    Compatible with PaddlePaddle, ONNX Runtime, and Optimum ORT.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # OpenCV loads BGR → convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. resize_short(256)
    img = _resize_short(img, RESIZE_SHORT)

    # 2. center_crop(224)
    img = _center_crop(img, CROP_SIZE)

    # 3. normalize: scale to [0,1] then ImageNet stats
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    # 4. HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


def rotate_image(image_path: str | Path, degrees: int) -> np.ndarray:
    """Return a rotated image array (HWC BGR uint8) for dataset generation."""
    assert degrees in (0, 90, 180, 270), "degrees must be 0, 90, 180, or 270"
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rotate_flags = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    flag = rotate_flags[degrees]
    return img if flag is None else cv2.rotate(img, flag)
