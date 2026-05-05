"""
Shared preprocessing pipeline for PP-LCNet_x1_0_doc_ori.
Must be used identically across all runtime experiments.

PP-StructureV3 orientation model expects:
  - Resize to 224x224
  - BGR channel order (OpenCV default)
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)
  - NCHW layout, float32, batch dim added
"""

from pathlib import Path

import cv2
import numpy as np


LABEL_TO_DEGREES = {0: 0, 1: 90, 2: 180, 3: 270}
DEGREES_TO_LABEL = {v: k for k, v in LABEL_TO_DEGREES.items()}

INPUT_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """
    Load an image and return a preprocessed (1, 3, 224, 224) float32 array.
    Compatible with PaddlePaddle, ONNX Runtime, and Optimum ORT.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0

    # OpenCV loads BGR; convert to RGB for ImageNet normalization
    img = img[:, :, ::-1]

    img = (img - MEAN) / STD

    # HWC -> CHW -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


def rotate_image(image_path: str | Path, degrees: int) -> np.ndarray:
    """Return a rotated image array (HWC, uint8) for dataset generation."""
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
