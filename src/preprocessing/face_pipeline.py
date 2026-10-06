"""
Unified Preprocessing Pipeline for Facial Emotion Recognition (Experiment A4)
Shared across Training, Benchmark, and Inference.
Parity verified in docs/benchmark_results/preprocessing_verification.json.
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def decode_and_convert_rgb(image_bytes: bytes) -> np.ndarray:
    """
    Decodes image byte array and ensures strict sRGB channel order.
    cv2.imdecode returns BGR; this function converts BGR to RGB.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for image decoding")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image from buffer")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def align_and_crop_face(
    img_rgb: np.ndarray,
    face_box: Tuple[int, int, int, int],
    margin: float = 1.30,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Applies 1.30 margin expansion around face bounding box and bilinear resize to 224x224.
    """
    h, w = img_rgb.shape[:2]
    bx, by, bw, bh = face_box

    mw = int(bw * (margin - 1.0) / 2)
    mh = int(bh * (margin - 1.0) / 2)

    x1 = max(0, bx - mw)
    y1 = max(0, by - mh)
    x2 = min(w, bx + bw + mw)
    y2 = min(h, by + bh + mh)

    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_rgb

    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

def normalize_imagenet_tensor(aligned_face_rgb: np.ndarray) -> np.ndarray:
    """
    Converts (224, 224, 3) RGB uint8 to (1, 3, 224, 224) float32 ImageNet normalized tensor.
    """
    tensor = aligned_face_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # (3, 224, 224)
    tensor = np.expand_dims(tensor, axis=0)   # (1, 3, 224, 224)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.astype(np.float32)
