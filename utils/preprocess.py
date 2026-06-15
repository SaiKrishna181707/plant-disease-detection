# =============================================================================
#  utils/preprocess.py — Image preprocessing utilities
#
#  Provides standalone functions used by the Flask app and predict.py to
#  preprocess a single image from disk or from raw bytes.
# =============================================================================

import os
import sys
import io
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, INPUT_SHAPE


def preprocess_image_path(image_path: str) -> np.ndarray:
    """
    Load an image from disk, resize, normalise, and add batch dimension.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        numpy array of shape (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        with pixel values in [0, 1].
    """
    import cv2

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)          # (1, H, W, C)


def preprocess_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Preprocess an image directly from raw bytes (from Flask file upload).

    Args:
        raw_bytes: Raw image bytes.

    Returns:
        numpy array of shape (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).
    """
    import cv2

    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_batch(image_paths: list) -> np.ndarray:
    """
    Preprocess a list of image paths into a single batched numpy array.

    Args:
        image_paths: List of file paths.

    Returns:
        numpy array of shape (N, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).
    """
    batch = [preprocess_image_path(p)[0] for p in image_paths]
    return np.stack(batch, axis=0)


def validate_image_file(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    from config import ALLOWED_EXTS
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    return ext in ALLOWED_EXTS
