"""Document preprocessing utilities."""
import io
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

MAX_DIM = 2400  # max dimension for processing


def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """Load an image from bytes into a BGR numpy array."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.error('Failed to load image from bytes: %s', exc)
        return None


def load_pil_image(data: bytes) -> Optional[Image.Image]:
    """Load a PIL image from bytes."""
    try:
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as exc:
        logger.error('Failed to load PIL image: %s', exc)
        return None


def resize_for_processing(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Resize image so its largest dimension is max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Apply basic enhancements to improve OCR accuracy."""
    # Increase sharpness
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    # Increase contrast slightly
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
    return pil_img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to BGR numpy array."""
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL image."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_to_bytes(img: np.ndarray, ext: str = '.png') -> bytes:
    """Encode CV2 image to bytes."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise ValueError('Failed to encode image')
    return buf.tobytes()


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to float32 in [0, 1]."""
    return img.astype(np.float32) / 255.0
