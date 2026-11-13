"""Suspicious region visualization module."""
import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Color scheme: BGR format for OpenCV
COLOR_GREEN  = (34, 139, 34)
COLOR_YELLOW = (0, 200, 200)
COLOR_RED    = (0, 0, 220)
COLOR_ORANGE = (0, 140, 255)


def draw_findings_overlay(
    cv_img: np.ndarray,
    image_result: Dict[str, Any],
    math_result: Dict[str, Any],
) -> np.ndarray:
    """Draw bounding boxes and labels on the image for detected findings.

    Returns a copy of the image with overlays drawn.
    """
    overlay = cv_img.copy()
    h, w = overlay.shape[:2]
    annotations: List[Tuple[str, Tuple, str]] = []  # (label, color, description)

    noise_blocks = image_result.get('noise', {}).get('suspicious_blocks', [])
    for block in noise_blocks[:8]:
        x, y = block['x'], block['y']
        size = block['size']
        x2, y2 = min(x + size, w - 1), min(y + size, h - 1)
        cv2.rectangle(overlay, (x, y), (x2, y2), COLOR_YELLOW, 2)
        label = 'Noise Anomaly'
        cv2.putText(overlay, label, (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA)

    copy_move_pairs = image_result.get('copy_move', {}).get('duplicate_pairs', [])
    for x1, y1, x2, y2 in copy_move_pairs[:4]:
        # Scale back if needed (copy-move uses downscaled coords)
        scale = image_result.get('_scale', 1.0)
        x1s, y1s = int(x1 / scale), int(y1 / scale)
        x2s, y2s = int(x2 / scale), int(y2 / scale)
        size = int(32 / scale)
        cv2.rectangle(overlay, (x1s, y1s), (x1s + size, y1s + size), COLOR_RED, 2)
        cv2.rectangle(overlay, (x2s, y2s), (x2s + size, y2s + size), COLOR_RED, 2)
        cv2.line(overlay, (x1s + size // 2, y1s + size // 2),
                 (x2s + size // 2, y2s + size // 2), COLOR_RED, 1)
        cv2.putText(overlay, 'Copy-Move?', (x1s, max(y1s - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RED, 1, cv2.LINE_AA)

    # Math findings – mark bottom region of image
    if math_result.get('total_mismatch', False):
        y_start = max(0, h - 60)
        cv2.rectangle(overlay, (0, y_start), (w - 1, h - 1), COLOR_ORANGE, 3)
        cv2.putText(overlay, 'Total Mismatch Detected', (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 1, cv2.LINE_AA)

    # Blend overlay with original for transparency effect
    result = cv2.addWeighted(overlay, 0.7, cv_img, 0.3, 0)

    # Add legend
    legend_items = [
        ('Yellow box: Noise anomaly region', COLOR_YELLOW),
        ('Red box: Possible copy-move region', COLOR_RED),
        ('Orange box: Math mismatch area', COLOR_ORANGE),
    ]
    legend_y = 20
    for text, color in legend_items:
        cv2.rectangle(result, (8, legend_y - 10), (20, legend_y + 2), color, -1)
        cv2.putText(result, text, (25, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 50), 1, cv2.LINE_AA)
        legend_y += 18

    return result


def create_ela_visualization(ela_array: np.ndarray) -> np.ndarray:
    """Convert ELA difference array into a coloured heatmap."""
    if ela_array is None or ela_array.size == 0:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY) if len(ela_array.shape) == 3 else ela_array
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return heatmap


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    ela: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a side-by-side comparison image."""
    h = max(original.shape[0], annotated.shape[0])
    # Resize both to same height
    orig_r = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    ann_r  = cv2.resize(annotated, (int(annotated.shape[1] * h / annotated.shape[0]), h))

    divider = np.full((h, 4, 3), 200, dtype=np.uint8)

    panels = [orig_r, divider, ann_r]
    if ela is not None:
        ela_r = cv2.resize(ela, (int(ela.shape[1] * h / ela.shape[0]), h))
        panels += [divider, ela_r]

    combined = np.hstack(panels)

    # Add panel labels
    label_y = 25
    for i, label in enumerate(['Original', 'Analysis Overlay', 'ELA Heatmap'][:len(panels)]):
        x_offsets = [0, orig_r.shape[1] + 4, orig_r.shape[1] + 4 + ann_r.shape[1] + 4]
        if i < len(x_offsets):
            cv2.putText(combined, label, (x_offsets[i] + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return combined
