"""Image manipulation analysis using OpenCV and Pillow.

All findings are described as indicators, not definitive proof.
"""
import io
import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# A. Error Level Analysis (ELA)
# ─────────────────────────────────────────────

def compute_ela(
    pil_img: Image.Image, quality: int = 75
) -> Tuple[np.ndarray, float]:
    """Compute a basic Error Level Analysis approximation.
    
    Re-saves the image at a reduced quality and computes the absolute
    difference. Regions with unusually high differences may indicate
    different compression history. This is an approximation only.
    """
    buf = io.BytesIO()
    pil_img = pil_img.convert('RGB')
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')

    orig_arr = np.array(pil_img, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    ela = np.abs(orig_arr - recomp_arr)

    # Normalize for visualization
    ela_norm = (ela / max(ela.max(), 1e-5) * 255).astype(np.uint8)
    compression_diff = float(ela.mean())
    return ela_norm, compression_diff


# ─────────────────────────────────────────────
# B. Noise Analysis
# ─────────────────────────────────────────────

def analyze_noise(img: np.ndarray) -> Dict[str, Any]:
    """Analyze local noise variance across image blocks.
    
    Identifies blocks with significantly different noise characteristics
    compared to the overall image median.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    block_size = max(64, min(h, w) // 8)
    variances = []
    suspicious_blocks = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size].astype(np.float32)
            lap = cv2.Laplacian(block, cv2.CV_32F)
            var = float(lap.var())
            variances.append((var, x, y, block_size))

    if not variances:
        return {
            'noise_variance': 0.0,
            'suspicious_blocks': [],
            'score': 0.0,
        }

    all_vars = [v[0] for v in variances]
    median_var = float(np.median(all_vars))
    std_var = float(np.std(all_vars)) or 1.0
    threshold = median_var + 2.5 * std_var

    suspicious_blocks = [
        {'x': x, 'y': y, 'size': bs, 'variance': var}
        for var, x, y, bs in variances
        if var > threshold
    ]

    return {
        'noise_variance': float(np.var(all_vars)),
        'median_block_variance': median_var,
        'suspicious_blocks': suspicious_blocks,
        'score': min(1.0, len(suspicious_blocks) / max(len(variances), 1)),
    }


# ─────────────────────────────────────────────
# C. Edge / Font Consistency
# ─────────────────────────────────────────────

def analyze_edge_consistency(img: np.ndarray) -> Dict[str, Any]:
    """Analyze edge sharpness consistency across the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    half_w = w // 2

    left_density = float(edges[:, :half_w].sum()) / max(half_w * h, 1)
    right_density = float(edges[:, half_w:].sum()) / max(half_w * h, 1)
    diff = abs(left_density - right_density)
    score = min(1.0, diff * 20)

    # Compute overall sharpness
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        'left_edge_density': left_density,
        'right_edge_density': right_density,
        'edge_asymmetry': diff,
        'sharpness': laplacian_var,
        'score': score,
    }


# ─────────────────────────────────────────────
# D. Copy-Move Detection (simplified)
# ─────────────────────────────────────────────

def detect_copy_move(img: np.ndarray) -> Dict[str, Any]:
    """Basic copy-move detection using block matching.
    
    This is a simplified approach that compares image blocks for similarity.
    It is not a forensically validated implementation — use results as
    soft indicators only.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape

    # Resize for speed
    max_dim = 400
    scale = min(1.0, max_dim / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    sh, sw = small.shape

    block_size = 32
    step = 16
    blocks: List[Tuple] = []

    for y in range(0, sh - block_size, step):
        for x in range(0, sw - block_size, step):
            block = small[y:y + block_size, x:x + block_size]
            # Use DCT as feature
            dct = cv2.dct(block.astype(np.float32))
            feature = dct[:8, :8].flatten()
            blocks.append((feature, x, y))

    # Find pairs with high similarity
    duplicate_pairs = []
    checked = set()
    for i, (f1, x1, y1) in enumerate(blocks):
        if i in checked:
            continue
        for j, (f2, x2, y2) in enumerate(blocks):
            if j <= i or j in checked:
                continue
            dist = float(np.linalg.norm(f1 - f2))
            spatial_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist < 15 and spatial_dist > block_size * 2:
                duplicate_pairs.append((x1, y1, x2, y2))
                checked.add(i)
                checked.add(j)
                if len(duplicate_pairs) >= 5:
                    break
        if len(duplicate_pairs) >= 5:
            break

    score = min(1.0, len(duplicate_pairs) * 0.25)
    return {
        'duplicate_pairs': duplicate_pairs,
        'pair_count': len(duplicate_pairs),
        'score': score,
        'details': (
            [f'{len(duplicate_pairs)} potentially duplicated region(s) detected (basic block analysis).']
            if duplicate_pairs else []
        ),
    }


# ─────────────────────────────────────────────
# E. Metadata Analysis
# ─────────────────────────────────────────────

def analyze_image_metadata(pil_img: Image.Image) -> Dict[str, Any]:
    """Extract and analyze EXIF/image metadata."""
    meta: Dict[str, Any] = {
        'raw': {},
        'anomalies': [],
        'score': 0.0,
    }
    try:
        exif_data = pil_img._getexif() if hasattr(pil_img, '_getexif') else None
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                meta['raw'][tag] = str(value)[:200]

            # Check for software tag suggesting editing
            software = meta['raw'].get('Software', '')
            if any(
                kw in software.lower()
                for kw in ['photoshop', 'gimp', 'paint.net', 'affinity', 'canva']
            ):
                meta['anomalies'].append(
                    f'Image editing software detected in metadata: {software}'
                )

            make = meta['raw'].get('Make', '')
            model = meta['raw'].get('Model', '')
            if not make and not model:
                meta['anomalies'].append(
                    'No camera make/model in EXIF data. '
                    'Document may have been created digitally or edited.'
                )
        else:
            meta['anomalies'].append('No EXIF metadata found in image.')
    except Exception as exc:
        logger.debug('Metadata analysis error: %s', exc)
        meta['anomalies'].append('Could not read image metadata.')

    meta['score'] = min(1.0, len(meta['anomalies']) * 0.2)
    return meta


# ─────────────────────────────────────────────
# Combined runner
# ─────────────────────────────────────────────

def run_image_analysis(
    pil_img: Image.Image,
    cv_img: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run all image-analysis checks and return combined results."""
    if cv_img is None:
        arr = np.array(pil_img.convert('RGB'))
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Resize for performance
    h, w = cv_img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    ela_arr, compression_diff = compute_ela(pil_img)
    noise_result = analyze_noise(cv_img)
    edge_result = analyze_edge_consistency(cv_img)
    copy_move_result = detect_copy_move(cv_img)
    metadata_result = analyze_image_metadata(pil_img)

    # Aggregate score
    composite_score = (
        min(1.0, compression_diff / 30) * 0.25
        + noise_result['score'] * 0.25
        + edge_result['score'] * 0.15
        + copy_move_result['score'] * 0.25
        + metadata_result['score'] * 0.10
    )

    findings = []
    if compression_diff > 8:
        findings.append({
            'type': 'ela_anomaly',
            'severity': 'medium' if compression_diff < 20 else 'high',
            'message': (
                f'Region(s) with different compression characteristics detected '
                f'(ELA mean difference: {compression_diff:.1f}). '
                'This may indicate that different image regions have different compression histories.'
            ),
        })
    if noise_result['suspicious_blocks']:
        findings.append({
            'type': 'noise_anomaly',
            'severity': 'medium',
            'message': (
                f'{len(noise_result["suspicious_blocks"])} block(s) with '
                'unusual noise characteristics found. '
                'These regions differ significantly from surrounding areas.'
            ),
        })
    findings.extend([
        {'type': 'copy_move', 'severity': 'medium', 'message': d}
        for d in copy_move_result.get('details', [])
    ])
    findings.extend([
        {'type': 'metadata', 'severity': 'low', 'message': a}
        for a in metadata_result.get('anomalies', [])
    ])

    return {
        'ela': {'compression_diff': compression_diff, 'array': ela_arr},
        'noise': noise_result,
        'edges': edge_result,
        'copy_move': copy_move_result,
        'metadata': metadata_result,
        'findings': findings,
        'composite_score': min(1.0, composite_score),
        'suspicious_region_count': (
            len(noise_result['suspicious_blocks'])
            + copy_move_result['pair_count']
        ),
    }
