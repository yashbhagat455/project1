"""Feature extraction module for ML models."""
import logging
from typing import Dict, Any, List

import numpy as np

from config import FEATURE_NAMES

logger = logging.getLogger(__name__)


def extract_features(
    ocr_result: Dict[str, Any],
    text_analysis: Dict[str, Any],
    image_analysis: Dict[str, Any],
    math_checks: Dict[str, Any],
    pdf_font_info: Dict[str, Any],
    pdf_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract a fixed feature vector from all analysis results.

    Returns a dict with feature names and values, plus the raw vector.
    """
    features: Dict[str, float] = {}

    # 1. text_density — word count normalized
    word_count = ocr_result.get('word_count', 0)
    features['text_density'] = min(1.0, word_count / 500)

    # 2. font_variance — from PDF font count
    font_count = pdf_font_info.get('font_count', 1)
    features['font_variance'] = min(1.0, max(0.0, (font_count - 1) / 5))

    # 3. font_size_variance — placeholder from PDF variance score
    features['font_size_variance'] = pdf_font_info.get('variance_score', 0.0)

    # 4. alignment_variance — from edge asymmetry
    edge_result = image_analysis.get('edges', {})
    features['alignment_variance'] = min(
        1.0, edge_result.get('edge_asymmetry', 0.0) * 10
    )

    # 5. number_of_fonts — raw count
    features['number_of_fonts'] = min(1.0, font_count / 8)

    # 6. image_noise_variance
    noise_result = image_analysis.get('noise', {})
    features['image_noise_variance'] = min(
        1.0, noise_result.get('noise_variance', 0.0) / 10000
    )

    # 7. compression_difference — from ELA
    ela_result = image_analysis.get('ela', {})
    comp_diff = ela_result.get('compression_diff', 0.0)
    features['compression_difference'] = min(1.0, comp_diff / 30)

    # 8. metadata_anomaly — from image and PDF metadata
    img_meta_score = image_analysis.get('metadata', {}).get('score', 0.0)
    pdf_meta_score = pdf_metadata.get('anomaly_score', 0.0)
    features['metadata_anomaly'] = min(1.0, (img_meta_score + pdf_meta_score) / 2)

    # 9. date_format_variance
    features['date_format_variance'] = min(
        1.0, text_analysis.get('date_format_variance', 0.0)
    )

    # 10. total_mismatch — binary flag from math checks
    features['total_mismatch'] = 1.0 if math_checks.get('total_mismatch', False) else 0.0

    # 11. ocr_confidence — normalized
    ocr_conf = ocr_result.get('confidence', 0.0)
    features['ocr_confidence'] = min(1.0, ocr_conf / 100)

    # 12. suspicious_region_count
    suspicious = image_analysis.get('suspicious_region_count', 0)
    features['suspicious_region_count'] = min(1.0, suspicious / 10)

    # Build ordered vector
    vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)

    return {
        'features': features,
        'vector': vector,
        'feature_names': FEATURE_NAMES,
    }
