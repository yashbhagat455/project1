"""Synthetic dataset generator for development and testing.

Creates a small reproducible dataset of genuine and manipulated document
feature vectors. These are programmatically generated — NOT real documents.
All labels are synthetic.
"""
import logging
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from config import FEATURE_NAMES, MODELS_DIR, DATA_DIR

logger = logging.getLogger(__name__)

NP_SEED = 42
N_GENUINE = 60
N_MANIPULATED = 40


def _genuine_features(rng: np.random.Generator) -> np.ndarray:
    """Sample a realistic feature vector for a genuine document."""
    return np.array([
        rng.uniform(0.3, 0.9),     # text_density
        rng.uniform(0.0, 0.2),     # font_variance (few fonts)
        rng.uniform(0.0, 0.1),     # font_size_variance
        rng.uniform(0.0, 0.15),    # alignment_variance
        rng.uniform(0.1, 0.4),     # number_of_fonts
        rng.uniform(0.0, 0.2),     # image_noise_variance
        rng.uniform(0.0, 0.15),    # compression_difference (JPEG OK)
        rng.uniform(0.0, 0.1),     # metadata_anomaly
        rng.uniform(0.0, 0.1),     # date_format_variance
        0.0,                        # total_mismatch (no mismatch)
        rng.uniform(0.7, 1.0),     # ocr_confidence
        rng.uniform(0.0, 0.1),     # suspicious_region_count
    ], dtype=np.float32)


def _manipulated_features(rng: np.random.Generator) -> np.ndarray:
    """Sample a realistic feature vector for a manipulated document."""
    # Flip some features to higher anomaly values
    v = _genuine_features(rng)
    # Introduce anomalies
    modification = rng.choice([
        'total_mismatch', 'ela', 'font', 'noise', 'metadata', 'multi'
    ])
    if modification in ('total_mismatch', 'multi'):
        v[9] = 1.0   # total_mismatch
    if modification in ('ela', 'multi'):
        v[6] = rng.uniform(0.5, 1.0)  # compression_difference
        v[11] = rng.uniform(0.3, 0.8)  # suspicious_region_count
    if modification in ('font', 'multi'):
        v[1] = rng.uniform(0.5, 1.0)  # font_variance
        v[4] = rng.uniform(0.6, 1.0)  # number_of_fonts
    if modification in ('noise', 'multi'):
        v[5] = rng.uniform(0.4, 1.0)  # image_noise_variance
    if modification in ('metadata', 'multi'):
        v[7] = rng.uniform(0.4, 1.0)  # metadata_anomaly
    return v


def generate_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible synthetic dataset.
    
    Returns:
        X: (N, n_features) feature matrix
        y: (N,) label array (0=genuine, 1=manipulated)
    """
    rng = np.random.default_rng(NP_SEED)
    genuine = np.array([_genuine_features(rng) for _ in range(N_GENUINE)])
    manipulated = np.array([_manipulated_features(rng) for _ in range(N_MANIPULATED)])
    X = np.vstack([genuine, manipulated])
    y = np.array([0] * N_GENUINE + [1] * N_MANIPULATED)
    return X, y


def get_labeled_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return the dataset with string labels."""
    X, y_int = generate_dataset()
    labels = np.array(['genuine' if v == 0 else 'manipulated' for v in y_int])
    return X, labels, FEATURE_NAMES


def save_dataset_info() -> None:
    """Save dataset info JSON."""
    info = {
        'description': 'Synthetic dataset for DocVerify AI development and testing.',
        'warning': 'These are programmatically generated feature vectors, NOT real documents.',
        'n_genuine': N_GENUINE,
        'n_manipulated': N_MANIPULATED,
        'seed': NP_SEED,
        'features': FEATURE_NAMES,
        'labels': ['genuine', 'manipulated'],
        'creation_method': (
            'Genuine samples: low anomaly feature values. '
            'Manipulated samples: genuine base with controlled perturbations '
            'to specific features (total mismatch, ELA, font, noise, metadata).'
        ),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    logger.info('Saved dataset info to %s', DATA_DIR / 'dataset_info.json')
