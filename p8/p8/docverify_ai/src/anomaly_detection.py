"""Isolation Forest anomaly detection module."""
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import (
    ANOMALY_MODEL_PATH, SCALER_PATH,
    IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE, FEATURE_NAMES
)

logger = logging.getLogger(__name__)


def load_or_create_model() -> tuple:
    """Load existing model + scaler or return None, None."""
    model = None
    scaler = None
    try:
        if ANOMALY_MODEL_PATH.exists() and SCALER_PATH.exists():
            model = joblib.load(ANOMALY_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info('Loaded anomaly model from %s', ANOMALY_MODEL_PATH)
    except Exception as exc:
        logger.warning('Could not load model: %s', exc)
    return model, scaler


def train_model(feature_vectors: np.ndarray) -> tuple:
    """Train an Isolation Forest model on the given feature vectors."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_vectors)
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)
    ANOMALY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ANOMALY_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info('Saved anomaly model to %s', ANOMALY_MODEL_PATH)
    return model, scaler


def score_document(
    feature_vector: np.ndarray,
    model: Optional[IsolationForest],
    scaler: Optional[StandardScaler],
) -> Dict[str, Any]:
    """Compute anomaly score for a single document.

    Returns a dict with:
    - anomaly_score: float in [0, 1], higher = more anomalous
    - is_anomaly: bool
    - raw_score: raw IF decision score
    """
    if model is None or scaler is None:
        # Fall back to heuristic based on feature values
        logger.warning('No ML model available — using heuristic scoring')
        anomaly_score = float(np.clip(feature_vector.mean() * 2, 0, 1))
        return {
            'anomaly_score': anomaly_score,
            'is_anomaly': anomaly_score > 0.5,
            'raw_score': anomaly_score,
            'method': 'heuristic',
        }

    try:
        X = scaler.transform(feature_vector.reshape(1, -1))
        raw = float(model.decision_function(X)[0])
        pred = int(model.predict(X)[0])
        # Convert: IF returns negative for anomalies; normalize to [0, 1]
        # Typical range of decision_function is roughly [-0.5, 0.5]
        anomaly_score = float(np.clip(0.5 - raw, 0, 1))
        return {
            'anomaly_score': anomaly_score,
            'is_anomaly': pred == -1,
            'raw_score': raw,
            'method': 'isolation_forest',
        }
    except Exception as exc:
        logger.error('Scoring failed: %s', exc)
        return {
            'anomaly_score': 0.3,
            'is_anomaly': False,
            'raw_score': 0.0,
            'method': 'fallback',
        }
