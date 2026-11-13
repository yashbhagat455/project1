"""Optional Random Forest classifier for Genuine vs Suspicious.

This module trains and evaluates a Random Forest on labeled data.
If insufficient data is available, the module gracefully degrades.
"""
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from config import CLASSIFIER_PATH, SCALER_PATH

logger = logging.getLogger(__name__)

MIN_SAMPLES = 10  # minimum per class to attempt training


def train_classifier(
    X: np.ndarray, y: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Train a Random Forest classifier if enough data is available.
    
    Returns evaluation metrics dict or None if insufficient data.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or any(c < MIN_SAMPLES for c in counts):
        logger.warning(
            'Insufficient labeled data for RF classifier. '
            'Minimum %d samples per class required. Got: %s',
            MIN_SAMPLES, dict(zip(classes, counts))
        )
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classes': le.classes_.tolist(),
        'report': classification_report(y_test, y_pred, target_names=le.classes_),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'note': 'Prototype evaluation — limited dataset.',
    }

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(le, CLASSIFIER_PATH.with_suffix('.le.pkl'))
    logger.info('Saved RF classifier to %s', CLASSIFIER_PATH)
    return metrics


def classify_document(
    feature_vector: np.ndarray,
) -> Dict[str, Any]:
    """Classify a document as Genuine or Suspicious using the trained RF model."""
    if not CLASSIFIER_PATH.exists():
        return {
            'available': False,
            'reason': 'Classifier not trained yet.',
        }
    try:
        clf = joblib.load(CLASSIFIER_PATH)
        le = joblib.load(CLASSIFIER_PATH.with_suffix('.le.pkl'))
        pred = clf.predict(feature_vector.reshape(1, -1))[0]
        proba = clf.predict_proba(feature_vector.reshape(1, -1))[0]
        label = le.inverse_transform([pred])[0]
        return {
            'available': True,
            'label': label,
            'probabilities': dict(zip(le.classes_.tolist(), proba.tolist())),
        }
    except Exception as exc:
        logger.error('Classification failed: %s', exc)
        return {'available': False, 'reason': str(exc)}
