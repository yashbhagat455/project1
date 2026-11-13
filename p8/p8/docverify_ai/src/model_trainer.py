"""Model training script. Run this to (re-)train the ML models."""
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_generator import generate_dataset, get_labeled_dataset, save_dataset_info
from src.anomaly_detection import train_model
from src.classification import train_classifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info('Generating synthetic dataset ...')
    save_dataset_info()
    X, y_str, feature_names = get_labeled_dataset()
    X_genuine = X[y_str == 'genuine']

    logger.info('Training Isolation Forest anomaly model on %d genuine samples...', len(X_genuine))
    model, scaler = train_model(X_genuine)
    logger.info('Isolation Forest trained.')

    logger.info('Training Random Forest classifier on %d samples...', len(X))
    metrics = train_classifier(X, y_str)
    if metrics:
        logger.info('Classification Report:')
        logger.info(metrics['report'])
        logger.info(
            'Accuracy=%.3f Precision=%.3f Recall=%.3f F1=%.3f',
            metrics['accuracy'], metrics['precision'],
            metrics['recall'], metrics['f1'],
        )
    else:
        logger.warning('Classifier not trained (insufficient data).')

    logger.info('Model training complete.')
    return metrics


if __name__ == '__main__':
    main()
