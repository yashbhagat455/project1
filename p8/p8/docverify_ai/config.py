import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Directories
UPLOADS_DIR = BASE_DIR / 'uploads'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
DATABASE_DIR = BASE_DIR / 'database'

# Database
DB_PATH = DATABASE_DIR / 'docverify.db'

# Model paths
ANOMALY_MODEL_PATH = MODELS_DIR / 'anomaly_model.pkl'
CLASSIFIER_PATH = MODELS_DIR / 'classifier.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'

# Risk bands
RISK_BANDS = {
    'LOW': (0, 30),
    'MEDIUM': (31, 60),
    'HIGH': (61, 80),
    'VERY HIGH': (81, 100),
}

RISK_COLORS = {
    'LOW': '#28a745',
    'MEDIUM': '#ffc107',
    'HIGH': '#fd7e14',
    'VERY HIGH': '#dc3545',
}

RISK_ICONS = {
    'LOW': '✅',
    'MEDIUM': '⚠️',
    'HIGH': '🔶',
    'VERY HIGH': '🚨',
}

# Supported file types
SUPPORTED_TYPES = ['pdf', 'png', 'jpg', 'jpeg']

# OCR settings
OCR_ENGINE = os.environ.get('OCR_ENGINE', 'tesseract')  # 'tesseract' or 'easyocr'

def get_tesseract_cmd() -> str:
    """Find and return the tesseract executable path."""
    env_cmd = os.environ.get('TESSERACT_CMD')
    if env_cmd and Path(env_cmd).exists():
        return env_cmd
    candidate_paths = [
        BASE_DIR.parent / 'Tesseract-OCR' / 'tesseract.exe',
        Path(r'D:\internship\p8\Tesseract-OCR\tesseract.exe'),
        Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe'),
        Path(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'),
        Path(r'C:\Users\terri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
    ]
    for p in candidate_paths:
        if p.exists():
            return str(p)
    import shutil
    which_cmd = shutil.which('tesseract')
    if which_cmd:
        return which_cmd
    return 'tesseract'

TESSERACT_CMD = get_tesseract_cmd()

# LLM settings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
LLM_ENABLED = bool(OPENAI_API_KEY or GEMINI_API_KEY)
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai')  # 'openai' or 'gemini'

# Isolation Forest settings
IF_CONTAMINATION = 0.15
IF_N_ESTIMATORS = 100
IF_RANDOM_STATE = 42

# Feature names for ML
FEATURE_NAMES = [
    'text_density',
    'font_variance',
    'font_size_variance',
    'alignment_variance',
    'number_of_fonts',
    'image_noise_variance',
    'compression_difference',
    'metadata_anomaly',
    'date_format_variance',
    'total_mismatch',
    'ocr_confidence',
    'suspicious_region_count',
]

# App info
APP_NAME = 'DocVerify AI'
APP_VERSION = '1.0.0'
APP_DESCRIPTION = 'AI-powered document authenticity and fraud risk analysis'
DISCLAIMER = (
    'This report identifies potential inconsistencies and manipulation indicators. '
    'It does NOT certify that a document is genuine or fraudulent. '
    'This is a project-defined risk score, not a certified probability of fraud. '
    'Manual verification by a qualified professional is always recommended.'
)
