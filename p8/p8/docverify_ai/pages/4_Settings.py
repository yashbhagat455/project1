"""Page 4: Settings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import os

from config import DB_PATH, MODELS_DIR, APP_NAME, APP_VERSION, DISCLAIMER
from src.database import initialize_database, get_statistics, clear_all_analyses
from src.dataset_generator import generate_dataset, get_labeled_dataset, save_dataset_info
from src.anomaly_detection import train_model
from src.classification import train_classifier

initialize_database()

st.set_page_config(
    page_title='Settings — DocVerify AI',
    page_icon='⚙️',
    layout='wide',
)

st.markdown("""
<style>
/* Main view */
[data-testid="stAppViewContainer"] {
    background-color: #f8f9fb !important;
    color: #1e293b !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0;
}
[data-testid="stSidebar"] a {
    color: #cbd5e1 !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    display: block !important;
    margin: 2px 0 !important;
}
[data-testid="stSidebar"] a:hover {
    background-color: #1e293b !important;
    color: #818cf8 !important;
}

/* Metric cards */
[data-testid="stMetricContainer"], [data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* Headers */
.section-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    padding: 10px 18px !important;
    border-radius: 8px !important;
    margin: 24px 0 14px 0 !important;
    font-size: 1.05em !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(79, 70, 229, 0.15) !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 14px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;">
        <h2 style='color:#ffffff; margin:0; font-size:1.35em;'>🔍 DocVerify AI</h2>
        <p style='color:#94a3b8; font-size:0.8em; margin:4px 0 0;'>Settings</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link('app.py', label='Dashboard', icon='🏠')
    st.page_link('pages/1_Analyze_Document.py', label='Analyze Document', icon='📤')
    st.page_link('pages/2_Analysis_History.py', label='Analysis History', icon='📊')
    st.page_link('pages/3_How_It_Works.py', label='How It Works', icon='📖')
    st.page_link('pages/4_Settings.py', label='Settings', icon='⚙️')

st.markdown('## ⚙️ Settings')

# ─ OCR Settings ───────────────────────────────────────────────
st.markdown("<div class='section-header'>🔤 OCR Engine Configuration</div>", unsafe_allow_html=True)

import config
import importlib
try:
    importlib.reload(config)
except Exception:
    pass

TESSERACT_CMD = getattr(config, 'TESSERACT_CMD', None)
if not TESSERACT_CMD and hasattr(config, 'get_tesseract_cmd'):
    TESSERACT_CMD = config.get_tesseract_cmd()
elif not TESSERACT_CMD:
    TESSERACT_CMD = r'D:\internship\p8\Tesseract-OCR\tesseract.exe'

# Check Tesseract detection
tess_found = False
tess_version = "Unknown"
if TESSERACT_CMD and Path(TESSERACT_CMD).exists():
    tess_found = True
    try:
        import pytesseract
        from src.ocr import configure_pytesseract
        configure_pytesseract()
        tess_version = str(pytesseract.get_tesseract_version())
    except Exception:
        tess_version = "Detected (binary ready)"

ocr_col1, ocr_col2 = st.columns([1, 1])

with ocr_col1:
    ocr_engine = st.selectbox(
        'Active OCR Engine',
        options=['tesseract', 'easyocr'],
        index=0 if st.session_state.get('ocr_engine', 'tesseract') == 'tesseract' else 1,
        help='Tesseract is fast and accurate for structured documents.'
    )
    if st.button('💾 Save OCR Setting'):
        st.session_state['ocr_engine'] = ocr_engine
        st.success(f'OCR engine set to {ocr_engine}.')

with ocr_col2:
    if tess_found:
        st.markdown(
            f"""
            <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:12px 16px; margin-top:24px;">
                <div style="font-weight:700; color:#15803d;">✅ Tesseract OCR Ready</div>
                <div style="font-size:0.85em; color:#166534; margin-top:4px;">
                    <strong>Version:</strong> {tess_version}<br>
                    <strong>Path:</strong> <code>{TESSERACT_CMD}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:12px 16px; margin-top:24px;">
                <div style="font-weight:700; color:#b91c1c;">⚠️ Tesseract Not Found</div>
                <div style="font-size:0.85em; color:#991b1b; margin-top:4px;">
                    System will fall back to EasyOCR or PDF text extraction.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Test Tesseract OCR button
if tess_found:
    with st.expander("🧪 Test Tesseract OCR"):
        st.write("Run a quick test on synthetic text to verify the OCR engine is functioning correctly.")
        if st.button("Run Quick OCR Test"):
            with st.spinner("Processing test image with Tesseract..."):
                try:
                    from PIL import Image, ImageDraw
                    from src.ocr import run_ocr
                    test_img = Image.new('RGB', (450, 100), 'white')
                    d = ImageDraw.Draw(test_img)
                    d.text((20, 35), "DocVerify AI: Tesseract 5.5 Active", fill='black')
                    test_res = run_ocr(test_img, engine='tesseract')
                    st.success(f"✅ OCR Test Passed! Extracted text: **{test_res.get('text', '').strip()}** (Confidence: {test_res.get('confidence', 0):.1f}%)")
                except Exception as e:
                    st.error(f"Test failed: {e}")

# ─ Risk Thresholds ───────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Risk Thresholds (Read-only)</div>", unsafe_allow_html=True)
st.markdown("""
| Risk Level | Score Range | Meaning |
|------------|------------|----------|
| ✅ LOW | 0 – 30 | Few or no anomalies detected |
| ⚠️ MEDIUM | 31 – 60 | Some potential indicators found |
| 🔴 HIGH | 61 – 80 | Multiple indicators; manual review recommended |
| 🚨 VERY HIGH | 81 – 100 | Significant indicators; immediate verification required |
""")
st.caption('These are project-defined thresholds, not certified forensic standards.')

# ─ LLM Configuration ──────────────────────────────────────────
st.markdown("<div class='section-header'>🤖 LLM Integration (Optional)</div>", unsafe_allow_html=True)
st.info("""
LLM integration is **optional**. The core analysis works without any API key.
The LLM is only used to generate a plain-language summary of the technical findings.
No document content is ever sent to an LLM by default.
""")

llm_provider = st.selectbox('LLM Provider', ['OpenAI', 'Google Gemini'], index=0)

if llm_provider == 'OpenAI':
    api_key_input = st.text_input(
        'OpenAI API Key',
        type='password',
        value='(set via OPENAI_API_KEY env var)' if os.environ.get('OPENAI_API_KEY') else '',
        help='Set via environment variable OPENAI_API_KEY. Do not paste keys into production UIs.'
    )
    st.caption('Set OPENAI_API_KEY as an environment variable or in .streamlit/secrets.toml')
else:
    api_key_input = st.text_input(
        'Gemini API Key',
        type='password',
        value='(set via GEMINI_API_KEY env var)' if os.environ.get('GEMINI_API_KEY') else '',
        help='Set via environment variable GEMINI_API_KEY.'
    )
    st.caption('Set GEMINI_API_KEY as an environment variable or in .streamlit/secrets.toml')

st.warning('🔐 Never hard-code API keys. Always use environment variables or Streamlit secrets.')

# ─ ML Models ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>🤖 ML Models & Evaluation</div>", unsafe_allow_html=True)

from config import ANOMALY_MODEL_PATH, SCALER_PATH, CLASSIFIER_PATH
model_col1, model_col2, model_col3 = st.columns(3)
with model_col1:
    exists = ANOMALY_MODEL_PATH.exists()
    st.metric('Anomaly Model', '✅ Trained' if exists else '❌ Not found')
with model_col2:
    exists = SCALER_PATH.exists()
    st.metric('Scaler', '✅ Available' if exists else '❌ Not found')
with model_col3:
    exists = CLASSIFIER_PATH.exists()
    st.metric('RF Classifier', '✅ Trained' if exists else '⚠️ Not trained')

if st.button('🔄 Retrain Models on Synthetic Dataset', type='secondary'):
    with st.spinner('Training models...'):
        try:
            save_dataset_info()
            X, y_str, feature_names = get_labeled_dataset()
            X_genuine = X[y_str == 'genuine']
            train_model(X_genuine)
            metrics = train_classifier(X, y_str)
            st.success('✅ Models retrained successfully!')
            if metrics:
                st.markdown('### Model Evaluation (Prototype — Synthetic Dataset)')
                st.caption('⚠️ Prototype evaluation — limited synthetic dataset. Results may not reflect real-world performance.')
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric('Accuracy', f"{metrics['accuracy']:.3f}")
                with m_col2:
                    st.metric('Precision', f"{metrics['precision']:.3f}")
                with m_col3:
                    st.metric('Recall', f"{metrics['recall']:.3f}")
                with m_col4:
                    st.metric('F1 Score', f"{metrics['f1']:.3f}")
                import pandas as pd
                import numpy as np
                cm = np.array(metrics['confusion_matrix'])
                classes = metrics['classes']
                cm_df = pd.DataFrame(cm, index=[f'Actual: {c}' for c in classes],
                                     columns=[f'Predicted: {c}' for c in classes])
                st.markdown('**Confusion Matrix:**')
                st.dataframe(cm_df)
                st.text(metrics['report'])
                st.info(f"Training set: {metrics['n_train']} samples | Test set: {metrics['n_test']} samples")
            else:
                st.info('RF classifier not trained (insufficient data). Isolation Forest is still active.')
        except Exception as e:
            st.error(f'Training failed: {e}')

# ─ Database Info ───────────────────────────────────────────────
st.markdown("<div class='section-header'>🖴 Database</div>", unsafe_allow_html=True)
stats = get_statistics()
db_col1, db_col2 = st.columns(2)
with db_col1:
    st.info(f'🖺 Database path: `{DB_PATH}`')
    st.metric('Total Records', stats['total'])
with db_col2:
    st.metric('Total Issues Found', stats['total_issues'])
    db_exists = DB_PATH.exists()
    if db_exists:
        size_kb = DB_PATH.stat().st_size / 1024
        st.metric('Database Size', f'{size_kb:.1f} KB')

# Privacy notice
st.markdown("<div class='section-header'>🔐 Privacy Notice</div>", unsafe_allow_html=True)
st.markdown("""
**DocVerify AI handles uploaded documents with the following privacy protections:**

- Documents are processed **in memory** and not permanently stored on disk
- Only analysis **metadata and feature scores** are saved to the SQLite database
- **No document content** is sent to external servers unless you have explicitly enabled LLM summarisation and configured an API key
- API keys are never stored in the database or shown in the interface after entry
- You can delete any or all analysis records at any time from the History page
- Use environment variables or Streamlit secrets for API key storage — never hard-code them

**Uploaded documents should be treated as sensitive.** Do not upload documents containing personal information to shared or public deployments of this tool.
""")

st.markdown('---')
st.caption(f'{APP_NAME} v{APP_VERSION} | {DISCLAIMER}')
