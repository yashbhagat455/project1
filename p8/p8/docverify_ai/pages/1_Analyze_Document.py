"""Page 1: Analyze Document — main analysis workflow."""
import sys
import io
import time
import tempfile
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from config import (
    SUPPORTED_TYPES, RISK_COLORS, RISK_ICONS, DISCLAIMER, APP_NAME,
)
from src.database import save_analysis, initialize_database
from src.preprocessing import load_image_from_bytes, load_pil_image, resize_for_processing, pil_to_cv2
from src.pdf_processor import (
    extract_pdf_info, render_pdf_pages, extract_pdf_text_by_page,
    extract_pdf_metadata, get_pdf_font_info,
)
from src.ocr import run_ocr, run_ocr_multi_page, extract_entities
from src.text_analysis import run_text_analysis
from src.mathematical_checks import run_mathematical_checks
from src.image_analysis import run_image_analysis
from src.feature_extraction import extract_features
from src.anomaly_detection import load_or_create_model, score_document
from src.classification import classify_document
from src.risk_engine import compute_risk_score
from src.explanation import generate_explanations, generate_llm_summary
from src.visualization import draw_findings_overlay, create_ela_visualization
from src.report_generator import generate_text_report, generate_html_report

initialize_database()

st.set_page_config(
    page_title='Analyze Document — DocVerify AI',
    page_icon='📤',
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

.finding-card {
    background: white; border-left: 4px solid #fd7e14;
    border-radius: 6px; padding: 12px 16px; margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    color: #1e293b !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 14px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;">
        <h2 style='color:#ffffff; margin:0; font-size:1.35em;'>🔍 DocVerify AI</h2>
        <p style='color:#94a3b8; font-size:0.8em; margin:4px 0 0;'>Analyze Document</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link('app.py', label='Dashboard', icon='🏠')
    st.page_link('pages/1_Analyze_Document.py', label='Analyze Document', icon='📤')
    st.page_link('pages/2_Analysis_History.py', label='Analysis History', icon='📊')
    st.page_link('pages/3_How_It_Works.py', label='How It Works', icon='📖')
    st.page_link('pages/4_Settings.py', label='Settings', icon='⚙️')
    st.markdown("---")
    st.caption("⚠️ This tool identifies indicators only. Always verify documents manually.")

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("## 📤 Analyze Document")
st.markdown("Upload a PDF or image document to begin automated risk analysis.")

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a document here",
    type=SUPPORTED_TYPES,
    help="Supported: PDF, PNG, JPG, JPEG. Max ~20 MB recommended.",
)

if uploaded_file is None:
    st.info("👆 Upload a document above to start analysis.")
    st.stop()

# ── File Info ────────────────────────────────────────────────────────────────
file_bytes = uploaded_file.read()
file_name  = uploaded_file.name
file_ext   = Path(file_name).suffix.lower().lstrip('.')
file_size  = len(file_bytes)

st.markdown("<div class='section-header'>📄 Document Information</div>", unsafe_allow_html=True)
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.metric("File Name", file_name)
with info_col2:
    st.metric("File Type", file_ext.upper())
with info_col3:
    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1_000_000 else f"{file_size / 1_000_000:.2f} MB"
    st.metric("File Size", size_str)

# ── Preview ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>👁️ Document Preview</div>", unsafe_allow_html=True)

preview_images = []
pdf_info = {}
pdf_pages_text = []
pdf_font_info_data = {}
pdf_meta_data = {'raw': {}, 'anomalies': [], 'anomaly_score': 0.0}

if file_ext == 'pdf':
    try:
        pdf_info = extract_pdf_info(file_bytes)
        if pdf_info.get('error'):
            st.error(f"⚠️ PDF load warning: {pdf_info['error']}")
        else:
            st.metric("Pages", pdf_info.get('page_count', '?'))
            preview_images = render_pdf_pages(file_bytes, max_pages=3)
            pdf_pages_text = extract_pdf_text_by_page(file_bytes)
            pdf_font_info_data = get_pdf_font_info(file_bytes)
            pdf_meta_data = extract_pdf_metadata(file_bytes)
    except Exception as e:
        st.error(f"Could not process PDF: {e}")
else:
    pil_img = load_pil_image(file_bytes)
    if pil_img:
        preview_images = [pil_img]
    else:
        st.error("Could not load image. Please check the file and try again.")
        st.stop()

if preview_images:
    preview_cols = st.columns(min(len(preview_images), 3))
    for i, img in enumerate(preview_images[:3]):
        with preview_cols[i]:
            st.image(img, caption=f"Page {i+1}" if file_ext == 'pdf' else file_name, use_container_width=True)

# ── Analyse Button ────────────────────────────────────────────────────────────
st.markdown("---")
analyze_btn = st.button("🔍 Analyze Document", type="primary", use_container_width=True)

if not analyze_btn:
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>⚙️ Analysis Pipeline</div>", unsafe_allow_html=True)
progress_bar = st.progress(0)
status_area  = st.empty()
step_log     = st.container()

def update_progress(pct: int, msg: str):
    progress_bar.progress(pct)
    status_area.markdown(f"**⏳ {msg}**")

ocr_engine = st.session_state.get('ocr_engine', 'tesseract')

# Step 1: Document loaded
update_progress(5, "Document loaded …")
step_log.markdown("✅ **Document loaded**")
time.sleep(0.1)

# Step 2: OCR
update_progress(15, "Running OCR …")
try:
    if file_ext == 'pdf':
        # Combine PDF text layer + OCR on rendered pages
        pdf_text = '\n'.join(pdf_pages_text)
        if len(pdf_text.strip()) < 50 and preview_images:
            ocr_result = run_ocr_multi_page(preview_images, engine=ocr_engine)
            ocr_result['text'] = pdf_text + '\n' + ocr_result.get('text', '')
        else:
            ocr_result = {
                'engine': 'pdf_text_layer',
                'text': pdf_text,
                'confidence': 90.0,
                'word_count': len(pdf_text.split()),
                'success': bool(pdf_text.strip()),
            }
    else:
        pil_img_for_ocr = preview_images[0] if preview_images else None
        if pil_img_for_ocr:
            ocr_result = run_ocr(pil_img_for_ocr, engine=ocr_engine)
        else:
            ocr_result = {'engine': 'none', 'text': '', 'confidence': 0, 'word_count': 0, 'success': False}
except Exception as e:
    st.warning(f"OCR encountered an issue: {e}. Continuing with empty text.")
    ocr_result = {'engine': 'error', 'text': '', 'confidence': 0, 'word_count': 0, 'success': False}

step_log.markdown(f"✅ **OCR completed** — engine: `{ocr_result.get('engine', 'unknown')}`, confidence: `{ocr_result.get('confidence', 0):.1f}%`, words: `{ocr_result.get('word_count', 0)}`")

# Step 3: Entity extraction + text analysis
update_progress(30, "Analyzing text …")
try:
    entities = extract_entities(ocr_result.get('text', ''))
    text_result = run_text_analysis(entities)
except Exception as e:
    st.warning(f"Text analysis error: {e}")
    entities = {'dates': [], 'currency_values': [], 'invoice_numbers': [], 'tax_rates': [], 'raw_text': ''}
    text_result = {'composite_score': 0.0, 'all_details': [], 'issue_count': 0, 'date_format_variance': 0.0}

step_log.markdown(f"✅ **Text analyzed** — {text_result.get('issue_count', 0)} potential text inconsistency(ies)")

# Step 4: Math checks
update_progress(45, "Running mathematical checks …")
try:
    math_result = run_mathematical_checks(ocr_result.get('text', ''), entities)
except Exception as e:
    st.warning(f"Math check error: {e}")
    math_result = {'has_data': False, 'findings': [], 'composite_score': 0.0, 'total_mismatch': False}

step_log.markdown(f"✅ **Mathematical checks** — {'data found, ' + str(len(math_result.get('findings', []))) + ' issue(s)' if math_result.get('has_data') else 'insufficient structured data'}")

# Step 5: Image analysis
update_progress(60, "Running image analysis …")
try:
    primary_img = preview_images[0] if preview_images else None
    if primary_img:
        cv_img = pil_to_cv2(primary_img)
        image_result = run_image_analysis(primary_img, cv_img)
    else:
        image_result = {
            'ela': {'compression_diff': 0.0, 'array': None},
            'noise': {'suspicious_blocks': [], 'noise_variance': 0.0, 'score': 0.0},
            'edges': {'score': 0.0},
            'copy_move': {'duplicate_pairs': [], 'pair_count': 0, 'score': 0.0, 'details': []},
            'metadata': {'anomalies': [], 'score': 0.0},
            'findings': [], 'composite_score': 0.0, 'suspicious_region_count': 0,
        }
except Exception as e:
    st.warning(f"Image analysis error: {e}")
    image_result = {
        'ela': {'compression_diff': 0.0, 'array': None},
        'noise': {'suspicious_blocks': [], 'noise_variance': 0.0, 'score': 0.0},
        'edges': {'score': 0.0},
        'copy_move': {'duplicate_pairs': [], 'pair_count': 0, 'score': 0.0, 'details': []},
        'metadata': {'anomalies': [], 'score': 0.0},
        'findings': [], 'composite_score': 0.0, 'suspicious_region_count': 0,
    }

step_log.markdown(f"✅ **Image analyzed** — {image_result.get('suspicious_region_count', 0)} suspicious region(s)")

# Step 6: Feature extraction
update_progress(72, "Extracting ML features …")
try:
    feature_data = extract_features(
        ocr_result, text_result, image_result, math_result,
        pdf_font_info_data, pdf_meta_data,
    )
except Exception as e:
    st.warning(f"Feature extraction error: {e}")
    from config import FEATURE_NAMES
    feature_data = {'features': {n: 0.0 for n in FEATURE_NAMES}, 'vector': np.zeros(12), 'feature_names': FEATURE_NAMES}

step_log.markdown("✅ **Features extracted**")

# Step 7: ML anomaly detection
update_progress(82, "Running ML anomaly detection …")
try:
    model, scaler = load_or_create_model()
    anomaly_result = score_document(feature_data['vector'], model, scaler)
    clf_result = classify_document(feature_data['vector'])
except Exception as e:
    st.warning(f"ML scoring error: {e}")
    anomaly_result = {'anomaly_score': 0.2, 'is_anomaly': False, 'raw_score': 0.0, 'method': 'fallback'}
    clf_result = {'available': False, 'reason': str(e)}

step_log.markdown(f"✅ **ML analysis** — anomaly score: `{anomaly_result['anomaly_score']:.3f}` (method: `{anomaly_result['method']}`)")

# Step 8: Risk score
update_progress(92, "Generating risk score …")
try:
    risk_result = compute_risk_score(
        anomaly_result, image_result, text_result, math_result, pdf_meta_data
    )
except Exception as e:
    st.error(f"Risk scoring failed: {e}")
    risk_result = {'risk_score': 0.0, 'risk_level': 'LOW', 'issue_count': 0, 'findings': [], 'major_findings': [], 'component_scores': {}}

# Step 9: Explanations
update_progress(97, "Generating explanations …")
explanations = generate_explanations(risk_result.get('findings', []))
step_log.markdown("✅ **Risk score generated**")
progress_bar.progress(100)
status_area.markdown("✅ **Analysis complete!**")

# ── Save to DB ────────────────────────────────────────────────────────────────
doc_type = 'PDF' if file_ext == 'pdf' else 'Image'
db_record = {
    'filename': file_name,
    'timestamp': datetime.now().isoformat(),
    'risk_score': risk_result['risk_score'],
    'risk_level': risk_result['risk_level'],
    'issue_count': risk_result['issue_count'],
    'document_type': doc_type,
    'major_findings': risk_result.get('major_findings', []),
    'component_scores': risk_result.get('component_scores', {}),
    'findings': risk_result.get('findings', []),
    'ocr_result': {k: v for k, v in ocr_result.items() if k != 'raw_data'},
    'image_result': {k: v for k, v in image_result.items() if k not in ('ela',)},
    'math_result': math_result,
    'anomaly_result': anomaly_result,
}
# Store ela without numpy array
ela_arr = image_result.get('ela', {}).get('array')
db_record['image_result']['ela'] = {'compression_diff': image_result.get('ela', {}).get('compression_diff', 0)}
try:
    record_id = save_analysis(db_record)
except Exception as e:
    st.warning(f"Could not save to database: {e}")
    record_id = None

# ═════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-header'>📊 Analysis Results</div>", unsafe_allow_html=True)

risk_score = risk_result['risk_score']
risk_level = risk_result['risk_level']
risk_color = RISK_COLORS.get(risk_level, '#6c757d')
risk_icon  = RISK_ICONS.get(risk_level, '❓')

# Risk gauge
col_gauge, col_details = st.columns([1, 2])
with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=risk_score,
        title={'text': 'Risk Score', 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30],   'color': '#d4edda'},
                {'range': [30, 60],  'color': '#fff3cd'},
                {'range': [60, 80],  'color': '#fde5cc'},
                {'range': [80, 100], 'color': '#f8d7da'},
            ],
            'threshold': {
                'line': {'color': risk_color, 'width': 4},
                'thickness': 0.75,
                'value': risk_score,
            },
        },
        number={'suffix': '/100', 'font': {'size': 32}},
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(
        f"<div style='text-align:center;'>"
        f"<span style='background:{risk_color};color:white;padding:8px 20px;"
        f"border-radius:20px;font-weight:700;font-size:1.2em;'>"
        f"{risk_icon} {risk_level}</span></div>",
        unsafe_allow_html=True
    )

with col_details:
    st.markdown("#### Component Scores")
    comp_scores = risk_result.get('component_scores', {})
    if comp_scores:
        fig_bar = px.bar(
            x=list(comp_scores.keys()),
            y=list(comp_scores.values()),
            color=list(comp_scores.values()),
            color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
            range_color=[0, 100],
            labels={'x': 'Component', 'y': 'Score (0-100)'},
            title='Finding Distribution by Component',
        )
        fig_bar.update_layout(
            showlegend=False, height=240,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.metric("Total Issues Detected", risk_result.get('issue_count', 0))
    if clf_result.get('available'):
        proba = clf_result.get('probabilities', {})
        suspicious_pct = proba.get('manipulated', 0) * 100
        st.metric("Classifier: Suspicious Probability", f"{suspicious_pct:.1f}%",
                  help="Optional Random Forest classifier. Treat as supplementary indicator only.")

# ── Findings ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🔎 Detailed Findings</div>", unsafe_allow_html=True)

if not explanations:
    st.success("✅ No significant issues detected in this document.")
else:
    sev_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_exp = sorted(explanations, key=lambda x: sev_order.get(x.get('severity', 'low'), 2))

    for exp in sorted_exp:
        sev = exp.get('severity', 'low')
        icon = {'high': '🔴', 'medium': '🟠', 'low': 'ℹ️'}.get(sev, 'ℹ️')
        with st.expander(f"{icon} {exp['title']}", expanded=(sev == 'high')):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**📌 What was detected?**")
                st.write(exp['what'])
                st.markdown("**📋 Evidence**")
                st.write(exp['evidence'])
            with col_b:
                st.markdown("**❓ Why it matters**")
                st.write(exp['why'])
                st.markdown("**✅ Recommended verification**")
                st.write(exp['action'])

# ── OCR Text ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📄 Extracted Text</div>", unsafe_allow_html=True)
with st.expander("📄 View Extracted Text (OCR output)"):
    ocr_text = ocr_result.get('text', '').strip()
    if ocr_text:
        st.text_area("Raw OCR text", ocr_text, height=250)
        st.caption(f"Engine: {ocr_result.get('engine')} | Confidence: {ocr_result.get('confidence', 0):.1f}% | Words: {ocr_result.get('word_count', 0)}")
    else:
        st.info("No text could be extracted from this document.")

    # Extracted entities
    if any(entities.get(k) for k in ('dates', 'currency_values', 'invoice_numbers')):
        st.markdown("**Extracted Entities:**")
        ent_col1, ent_col2, ent_col3 = st.columns(3)
        with ent_col1:
            st.markdown("**📅 Dates**")
            for d in entities.get('dates', [])[:10]:
                st.write(f"• {d}")
        with ent_col2:
            st.markdown("**💰 Currency Values**")
            for v in entities.get('currency_values', [])[:10]:
                st.write(f"• {v:,.2f}")
        with ent_col3:
            st.markdown("**🔢 Invoice Numbers**")
            for n in entities.get('invoice_numbers', [])[:5]:
                st.write(f"• {n}")

# ── Math Results ──────────────────────────────────────────────────────────────
if math_result.get('has_data'):
    st.markdown("<div class='section-header'>🧮 Mathematical Consistency</div>", unsafe_allow_html=True)
    extracted = math_result.get('invoice', {}).get('extracted', {})
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Subtotal", f"{extracted.get('subtotal', 0):,.2f}" if extracted.get('subtotal') else "N/A")
    with m_col2:
        st.metric("Tax Rate", f"{extracted.get('tax_rate', 0):.1f}%" if extracted.get('tax_rate') else "N/A")
    with m_col3:
        st.metric("Document Total", f"{extracted.get('total', 0):,.2f}" if extracted.get('total') else "N/A")
    with m_col4:
        st.metric("Expected Total", f"{extracted.get('expected_total', 0):,.2f}" if extracted.get('expected_total') else "N/A")

    if math_result.get('total_mismatch'):
        st.error("⚠️ Total amount mismatch detected. The document total does not match the calculated expected total.")
    else:
        st.success("✅ Mathematical totals appear consistent.")

# ── Image Analysis Visual ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🖼️ Image Analysis</div>", unsafe_allow_html=True)

if primary_img := (preview_images[0] if preview_images else None):
    img_tabs = st.tabs(["📸 Original", "🔬 Analysis Overlay", "🌡️ ELA Heatmap"])

    with img_tabs[0]:
        st.image(primary_img, caption="Original document", use_container_width=True)

    with img_tabs[1]:
        try:
            cv_orig = pil_to_cv2(primary_img)
            annotated = draw_findings_overlay(cv_orig, image_result, math_result)
            # Convert BGR to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Analysis overlay (Yellow=noise, Red=copy-move, Orange=math issue)", use_container_width=True)
            if not image_result.get('findings'):
                st.success("✅ No suspicious regions highlighted.")
        except Exception as e:
            st.info(f"Overlay generation: {e}")

    with img_tabs[2]:
        try:
            ela_arr_data = image_result.get('ela', {}).get('array')
            if ela_arr_data is not None and ela_arr_data.size > 0:
                ela_heatmap = create_ela_visualization(ela_arr_data)
                ela_heatmap_rgb = cv2.cvtColor(ela_heatmap, cv2.COLOR_BGR2RGB)
                st.image(ela_heatmap_rgb, caption="ELA Heatmap — brighter regions may indicate different compression history", use_container_width=True)
                st.caption(f"ELA compression difference score: {image_result.get('ela', {}).get('compression_diff', 0):.2f}")
            else:
                st.info("ELA analysis data not available for this document.")
        except Exception as e:
            st.info(f"ELA visualization: {e}")
else:
    st.info("No image available for visualization.")

# ── ML Features ──────────────────────────────────────────────────────────────
with st.expander("🤖 ML Feature Vector"):
    feat_df = pd.DataFrame([
        {'Feature': k, 'Value': round(v, 4), 'Interpretation': '↑ Higher = more anomalous' if k != 'ocr_confidence' else '↑ Higher = more reliable OCR'}
        for k, v in feature_data['features'].items()
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.caption(f"Anomaly method: {anomaly_result.get('method')} | Raw IF score: {anomaly_result.get('raw_score', 0):.4f}")

# ── LLM Summary ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>💬 Plain-Language Summary</div>", unsafe_allow_html=True)
structured_for_llm = {
    'total_mismatch': math_result.get('total_mismatch', False),
    'ela_anomaly': image_result.get('ela', {}).get('compression_diff', 0) > 8,
    'copy_move': image_result.get('copy_move', {}).get('pair_count', 0) > 0,
    'date_inconsistency': len(text_result.get('checks', {}).get('date_format', {}).get('formats_found', [])) > 1,
    'risk_score': risk_score,
    'risk_level': risk_level,
    'issue_count': risk_result.get('issue_count', 0),
}
llm_summary = generate_llm_summary(structured_for_llm)
st.info(llm_summary)

# ── Download Report ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📥 Download Report</div>", unsafe_allow_html=True)
report_data = dict(db_record)
report_data['math_result'] = math_result
report_data['image_result'] = image_result
report_data['ocr_result'] = ocr_result

dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    text_report = generate_text_report(report_data)
    st.download_button(
        label="📥 Download Text Report (.txt)",
        data=text_report,
        file_name=f"docverify_{Path(file_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
with dl_col2:
    html_report = generate_html_report(report_data)
    st.download_button(
        label="📥 Download HTML Report (.html)",
        data=html_report,
        file_name=f"docverify_{Path(file_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        use_container_width=True,
    )

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.warning(f"⚠️ **Disclaimer:** {DISCLAIMER}")
if record_id:
    st.caption(f"Analysis saved to database (ID: {record_id})")
