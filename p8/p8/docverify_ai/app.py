"""DocVerify AI — Main Streamlit application entry point."""
import sys
import os
from pathlib import Path

# Ensure the project root is on the Python path so all imports resolve
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────
# Page config MUST be called first
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title='DocVerify AI — Fraud Risk Detection',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────
# Initialise database and models on first run
# ─────────────────────────────────────────────────────────
from src.database import initialize_database, get_statistics, get_all_analyses
from config import APP_NAME, APP_DESCRIPTION, RISK_COLORS, RISK_ICONS, MODELS_DIR

initialize_database()

# Auto-train models if they don't exist
def _ensure_models():
    from config import ANOMALY_MODEL_PATH, SCALER_PATH
    if not ANOMALY_MODEL_PATH.exists() or not SCALER_PATH.exists():
        try:
            from src.dataset_generator import generate_dataset, get_labeled_dataset, save_dataset_info
            from src.anomaly_detection import train_model
            from src.classification import train_classifier
            save_dataset_info()
            X, y_str, _ = get_labeled_dataset()
            X_genuine = X[y_str == 'genuine']
            train_model(X_genuine)
            train_classifier(X, y_str)
        except Exception:
            pass  # Non-critical; app works without models

_ensure_models()

# ─────────────────────────────────────────────────────────
# Custom Professional CSS (Theme-proof, High-contrast)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main app styling */
[data-testid="stAppViewContainer"] {
    background-color: #f8f9fb !important;
    color: #1e293b !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0;
}
[data-testid="stSidebar"] .sidebar-brand h1 {
    color: #ffffff !important;
    font-size: 1.35em !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .sidebar-brand p {
    color: #94a3b8 !important;
    font-size: 0.8em !important;
    margin: 4px 0 0 0 !important;
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

/* Metric Cards */
[data-testid="stMetricContainer"], [data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    padding: 10px 18px !important;
    border-radius: 8px !important;
    margin: 24px 0 14px 0 !important;
    font-size: 1.05em !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 4px rgba(79, 70, 229, 0.15) !important;
}

/* Hide default Streamlit footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand" style="text-align:center; padding: 20px 0 14px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;">
        <h1>🔍 DocVerify AI</h1>
        <p>Document Authenticity & Fraud Risk</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; color:#64748b; font-weight:700; margin-bottom:8px;'>Navigation</p>", unsafe_allow_html=True)
    st.page_link('app.py', label='Dashboard', icon='🏠')
    st.page_link('pages/1_Analyze_Document.py', label='Analyze Document', icon='📤')
    st.page_link('pages/2_Analysis_History.py', label='Analysis History', icon='📊')
    st.page_link('pages/3_How_It_Works.py', label='How It Works', icon='📖')
    st.page_link('pages/4_Settings.py', label='Settings', icon='⚙️')

    st.markdown("<div style='margin-top:40px; padding:12px; background:#1e293b; border-radius:8px; border:1px solid #334155;'>", unsafe_allow_html=True)
    st.caption("⚠️ **Risk Assessment Tool**<br>Provides indicators and anomaly scores. Always perform manual verification before making decisions.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Dashboard Content
# ─────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#0f172a; margin-bottom:2px; font-weight:800;'>🔍 DocVerify AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#64748b; font-size:1.1em; margin-bottom:18px;'>AI-powered document authenticity and fraud risk analysis dashboard.</p>",
    unsafe_allow_html=True
)

st.info(
    '⚠️ **Disclaimer:** DocVerify AI identifies potential inconsistencies and manipulation '
    'indicators. It does **NOT** certify that a document is genuine or fraudulent. '
    'Manual verification by a qualified professional is always recommended.',
    icon=None
)

# Statistics
stats = get_statistics()

st.markdown("<div class='section-header'>📊 Overview Statistics</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric('📄 Total Analyzed', stats['total'])
with col2:
    st.metric('✅ Low Risk', stats['LOW'])
with col3:
    st.metric('⚠️ Medium Risk', stats['MEDIUM'])
with col4:
    st.metric('🔴 High Risk', stats['HIGH'])
with col5:
    st.metric('🚨 Issues Detected', stats['total_issues'])

st.markdown("<div class='section-header'>📋 Recent Analyses</div>", unsafe_allow_html=True)

analyses = get_all_analyses()
if analyses:
    df = pd.DataFrame(analyses)
    df_display = df[['id', 'filename', 'timestamp', 'risk_score', 'risk_level', 'issue_count', 'document_type']].copy()
    df_display.columns = ['ID', 'File', 'Timestamp', 'Risk Score', 'Risk Level', 'Issues', 'Doc Type']
    
    # Safe robust datetime formatting
    try:
        dt_parsed = pd.to_datetime(df_display['Timestamp'], format='mixed', errors='coerce')
        df_display['Timestamp'] = dt_parsed.dt.strftime('%Y-%m-%d %H:%M').fillna(df_display['Timestamp'].astype(str))
    except Exception:
        df_display['Timestamp'] = df_display['Timestamp'].astype(str)

    st.dataframe(
        df_display.head(10),
        use_container_width=True,
        hide_index=True,
    )

    # Risk history chart
    if len(df) > 1:
        import plotly.express as px
        chart_df = df_display.head(15).iloc[::-1].copy()
        fig = px.bar(
            chart_df,
            x='File', y='Risk Score',
            color='Risk Level',
            color_discrete_map={
                'LOW': '#28a745', 'MEDIUM': '#eab308',
                'HIGH': '#f97316', 'VERY HIGH': '#ef4444'
            },
            title='Recent Document Risk Scores',
            labels={'Risk Score': 'Risk Score (0–100)'},
        )
        fig.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_color='#1e293b',
            xaxis_tickangle=-30,
            margin=dict(l=20, r=20, t=40, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No documents analyzed yet. Go to **📤 Analyze Document** to upload your first invoice or certificate!')
    if st.button('📤 Analyze a Document Now', type='primary'):
        st.switch_page('pages/1_Analyze_Document.py')
