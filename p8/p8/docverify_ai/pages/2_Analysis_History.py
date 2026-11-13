"""Page 2: Analysis History — browse and inspect previous document analyses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime

from src.database import get_all_analyses, get_analysis_by_id, delete_analysis, clear_all_analyses, initialize_database
from src.report_generator import generate_text_report, generate_html_report
from config import RISK_COLORS, RISK_ICONS

initialize_database()

st.set_page_config(
    page_title='Analysis History — DocVerify AI',
    page_icon='📊',
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
        <p style='color:#94a3b8; font-size:0.8em; margin:4px 0 0;'>Analysis History</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link('app.py', label='Dashboard', icon='🏠')
    st.page_link('pages/1_Analyze_Document.py', label='Analyze Document', icon='📤')
    st.page_link('pages/2_Analysis_History.py', label='Analysis History', icon='📊')
    st.page_link('pages/3_How_It_Works.py', label='How It Works', icon='📖')
    st.page_link('pages/4_Settings.py', label='Settings', icon='⚙️')

st.markdown("<h2 style='color:#0f172a; font-weight:800;'>📊 Analysis History</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;'>Browse, search, inspect, and download reports from previous document analyses.</p>", unsafe_allow_html=True)

analyses = get_all_analyses()

if not analyses:
    st.info('💭 No previous analyses found. Go to **📤 Analyze Document** to run your first verification.')
    if st.button('📤 Analyze a Document', type='primary'):
        st.switch_page('pages/1_Analyze_Document.py')
    st.stop()

# ── Safe Dataframe Construction ──────────────────────────────
df = pd.DataFrame(analyses)

# Format timestamps safely without throwing ValueError
try:
    df['parsed_ts'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df['formatted_time'] = df['parsed_ts'].dt.strftime('%Y-%m-%d %H:%M').fillna(df['timestamp'].astype(str))
except Exception:
    df['formatted_time'] = df['timestamp'].astype(str)

# ── Summary Charts ───────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Risk Analytics</div>", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    level_counts = df['risk_level'].value_counts().reset_index()
    level_counts.columns = ['Risk Level', 'Count']
    fig_pie = px.pie(
        level_counts, values='Count', names='Risk Level',
        color='Risk Level',
        color_discrete_map={
            'LOW': '#28a745', 'MEDIUM': '#eab308',
            'HIGH': '#f97316', 'VERY HIGH': '#ef4444'
        },
        title='Documents by Risk Level',
    )
    fig_pie.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='#ffffff',
        font_color='#1e293b',
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    fig_hist = px.histogram(
        df, x='risk_score', nbins=10,
        title='Risk Score Distribution (0–100)',
        labels={'risk_score': 'Risk Score'},
        color_discrete_sequence=['#4f46e5'],
    )
    fig_hist.update_layout(
        height=280,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font_color='#1e293b',
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Search & Filter Table ─────────────────────────────────────
st.markdown("<div class='section-header'>📋 All Analysis Records</div>", unsafe_allow_html=True)

filter_col1, filter_col2 = st.columns([2, 1])
with filter_col1:
    search = st.text_input('🔍 Search by document name', placeholder='Type filename to filter...')
with filter_col2:
    risk_filter = st.multiselect('Filter by Risk Level', ['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH'], default=[])

filtered_df = df.copy()
if search:
    filtered_df = filtered_df[filtered_df['filename'].str.contains(search, case=False, na=False)]
if risk_filter:
    filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]

table_df = filtered_df[['id', 'filename', 'formatted_time', 'risk_score', 'risk_level', 'issue_count', 'document_type']].copy()
table_df.columns = ['ID', 'Filename', 'Timestamp', 'Risk Score', 'Risk Level', 'Issues', 'Type']
st.dataframe(table_df, use_container_width=True, hide_index=True)

# ── Detailed Record Inspector ─────────────────────────────────
st.markdown("<div class='section-header'>🔍 Inspect Analysis Details</div>", unsafe_allow_html=True)

all_ids = [r['id'] for r in analyses]
id_options = {r['id']: f"ID #{r['id']} — {r['filename']} ({r['risk_level']}, Score: {r['risk_score']})" for r in analyses}

selected_id = st.selectbox(
    'Select a record to view details:',
    options=all_ids,
    format_func=lambda x: id_options.get(x, f"ID #{x}")
)

if selected_id:
    detail = get_analysis_by_id(selected_id)
    if detail:
        full = detail.get('full_result') or {}
        risk_level = detail.get('risk_level', 'LOW')
        risk_score = detail.get('risk_score', 0)
        risk_color = RISK_COLORS.get(risk_level, '#64748b')
        risk_icon  = RISK_ICONS.get(risk_level, '❓')

        st.markdown(f"### Analysis #{selected_id}: `{detail.get('filename')}`")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric('Risk Score', f"{risk_score:.1f} / 100")
        with m2:
            st.markdown(
                f"<div style='text-align:center; padding-top:14px;'>"
                f"<span style='background:{risk_color}; color:#ffffff; padding:8px 18px; border-radius:20px; font-weight:700;'>"
                f"{risk_icon} {risk_level}</span></div>",
                unsafe_allow_html=True
            )
        with m3:
            st.metric('Issues Detected', detail.get('issue_count', 0))
        with m4:
            st.metric('Document Type', detail.get('document_type', 'Unknown'))

        # Major Findings
        findings = full.get('findings') or detail.get('major_findings') or []
        if findings:
            st.markdown("#### 🚨 Key Findings")
            for item in findings:
                if isinstance(item, dict):
                    msg = item.get('message', '')
                    sev = item.get('severity', 'medium').upper()
                    st.markdown(f"- **[{sev}]** {msg}")
                else:
                    st.markdown(f"- {item}")
        else:
            st.success("✅ No significant issues detected for this document.")

        # Download Report Buttons for this past record
        st.markdown("#### 📥 Download Report for this Document")
        report_data = full if full else {
            'filename': detail.get('filename'),
            'timestamp': detail.get('timestamp'),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'issue_count': detail.get('issue_count', 0),
            'document_type': detail.get('document_type', 'Unknown'),
            'findings': findings,
            'component_scores': full.get('component_scores', {}),
        }
        
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            txt_rep = generate_text_report(report_data)
            st.download_button(
                label=f"📥 Download Text Report (#{selected_id})",
                data=txt_rep,
                file_name=f"report_{selected_id}_{detail.get('filename', 'doc')}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_txt_{selected_id}",
            )
        with dl_col2:
            html_rep = generate_html_report(report_data)
            st.download_button(
                label=f"📥 Download HTML Report (#{selected_id})",
                data=html_rep,
                file_name=f"report_{selected_id}_{detail.get('filename', 'doc')}.html",
                mime="text/html",
                use_container_width=True,
                key=f"dl_html_{selected_id}",
            )

        with st.expander("📄 Full Stored Record (JSON)"):
            st.json(full if full else detail)

        # Delete single record
        st.markdown("---")
        del_col1, del_col2 = st.columns([1, 4])
        with del_col1:
            if st.button(f'🗑️ Delete Record #{selected_id}', type='secondary'):
                if delete_analysis(selected_id):
                    st.success(f"Record #{selected_id} deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete record.")

# ── Danger Zone: Clear History ────────────────────────────────
st.markdown("---")
with st.expander("⚠️ Danger Zone: Clear All History"):
    st.warning("This will permanently delete ALL document verification records from SQLite.")
    if st.button("🗑️ Clear All Analysis History", type="secondary"):
        clear_all_analyses()
        st.success("All history records have been cleared.")
        st.rerun()
