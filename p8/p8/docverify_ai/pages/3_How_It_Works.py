"""Page 3: How It Works — transparent explanation of the document verification pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title='How It Works — DocVerify AI',
    page_icon='📖',
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

/* Step Card */
.step-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin: 14px 0 !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.04) !important;
}
.step-header {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.step-num {
    background: #4f46e5;
    color: #ffffff !important;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.95em;
    margin-right: 12px;
    flex-shrink: 0;
}
.step-title {
    color: #0f172a !important;
    font-size: 1.15em !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
.step-desc {
    color: #334155 !important;
    font-size: 0.95em !important;
    line-height: 1.6 !important;
    margin-bottom: 12px !important;
}
.tag-badge {
    background-color: #eef2ff !important;
    color: #4338ca !important;
    border: 1px solid #c7d2fe !important;
    padding: 4px 12px !important;
    border-radius: 16px !important;
    font-size: 0.8em !important;
    font-weight: 600 !important;
    margin-right: 6px !important;
    display: inline-block !important;
}

/* Limitation card */
.limitation-box {
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    border-left: 5px solid #f59e0b !important;
    border-radius: 8px !important;
    padding: 20px 24px !important;
    margin: 20px 0 !important;
    color: #78350f !important;
}
.limitation-box h4 {
    color: #92400e !important;
    margin-top: 0 !important;
    font-size: 1.15em !important;
}
.limitation-box ul {
    margin-bottom: 0 !important;
    color: #78350f !important;
}
.limitation-box li {
    margin: 6px 0 !important;
}

/* Header */
.section-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    padding: 10px 18px !important;
    border-radius: 8px !important;
    margin: 24px 0 14px 0 !important;
    font-size: 1.05em !important;
    font-weight: 600 !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 14px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;">
        <h2 style='color:#ffffff; margin:0; font-size:1.35em;'>🔍 DocVerify AI</h2>
        <p style='color:#94a3b8; font-size:0.8em; margin:4px 0 0;'>How It Works</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link('app.py', label='Dashboard', icon='🏠')
    st.page_link('pages/1_Analyze_Document.py', label='Analyze Document', icon='📤')
    st.page_link('pages/2_Analysis_History.py', label='Analysis History', icon='📊')
    st.page_link('pages/3_How_It_Works.py', label='How It Works', icon='📖')
    st.page_link('pages/4_Settings.py', label='Settings', icon='⚙️')

st.markdown("<h2 style='color:#0f172a; font-weight:800;'>📖 How DocVerify AI Works</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; font-size:1.05em;'>A transparent look at the multi-layered document verification pipeline.</p>", unsafe_allow_html=True)

# ── Visual Pipeline Banner ────────────────────────────────────
st.markdown("<div class='section-header'>🔄 End-to-End Analysis Pipeline</div>", unsafe_allow_html=True)

pipeline_html = """
<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin-bottom:24px;">
    <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px;">
        <div style="background:#f1f5f9; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#334155; flex:1; min-width:110px;">1. Upload</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#f1f5f9; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#334155; flex:1; min-width:110px;">2. OCR Engine</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#f1f5f9; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#334155; flex:1; min-width:110px;">3. Text Analysis</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#f1f5f9; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#334155; flex:1; min-width:110px;">4. Math Check</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#f1f5f9; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#334155; flex:1; min-width:110px;">5. Image Forensics</div>
    </div>
    <div style="text-align:center; margin:10px 0; color:#94a3b8; font-weight:bold;">⬇</div>
    <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px;">
        <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#4338ca; flex:1; min-width:110px;">6. Features (12)</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#4338ca; flex:1; min-width:110px;">7. Isolation Forest</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#4338ca; flex:1; min-width:110px;">8. Risk Engine</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:10px 14px; text-align:center; font-weight:600; color:#4338ca; flex:1; min-width:110px;">9. Explainability</div>
        <div style="color:#94a3b8; font-weight:bold;">➔</div>
        <div style="background:#dcfce7; border:1px solid #86efac; border-radius:8px; padding:10px 14px; text-align:center; font-weight:700; color:#15803d; flex:1; min-width:110px;">10. Report Output</div>
    </div>
</div>
"""
st.markdown(pipeline_html, unsafe_allow_html=True)

# ── Detailed Step Cards ───────────────────────────────────────
st.markdown("<div class='section-header'>🔬 Detailed Pipeline Components</div>", unsafe_allow_html=True)

steps = [
    {
        'n': '1', 'icon': '📤', 'title': 'Document Upload & Ingestion',
        'desc': 'Accepts PDF, PNG, JPG, or JPEG documents. Files are read securely into memory for processing. No file is permanently retained or transferred to external servers.',
        'tech': ['Streamlit', 'Python Memory Buffer'],
    },
    {
        'n': '2', 'icon': '📄', 'title': 'PDF & Image Preprocessing',
        'desc': 'PDFs are processed via PyMuPDF to extract embedded text layers, document metadata, and font dictionaries, while rendering individual pages at 150 DPI for visual forensic analysis. Images are normalized and contrast-enhanced for OCR.',
        'tech': ['PyMuPDF', 'Pillow', 'OpenCV'],
    },
    {
        'n': '3', 'icon': '🔤', 'title': 'OCR & Entity Extraction',
        'desc': 'Optical Character Recognition extracts text from document imagery. The engine uses Tesseract with automatic fallback to EasyOCR. Regex patterns extract dates, currency values, invoice numbers, and tax percentages.',
        'tech': ['pytesseract', 'EasyOCR', 'Regular Expressions'],
    },
    {
        'n': '4', 'icon': '🔍', 'title': 'Text Consistency Analysis',
        'desc': 'Analyzes the extracted text for subtle editing indicators: mixed date formats (e.g. DD/MM/YYYY mixed with YYYY-MM-DD), impossible calendar dates, excessive double-spacing, duplicated lines, and sudden capitalization shifts.',
        'tech': ['String Forensics', 'Heuristic Rules'],
    },
    {
        'n': '5', 'icon': '🧮', 'title': 'Mathematical Consistency Engine',
        'desc': 'For invoices, bills, and purchase orders, extracts Subtotal, Tax/GST rate, Discount, and Grand Total. It computes expected totals and checks for line-item quantity × unit price arithmetic discrepancies.',
        'tech': ['Arithmetic Validator', 'Tax Calculation Logic'],
    },
    {
        'n': '6', 'icon': '🔬', 'title': 'Image Forensic Analysis',
        'desc': 'Applies four practical image forensic indicators:<br>• <strong>Error Level Analysis (ELA):</strong> Identifies regions with different JPEG compression histories.<br>• <strong>Noise Variance:</strong> Flags image blocks with noise significantly diverging from image median.<br>• <strong>Edge Consistency:</strong> Measures sharpness and edge distribution across regions.<br>• <strong>Copy-Move Detection:</strong> Detects identical or cloned image patches using DCT block matching.',
        'tech': ['Error Level Analysis (ELA)', 'Laplacian Noise', 'DCT Block Matching'],
    },
    {
        'n': '7', 'icon': '🤖', 'title': 'Feature Extraction (12 Signals)',
        'desc': 'Constructs a standardized 12-dimensional vector encompassing: text density, font variance, alignment variance, number of fonts, noise variance, compression difference, metadata anomaly score, date format variance, total arithmetic mismatch, OCR confidence, and suspicious region count.',
        'tech': ['NumPy Feature Vector', 'StandardScaler'],
    },
    {
        'n': '8', 'icon': '🌲', 'title': 'Isolation Forest Anomaly Detection',
        'desc': 'An unsupervised Isolation Forest algorithm (trained on genuine document vectors) computes an anomaly score based on feature isolation depth. Documents with atypical combinations of characteristics receive higher anomaly ratings.',
        'tech': ['scikit-learn', 'IsolationForest (contamination=0.15)'],
    },
    {
        'n': '9', 'icon': '⚖️', 'title': 'Weighted Risk Engine',
        'desc': 'Combines the ML anomaly score (30%), image forensic score (25%), mathematical validation (20%), text consistency (20%), and metadata score (5%) into a calibrated 0–100 risk score classified into LOW, MEDIUM, HIGH, or VERY HIGH.',
        'tech': ['Multi-factor Risk Fusion', 'Calibrated Scoring'],
    },
    {
        'n': '10', 'icon': '💬', 'title': 'Explainability & Report Generation',
        'desc': 'Translates technical forensic signals into clear, actionable findings formatted as What Was Detected, Evidence, Why It Matters, and Recommended Verification. Generates downloadable TXT and HTML analysis reports.',
        'tech': ['Explainable AI (XAI)', 'Template Engine', 'Optional LLM Summary'],
    },
]

for s in steps:
    tech_html = ''.join(f'<span class="tag-badge">{t}</span>' for t in s['tech'])
    card_html = f"""
    <div class="step-card">
        <div class="step-header">
            <span class="step-num">{s['n']}</span>
            <h3 class="step-title">{s['icon']} {s['title']}</h3>
        </div>
        <div class="step-desc">{s['desc']}</div>
        <div>{tech_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# ── Why AI Cannot Guarantee Authenticity ──────────────────────
st.markdown("<div class='section-header'>⚠️ Why AI Cannot Guarantee Authenticity</div>", unsafe_allow_html=True)

st.markdown("""
<div class="limitation-box">
    <h4>Important System Limitations & Ethical Notice</h4>
    <p>DocVerify AI is designed as a <strong>decision-support and risk-triaging tool</strong>. It identifies potential indicators of inconsistency and manipulation, but cannot legally certify authenticity:</p>
    <ul>
        <li><strong>No External Ground Truth:</strong> Automated systems cannot contact the document's original issuer or access proprietary ERP records to cross-reference transactions.</li>
        <li><strong>Compression Artifacts vs. Manipulation:</strong> Resaving or transmitting an image via messaging apps (e.g., WhatsApp, Telegram) introduces legitimate compression artifacts that can trigger ELA or noise indicators.</li>
        <li><strong>OCR Variability:</strong> Poor document resolution, stylized fonts, or handwritten text can cause OCR transcription errors that mimic textual or arithmetic anomalies.</li>
        <li><strong>Sophisticated Forgeries:</strong> High-end document alterations generated using identical fonts, vector graphics, and matched noise profiles may leave minimal statistical footprints.</li>
    </ul>
    <p style="margin-top:12px; font-weight:600;">
        Recommendation: Use DocVerify AI findings as indicators to guide manual auditing by a qualified professional. Never take irreversible legal or financial actions solely based on an automated score.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Risk Score Bands ──────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Risk Bands & Action Matrix</div>", unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)
with b1:
    st.markdown("""
    <div style='background:#f0fdf4; border:2px solid #22c55e; border-radius:10px; padding:16px; text-align:center;'>
        <span style='font-size:1.5em;'>✅</span>
        <h4 style='color:#15803d; margin:6px 0;'>LOW RISK</h4>
        <div style='font-size:1.6em; font-weight:800; color:#15803d;'>0 – 30</div>
        <p style='color:#166534; font-size:0.85em; margin-top:8px;'>Internal consistency verified. No suspicious anomalies detected.</p>
    </div>
    """, unsafe_allow_html=True)

with b2:
    st.markdown("""
    <div style='background:#fefce8; border:2px solid #eab308; border-radius:10px; padding:16px; text-align:center;'>
        <span style='font-size:1.5em;'>⚠️</span>
        <h4 style='color:#a16207; margin:6px 0;'>MEDIUM RISK</h4>
        <div style='font-size:1.6em; font-weight:800; color:#a16207;'>31 – 60</div>
        <p style='color:#854d0e; font-size:0.85em; margin-top:8px;'>Minor inconsistencies found (e.g. date format or slight compression variation).</p>
    </div>
    """, unsafe_allow_html=True)

with b3:
    st.markdown("""
    <div style='background:#fff7ed; border:2px solid #f97316; border-radius:10px; padding:16px; text-align:center;'>
        <span style='font-size:1.5em;'>🔶</span>
        <h4 style='color:#c2410c; margin:6px 0;'>HIGH RISK</h4>
        <div style='font-size:1.6em; font-weight:800; color:#c2410c;'>61 – 80</div>
        <p style='color:#9a3412; font-size:0.85em; margin-top:8px;'>Multiple strong indicators detected (e.g. arithmetic mismatch or copy-move).</p>
    </div>
    """, unsafe_allow_html=True)

with b4:
    st.markdown("""
    <div style='background:#fef2f2; border:2px solid #ef4444; border-radius:10px; padding:16px; text-align:center;'>
        <span style='font-size:1.5em;'>🚨</span>
        <h4 style='color:#b91c1c; margin:6px 0;'>VERY HIGH RISK</h4>
        <div style='font-size:1.6em; font-weight:800; color:#b91c1c;'>81 – 100</div>
        <p style='color:#991b1b; font-size:0.85em; margin-top:8px;'>Severe anomalies across multiple forensic layers. Manual audit required.</p>
    </div>
    """, unsafe_allow_html=True)
