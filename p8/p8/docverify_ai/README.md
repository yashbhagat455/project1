# DocVerify AI

## AI-Powered Document Authenticity and Fraud Risk Detection System

> ⚠️ **Important Disclaimer:** DocVerify AI identifies potential inconsistencies and manipulation indicators. It does NOT certify that a document is genuine or fraudulent. All findings are indicators only. Manual verification by a qualified professional is always recommended.

---

## 1. Project Overview

DocVerify AI is a complete, functional Python + Streamlit application that analyzes documents for potential signs of manipulation, fraud, or internal inconsistency.

The system produces a **risk assessment**, not a verdict. Output is worded as:
> "Potential inconsistencies detected. Manual verification is recommended."

---

## 2. Problem Statement

Document fraud is a significant challenge in organizations that process:
- Invoices and purchase orders
- Salary slips and financial statements
- College certificates and transcripts
- Utility and service bills
- Payment receipts and quotations

Manual verification is time-consuming. Automated tools can help prioritize which documents warrant closer human scrutiny.

---

## 3. Objectives

- Detect potential textual inconsistencies in documents
- Identify mathematical discrepancies in financial documents
- Analyze image-level manipulation indicators
- Produce an explainable, scored risk assessment
- Store and track analysis history
- Provide a professional, production-quality UI

---

## 4. Features

| Feature | Description |
|---------|-------------|
| 📄 PDF Support | Multi-page PDF processing via PyMuPDF |
| 🔤 OCR | Tesseract + EasyOCR with fallback |
| 🧮 Math Checks | Invoice arithmetic validation |
| 🔬 Image Analysis | ELA, noise, copy-move, metadata |
| 🤖 ML Anomaly Detection | Isolation Forest on 12-feature vector |
| 📊 Risk Engine | Weighted 0–100 composite score |
| 🗨️ Explainability | Per-finding plain-language explanations |
| 🖴 History | SQLite-backed analysis history |
| 📥 Reports | Downloadable text and HTML reports |
| 🤖 LLM Summary | Optional OpenAI/Gemini summarisation |

---

## 5. System Architecture

```
docverify_ai/
│
├── app.py                  # Dashboard entry point
├── config.py               # Configuration constants
├── requirements.txt
├── README.md
│
├── pages/
│   ├── 1_Analyze_Document.py   # Main analysis workflow
│   ├── 2_Analysis_History.py   # History browser
│   ├── 3_How_It_Works.py       # Explanatory page
│   └── 4_Settings.py           # Settings and model management
│
├── src/
│   ├── preprocessing.py        # Image loading and enhancement
│   ├── pdf_processor.py        # PDF extraction and rendering
│   ├── ocr.py                  # OCR with Tesseract/EasyOCR
│   ├── text_analysis.py        # Text consistency checks
│   ├── mathematical_checks.py  # Invoice math validation
│   ├── image_analysis.py       # ELA, noise, edge, copy-move
│   ├── feature_extraction.py   # 12-feature ML vector
│   ├── anomaly_detection.py    # Isolation Forest
│   ├── classification.py       # Random Forest classifier
│   ├── risk_engine.py          # Combined risk scoring
│   ├── explanation.py          # Per-finding explanations
│   ├── visualization.py        # Overlay generation
│   ├── report_generator.py     # Text/HTML report generation
│   ├── database.py             # SQLite CRUD
│   ├── dataset_generator.py    # Synthetic dataset
│   └── model_trainer.py        # Training script
│
├── models/                     # Trained model files (.pkl)
├── data/                       # Dataset info
├── uploads/                    # Temporary uploads
├── results/                    # Generated results
└── database/
    └── docverify.db            # SQLite database
```

---

## 6. ML Methodology

### Primary: Isolation Forest (Anomaly Detection)
- **Algorithm:** scikit-learn IsolationForest
- **Training data:** Synthetic genuine document feature vectors
- **Contamination:** 15% (expected fraction of anomalies)
- **Purpose:** Identify documents whose feature profile diverges from typical genuine documents

### Secondary: Random Forest Classifier (Optional)
- **Algorithm:** scikit-learn RandomForestClassifier
- **Classes:** genuine / manipulated
- **Training data:** Synthetic labeled dataset
- **Purpose:** Supplementary classification label

---

## 7. Dataset

> ⚠️ **This is a synthetic dataset for development and testing. It does NOT contain real documents.**

The dataset is generated programmatically in `src/dataset_generator.py`:

- **60 genuine samples:** Feature vectors with low anomaly values
- **40 manipulated samples:** Genuine base + controlled perturbations

Manipulation types simulated:
| Type | Features Affected |
|------|------------------|
| Total mismatch | `total_mismatch` → 1.0 |
| ELA anomaly | `compression_difference` + `suspicious_region_count` ↑ |
| Font inconsistency | `font_variance` + `number_of_fonts` ↑ |
| Noise anomaly | `image_noise_variance` ↑ |
| Metadata anomaly | `metadata_anomaly` ↑ |
| Multiple | All of the above |

---

## 8. Feature Engineering

12 features are extracted per document:

| # | Feature | Source |
|---|---------|--------|
| 1 | `text_density` | OCR word count |
| 2 | `font_variance` | PDF font list |
| 3 | `font_size_variance` | PDF font info |
| 4 | `alignment_variance` | Edge asymmetry |
| 5 | `number_of_fonts` | PDF font count |
| 6 | `image_noise_variance` | Noise block analysis |
| 7 | `compression_difference` | ELA score |
| 8 | `metadata_anomaly` | EXIF + PDF metadata |
| 9 | `date_format_variance` | Date extraction |
| 10 | `total_mismatch` | Invoice math check |
| 11 | `ocr_confidence` | Tesseract confidence |
| 12 | `suspicious_region_count` | Noise + copy-move blocks |

---

## 9. Model Training

Models are automatically trained on first run. To manually retrain:

```bash
# Via Settings page
# Go to Settings → ML Models → Retrain Models

# Or via command line:
python -m src.model_trainer
```

---

## 10. Evaluation

Model evaluation is run on the synthetic test split (25% holdout). Metrics are displayed on the Settings page after retraining.

> ⚠️ **Note:** "Prototype evaluation — limited dataset." The synthetic dataset is small. Results should not be interpreted as real-world performance benchmarks.

---

## 11. Installation

### Prerequisites

1. **Python 3.9+**
2. **Tesseract OCR** (required for primary OCR):
   - **Local installation in project parent:** `D:\internship\p8\Tesseract-OCR\tesseract.exe` (automatically detected and used by DocVerify AI)
   - **Alternative Windows:** https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS:** `brew install tesseract`
   - **Linux:** `sudo apt install tesseract-ocr`
   - **Fallback:** If Tesseract is not installed, the application automatically falls back to EasyOCR or PDF direct text extraction.

### Setup

```bash
# 1. Navigate to project directory
cd docverify_ai

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 12. Usage

1. Open the app at `http://localhost:8501`
2. Navigate to **📤 Analyze Document**
3. Upload a PDF or image
4. Click **🔍 Analyze Document**
5. Review findings, visualizations, and risk score
6. Download the report
7. View history in **📊 Analysis History**

---

## 13. Configuration

Optional environment variables:

```bash
# LLM (optional)
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export LLM_PROVIDER="openai"  # or "gemini"

# OCR
export OCR_ENGINE="tesseract"  # or "easyocr"
```

Or create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

---

## 14. Limitations

- **Not forensically certified:** Results are risk indicators, not expert opinions
- **ELA limitations:** JPEG compression naturally creates ELA variations; high scores do not definitively indicate editing
- **OCR accuracy:** Handwritten text and low-resolution scans may produce unreliable OCR output
- **Small training set:** The synthetic dataset is limited; real-world performance may vary
- **No database access:** The system cannot verify documents against issuing organization records
- **Mathematical checks:** Only applicable to structured financial documents with detectable numbers

---

## 15. Privacy Considerations

- Documents are processed in memory; content is not permanently stored
- Only metadata and scores are saved to SQLite
- No document content is sent to external servers unless LLM is explicitly enabled
- API keys must be provided via environment variables or Streamlit secrets
- Delete analysis records at any time from the History page

---

## 16. Future Improvements

- [ ] Digital signature verification
- [ ] QR code / barcode validation
- [ ] Trained models on real labeled datasets
- [ ] Template-based document verification (known invoice formats)
- [ ] Batch processing mode
- [ ] Multi-language OCR support
- [ ] REST API endpoint for programmatic access
- [ ] Integration with document management systems

---

*Built with Python, Streamlit, OpenCV, scikit-learn, PyMuPDF, and Tesseract.*
