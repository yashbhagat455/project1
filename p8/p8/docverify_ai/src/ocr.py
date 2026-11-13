"""OCR module supporting Tesseract and EasyOCR."""
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

import config

def _get_configured_tesseract_cmd() -> str:
    cmd = getattr(config, 'TESSERACT_CMD', None)
    if cmd:
        return cmd
    if hasattr(config, 'get_tesseract_cmd'):
        return config.get_tesseract_cmd()
    return r'D:\internship\p8\Tesseract-OCR\tesseract.exe'

TESSERACT_CMD = _get_configured_tesseract_cmd()

logger = logging.getLogger(__name__)


def configure_pytesseract():
    """Ensure pytesseract knows the location of the tesseract executable."""
    try:
        import pytesseract
        if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            tessdata = os.path.join(os.path.dirname(TESSERACT_CMD), 'tessdata')
            if os.path.isdir(tessdata) and 'TESSDATA_PREFIX' not in os.environ:
                os.environ['TESSDATA_PREFIX'] = tessdata
    except Exception as exc:
        logger.debug('pytesseract configuration error: %s', exc)


# Configure at module import
configure_pytesseract()


def _try_tesseract(pil_img: Image.Image) -> Dict[str, Any]:
    """Run Tesseract OCR and return structured results."""
    try:
        import pytesseract
        configure_pytesseract()
        data = pytesseract.image_to_data(
            pil_img, output_type=pytesseract.Output.DICT, lang='eng'
        )
        confidences = [
            int(c) for c in data['conf'] if str(c).strip() not in ('', '-1')
        ]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        raw_text = pytesseract.image_to_string(pil_img, lang='eng')
        return {
            'engine': 'tesseract',
            'text': raw_text,
            'confidence': avg_conf,
            'word_count': len([w for w in data['text'] if w.strip()]),
            'success': True,
            'error': None,
        }
    except ImportError:
        return {
            'engine': 'tesseract',
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'success': False,
            'error': 'pytesseract not installed',
        }
    except Exception as exc:
        logger.error('Tesseract OCR failed: %s', exc)
        return {
            'engine': 'tesseract',
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'success': False,
            'error': str(exc),
        }


def _try_easyocr(pil_img: Image.Image) -> Dict[str, Any]:
    """Run EasyOCR and return structured results."""
    try:
        import easyocr
        import numpy as np
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        arr = np.array(pil_img)
        results = reader.readtext(arr)
        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        raw_text = ' '.join(texts)
        avg_conf = float(np.mean(confidences) * 100) if confidences else 0.0
        return {
            'engine': 'easyocr',
            'text': raw_text,
            'confidence': avg_conf,
            'word_count': len(raw_text.split()),
            'success': True,
            'error': None,
        }
    except ImportError:
        return {
            'engine': 'easyocr',
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'success': False,
            'error': 'easyocr not installed',
        }
    except Exception as exc:
        logger.error('EasyOCR failed: %s', exc)
        return {
            'engine': 'easyocr',
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'success': False,
            'error': str(exc),
        }


def run_ocr(
    pil_img: Image.Image, engine: str = 'tesseract'
) -> Dict[str, Any]:
    """Run OCR using the selected engine, with fallback."""
    if engine == 'easyocr':
        result = _try_easyocr(pil_img)
        if not result['success']:
            logger.info('EasyOCR failed, trying Tesseract')
            result = _try_tesseract(pil_img)
    else:
        result = _try_tesseract(pil_img)
        if not result['success']:
            logger.info('Tesseract failed, trying EasyOCR')
            result = _try_easyocr(pil_img)
    return result


def run_ocr_multi_page(
    pil_images: List[Image.Image], engine: str = 'tesseract'
) -> Dict[str, Any]:
    """Run OCR on multiple pages and aggregate results."""
    all_text_parts: List[str] = []
    confidences: List[float] = []
    for i, img in enumerate(pil_images):
        r = run_ocr(img, engine=engine)
        if r['text']:
            all_text_parts.append(f'--- Page {i+1} ---\n{r["text"]}')
        if r['confidence'] > 0:
            confidences.append(r['confidence'])
    combined_text = '\n'.join(all_text_parts)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return {
        'text': combined_text,
        'confidence': avg_conf,
        'page_count': len(pil_images),
        'word_count': len(combined_text.split()),
        'success': bool(combined_text.strip()),
    }


# ─────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────

DATE_PATTERNS = [
    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
    r'\b(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b',
    r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,\s]+\d{4})\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[,\s]+\d{4})\b',
]

CURRENCY_PATTERN = r'(?:Rs\.?|INR|\$|€|£|₹)?\s*(\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?)'
INVOICE_NUMBER_PATTERN = r'(?:Invoice|Inv|Bill|Receipt|Ref|Order|PO|Quote)[\s#.:]*([A-Z0-9\-/]+)'
TAX_PATTERN = r'(?:GST|VAT|TAX|SGST|CGST|IGST)[^\d]*(\d+(?:\.\d+)?)\s*%?'


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract structured entities from raw OCR text."""
    dates: List[str] = []
    for pat in DATE_PATTERNS:
        dates.extend(re.findall(pat, text, re.IGNORECASE))
    dates = list(dict.fromkeys(dates))  # deduplicate preserving order

    currency_matches = re.findall(CURRENCY_PATTERN, text)
    numbers = []
    for m in currency_matches:
        try:
            numbers.append(float(m.replace(',', '')))
        except ValueError:
            pass

    invoice_numbers = re.findall(INVOICE_NUMBER_PATTERN, text, re.IGNORECASE)
    tax_rates = re.findall(TAX_PATTERN, text, re.IGNORECASE)

    return {
        'dates': dates,
        'currency_values': numbers,
        'invoice_numbers': invoice_numbers,
        'tax_rates': [float(t) for t in tax_rates if t],
        'raw_text': text,
    }
