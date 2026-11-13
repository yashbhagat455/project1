"""PDF processing module using PyMuPDF."""
import io
import logging
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DPI = 150  # render DPI


def extract_pdf_info(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata and basic info from a PDF."""
    info: Dict[str, Any] = {
        'page_count': 0,
        'metadata': {},
        'has_text_layer': False,
        'is_encrypted': False,
        'error': None,
    }
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        info['page_count'] = doc.page_count
        info['metadata'] = doc.metadata or {}
        info['is_encrypted'] = doc.is_encrypted
        # Check if text layer exists
        for page in doc:
            text = page.get_text('text').strip()
            if text:
                info['has_text_layer'] = True
                break
        doc.close()
    except Exception as exc:
        logger.error('PDF info extraction failed: %s', exc)
        info['error'] = str(exc)
    return info


def render_pdf_pages(
    pdf_bytes: bytes, dpi: int = DPI, max_pages: int = 10
) -> List[Image.Image]:
    """Render PDF pages as PIL images."""
    images: List[Image.Image] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        n = min(doc.page_count, max_pages)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page_num in range(n):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
    except Exception as exc:
        logger.error('PDF page rendering failed: %s', exc)
    return images


def extract_pdf_text_by_page(pdf_bytes: bytes) -> List[str]:
    """Extract text from each page of a PDF."""
    texts: List[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        for page in doc:
            texts.append(page.get_text('text'))
        doc.close()
    except Exception as exc:
        logger.error('PDF text extraction failed: %s', exc)
    return texts


def extract_pdf_metadata(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract and analyze PDF metadata for anomalies."""
    meta: Dict[str, Any] = {
        'raw': {},
        'anomalies': [],
        'anomaly_score': 0.0,
    }
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        raw = doc.metadata or {}
        meta['raw'] = raw
        doc.close()

        # Check for missing standard fields
        if not raw.get('creator'):
            meta['anomalies'].append('Missing creator field in metadata')
        if not raw.get('producer'):
            meta['anomalies'].append('Missing producer field in metadata')

        # Check for modification date != creation date as a soft indicator
        created = raw.get('creationDate', '')
        modified = raw.get('modDate', '')
        if created and modified and created != modified:
            meta['anomalies'].append(
                f'PDF modification date differs from creation date '
                f'(created: {created[:16]}, modified: {modified[:16]})'
            )

        # Score proportional to anomaly count
        meta['anomaly_score'] = min(1.0, len(meta['anomalies']) * 0.3)
    except Exception as exc:
        logger.error('PDF metadata extraction failed: %s', exc)
        meta['anomalies'].append(f'Metadata extraction error: {exc}')
    return meta


def get_pdf_font_info(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract font usage information from a PDF."""
    result: Dict[str, Any] = {'fonts': [], 'font_count': 0, 'variance_score': 0.0}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        font_names = set()
        for page in doc:
            for block in page.get_text('dict').get('blocks', []):
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        font_names.add(span.get('font', 'unknown'))
        doc.close()
        result['fonts'] = list(font_names)
        result['font_count'] = len(font_names)
        # More than 3 distinct fonts in a simple document may be suspicious
        if result['font_count'] > 3:
            result['variance_score'] = min(1.0, (result['font_count'] - 3) * 0.2)
    except Exception as exc:
        logger.error('Font info extraction failed: %s', exc)
    return result
