"""Explainability module — generates human-readable finding explanations."""
import logging
from typing import Dict, Any, List
import os

from config import DISCLAIMER, LLM_ENABLED, OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Template-based explanations (no LLM required)
# ─────────────────────────────────────────────────────────

FINDING_TEMPLATES: Dict[str, Dict[str, str]] = {
    'total_mismatch': {
        'title': '⚠ Total Amount Mismatch',
        'what': 'The calculated total does not match the amount shown in the document.',
        'evidence': 'The system computed an expected total from detected subtotal, tax, and discount values. The result differs from the total displayed in the document.',
        'why': 'In genuine documents, all arithmetic should be self-consistent. A discrepancy between the calculated and displayed total may indicate that individual line items or the total amount was altered.',
        'action': 'Verify the original invoice or accounting record. Check each line item, the applicable tax rate, and the grand total independently.',
    },
    'ela_anomaly': {
        'title': '⚠ Compression Characteristic Anomaly (ELA)',
        'what': 'Regions with different compression characteristics were detected in the image.',
        'evidence': 'Error Level Analysis re-compressed the image and compared it with the original. Areas showing unusually high differences may have a different compression history from the rest of the document.',
        'why': 'When an image region has been added from another source or edited, it often retains a different compression signature. This is an indicator, not proof, of editing.',
        'action': 'Inspect the original source document or request a higher-quality version for comparison.',
    },
    'noise_anomaly': {
        'title': '⚠ Unusual Noise Pattern',
        'what': 'Image blocks with noise characteristics significantly different from surrounding areas were detected.',
        'evidence': 'The image was divided into blocks. Blocks whose noise variance exceeded 2.5 standard deviations above the image median were flagged.',
        'why': 'Inserted or edited regions frequently have different noise characteristics from the surrounding original image content.',
        'action': 'Request the original physical document or a high-resolution scan from the document issuer.',
    },
    'copy_move': {
        'title': '⚠ Potentially Duplicated Region',
        'what': 'Image blocks with very high similarity were found at different locations.',
        'evidence': 'A simplified block-matching analysis found image segments that appear visually identical despite being at different positions in the document.',
        'why': 'Copy-pasting a region within a document image is a common manipulation technique (e.g., duplicating a stamp, signature, or amount). This is a basic indicator only.',
        'action': 'Manually inspect the flagged regions. Compare with the original document.',
    },
    'metadata': {
        'title': 'ℹ Metadata Observation',
        'what': 'An observation was made about the document or image metadata.',
        'evidence': 'See details below.',
        'why': 'Metadata observations can indicate how and when a document was created or modified. They are informational rather than definitive indicators.',
        'action': 'Use metadata as supporting context alongside other findings.',
    },
    'text': {
        'title': '⚠ Text Inconsistency',
        'what': 'A potential textual inconsistency was detected.',
        'evidence': 'See details below.',
        'why': 'Inconsistent text formatting, dates, or numbers may indicate the document was created by combining content from multiple sources.',
        'action': 'Review the flagged text sections and compare with the original document.',
    },
    'qty_price_mismatch': {
        'title': '⚠ Quantity × Price Mismatch',
        'what': 'A line-item total does not match the product of quantity and unit price.',
        'evidence': 'Detected numeric triplets were checked: quantity × unit price ≠ line total.',
        'why': 'Arithmetic inconsistencies in line items may indicate that an amount was changed after the document was created.',
        'action': 'Verify each line item independently with the original supplier or issuer.',
    },
    'date_format': {
        'title': '⚠ Date Format Inconsistency',
        'what': 'Multiple date formats were found within the same document.',
        'evidence': 'Extracted dates use more than one format (e.g., DD/MM/YYYY and YYYY-MM-DD).',
        'why': 'Genuine documents typically use a consistent date format throughout. Mixed formats can indicate that dates were individually typed or modified.',
        'action': 'Verify all dates against the original transaction records.',
    },
}


def get_template_explanation(finding_type: str, message: str) -> Dict[str, str]:
    """Return a template-based explanation for a finding type."""
    template = FINDING_TEMPLATES.get(finding_type, FINDING_TEMPLATES.get('text', {}))
    return {
        'title': template.get('title', '⚠ Potential Issue'),
        'what': template.get('what', 'An anomaly was detected.'),
        'evidence': message or template.get('evidence', ''),
        'why': template.get('why', 'This may indicate a potential inconsistency.'),
        'action': template.get('action', 'Perform manual verification.'),
    }


def generate_explanations(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate an explanation for each finding."""
    explained = []
    for f in findings:
        ftype = f.get('type', 'text')
        message = f.get('message', '')
        exp = get_template_explanation(ftype, message)
        exp['severity'] = f.get('severity', 'low')
        exp['raw_type'] = ftype
        explained.append(exp)
    return explained


# ─────────────────────────────────────────────────────────
# Optional LLM summary
# ─────────────────────────────────────────────────────────

def _build_llm_prompt(structured_findings: Dict[str, Any]) -> str:
    """Build a plain-text prompt for LLM summarisation."""
    lines = ['You are a document verification assistant. Below are structured technical findings from an automated document analysis system.']
    lines.append('Summarise these findings in clear, plain English for a non-technical user.')
    lines.append('Do NOT say the document is fake or genuine. Say findings are indicators only and recommend human verification.')
    lines.append('Keep the summary under 200 words.\n')
    lines.append('FINDINGS:')
    for k, v in structured_findings.items():
        lines.append(f'  {k}: {v}')
    return '\n'.join(lines)


def generate_llm_summary(structured_findings: Dict[str, Any]) -> str:
    """Try to generate an LLM-powered summary. Falls back to template on failure."""
    if not LLM_ENABLED:
        return _fallback_summary(structured_findings)

    prompt = _build_llm_prompt(structured_findings)

    if LLM_PROVIDER == 'openai' and OPENAI_API_KEY:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning('OpenAI summary failed: %s', exc)

    if LLM_PROVIDER == 'gemini' and GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            logger.warning('Gemini summary failed: %s', exc)

    return _fallback_summary(structured_findings)


def _fallback_summary(structured_findings: Dict[str, Any]) -> str:
    """Generate a rule-based plain-English summary without an LLM."""
    parts = ['The automated analysis identified the following potential concerns:']

    if structured_findings.get('total_mismatch'):
        parts.append(
            '• A discrepancy was detected between the calculated total and the '
            'amount displayed in the document.'
        )
    if structured_findings.get('ela_anomaly'):
        parts.append(
            '• Regions with different image compression characteristics were found, '
            'which may indicate that parts of the image have different origins.'
        )
    if structured_findings.get('copy_move'):
        parts.append(
            '• Potentially duplicated image regions were detected.'
        )
    if structured_findings.get('date_inconsistency'):
        parts.append(
            '• Date format inconsistencies were found within the document.'
        )
    if not parts[1:]:
        parts.append('• No significant anomalies were detected.')

    parts.append(
        '\n⚠ These are automated indicators. Manual verification by a qualified '
        'professional is always recommended before taking any action based on this report.'
    )
    return '\n'.join(parts)
