"""Risk Engine — combines all analysis signals into a final risk score."""
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

RISK_BANDS = [
    (81, 100, 'VERY HIGH'),
    (61, 80,  'HIGH'),
    (31, 60,  'MEDIUM'),
    (0,  30,  'LOW'),
]

WEIGHTS = {
    'anomaly':  0.30,
    'image':    0.25,
    'text':     0.20,
    'math':     0.20,
    'metadata': 0.05,
}


def score_to_band(score: float) -> str:
    """Return risk band label for a 0–100 score."""
    for lo, hi, label in RISK_BANDS:
        if lo <= score <= hi:
            return label
    return 'LOW'


def compute_risk_score(
    anomaly_result: Dict[str, Any],
    image_result: Dict[str, Any],
    text_result: Dict[str, Any],
    math_result: Dict[str, Any],
    pdf_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine component scores into a 0–100 project-defined risk score.

    IMPORTANT: This is a project-defined risk indicator, NOT a certified
    probability of fraud.  Human verification is always recommended.
    """
    anomaly_score = float(anomaly_result.get('anomaly_score', 0.0))
    image_score   = float(image_result.get('composite_score', 0.0))
    text_score    = float(text_result.get('composite_score', 0.0))
    math_score    = float(math_result.get('composite_score', 0.0))
    meta_score    = float(pdf_metadata.get('anomaly_score', 0.0))

    weighted = (
        anomaly_score * WEIGHTS['anomaly']
        + image_score * WEIGHTS['image']
        + text_score  * WEIGHTS['text']
        + math_score  * WEIGHTS['math']
        + meta_score  * WEIGHTS['metadata']
    )

    # Scale to 0–100 and clamp
    risk_score = float(min(100.0, max(0.0, weighted * 100)))

    # Collect all findings
    findings: List[Dict[str, Any]] = []
    findings.extend(image_result.get('findings', []))
    findings.extend(math_result.get('findings', []))
    findings.extend(
        [{'type': 'text', 'severity': 'low', 'message': d}
         for d in text_result.get('all_details', [])]
    )
    findings.extend(
        [{'type': 'metadata', 'severity': 'low', 'message': a}
         for a in pdf_metadata.get('anomalies', [])]
    )

    risk_level = score_to_band(risk_score)

    # Build major findings list (top concerns)
    major = [f['message'] for f in findings if f.get('severity') in ('high', 'medium')][:5]

    return {
        'risk_score': round(risk_score, 1),
        'risk_level': risk_level,
        'issue_count': len(findings),
        'findings': findings,
        'major_findings': major,
        'component_scores': {
            'anomaly':  round(anomaly_score * 100, 1),
            'image':    round(image_score * 100, 1),
            'text':     round(text_score * 100, 1),
            'math':     round(math_score * 100, 1),
            'metadata': round(meta_score * 100, 1),
        },
    }
