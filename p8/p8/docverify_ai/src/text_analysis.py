"""Text consistency analysis module."""
import re
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────

def check_date_format_consistency(dates: List[str]) -> Dict[str, Any]:
    """Check whether extracted dates use consistent formats."""
    if len(dates) < 2:
        return {'consistent': True, 'formats_found': [], 'score': 0.0, 'details': []}

    def infer_format(d: str) -> str:
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$', d):
            return 'DD/MM/YYYY'
        if re.match(r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', d):
            return 'YYYY-MM-DD'
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2}$', d):
            return 'DD/MM/YY'
        return 'OTHER'

    formats = [infer_format(d) for d in dates]
    unique_formats = list(set(formats))
    consistent = len(unique_formats) == 1
    score = 0.0 if consistent else min(1.0, (len(unique_formats) - 1) * 0.4)
    details = []
    if not consistent:
        details.append(
            f'Multiple date formats detected: {unique_formats}. '
            'Inconsistent date formatting may indicate manual editing.'
        )
    return {
        'consistent': consistent,
        'formats_found': unique_formats,
        'score': score,
        'details': details,
    }


def check_impossible_dates(dates: List[str]) -> Dict[str, Any]:
    """Detect dates that appear implausible."""
    from datetime import datetime
    issues = []
    current_year = datetime.now().year
    for d in dates:
        # Extract year
        year_match = re.search(r'(19|20)(\d{2})', d)
        if year_match:
            year = int(year_match.group(0))
            if year > current_year + 1:
                issues.append(f'Future date detected: {d}')
            elif year < 1980:
                issues.append(f'Very old date detected: {d}')
    return {
        'issues': issues,
        'score': min(1.0, len(issues) * 0.4),
    }


def check_unusual_characters(text: str) -> Dict[str, Any]:
    """Detect unusual or unexpected characters."""
    # Look for non-standard Unicode outside common ranges
    unusual = re.findall(r'[^\x00-\x7F\u20A8\u20B9\u00A3\u20AC]', text)
    suspicious = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text)
    total = len(unusual) + len(suspicious)
    return {
        'unusual_char_count': total,
        'score': min(1.0, total * 0.05),
        'details': (
            [f'{total} unusual characters found in document text']
            if total > 5 else []
        ),
    }


def check_spacing_consistency(text: str) -> Dict[str, Any]:
    """Check for inconsistent spacing patterns."""
    double_spaces = len(re.findall(r' {2,}', text))
    mixed_newlines = len(re.findall(r'\r\n|\r(?!\n)', text))
    score = min(1.0, (double_spaces * 0.02 + mixed_newlines * 0.05))
    details = []
    if double_spaces > 10:
        details.append(f'{double_spaces} instances of multiple consecutive spaces detected.')
    return {
        'double_space_count': double_spaces,
        'score': score,
        'details': details,
    }


def check_duplicate_text(text: str) -> Dict[str, Any]:
    """Detect suspiciously duplicated lines or phrases."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    counts = Counter(lines)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    score = min(1.0, len(duplicates) * 0.3)
    details = []
    if duplicates:
        examples = list(duplicates.items())[:3]
        for line, cnt in examples:
            details.append(f'Line appears {cnt}x: "{line[:60]}..."')
    return {
        'duplicate_count': len(duplicates),
        'score': score,
        'details': details,
    }


def check_number_format_consistency(text: str) -> Dict[str, Any]:
    """Check whether numbers follow a consistent format."""
    # Find all number patterns
    with_commas = re.findall(r'\d{1,3}(?:,\d{3})+', text)
    without_commas = re.findall(r'(?<!,)\b\d{4,}\b(?!,)', text)
    score = 0.0
    details = []
    if with_commas and without_commas:
        score = 0.3
        details.append(
            'Mixed number formatting: some numbers use comma separators, others do not. '
            'This may indicate text was edited from different sources.'
        )
    return {'score': score, 'details': details}


def check_capitalization_consistency(text: str) -> Dict[str, Any]:
    """Check for unusual capitalization patterns."""
    # Look for ALL-CAPS words mixed with normal text (more than occasional)
    words = re.findall(r'\b[A-Z]{4,}\b', text)
    total_words = max(len(text.split()), 1)
    ratio = len(words) / total_words
    score = min(0.5, ratio * 2)
    details = []
    if ratio > 0.15:
        details.append(
            f'High proportion of all-caps words ({len(words)} out of {total_words}). '
            'This may indicate formatting inconsistency.'
        )
    return {'score': score, 'details': details}


# ─────────────────────────────────────────────
# Combined runner
# ─────────────────────────────────────────────

def run_text_analysis(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Run all text-consistency checks and combine results."""
    text = entities.get('raw_text', '')
    dates = entities.get('dates', [])

    results: Dict[str, Any] = {}
    all_details: List[str] = []
    composite_score = 0.0

    checks = [
        ('date_format', check_date_format_consistency(dates)),
        ('impossible_dates', check_impossible_dates(dates)),
        ('unusual_chars', check_unusual_characters(text)),
        ('spacing', check_spacing_consistency(text)),
        ('duplicates', check_duplicate_text(text)),
        ('number_format', check_number_format_consistency(text)),
        ('capitalization', check_capitalization_consistency(text)),
    ]

    weights = {
        'date_format': 0.2,
        'impossible_dates': 0.2,
        'unusual_chars': 0.1,
        'spacing': 0.05,
        'duplicates': 0.2,
        'number_format': 0.15,
        'capitalization': 0.1,
    }

    for name, check_result in checks:
        results[name] = check_result
        composite_score += check_result.get('score', 0.0) * weights.get(name, 0.1)
        all_details.extend(check_result.get('details', []))

    # Determine date_format_variance for feature extraction
    date_fmt_result = results.get('date_format', {})
    date_format_variance = len(date_fmt_result.get('formats_found', [])) / max(len(dates), 1)

    return {
        'checks': results,
        'composite_score': min(1.0, composite_score),
        'all_details': all_details,
        'issue_count': len([d for d in all_details if d]),
        'date_format_variance': date_format_variance,
    }
