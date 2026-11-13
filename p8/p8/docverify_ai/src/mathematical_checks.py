"""Mathematical consistency engine for invoices and financial documents."""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TOLERANCE = 1.0  # rupee/dollar tolerance for rounding

_CURRENCY_PREFIX = r'(?:Rs\.?|INR|\$|\u20ac|\u00a3|\u20b9)?'
_AMOUNT_PATTERN = r'(\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?)'


def _parse_amounts(text: str) -> List[float]:
    """Extract all numeric amounts from text."""
    pattern = _CURRENCY_PREFIX + r'\s*' + _AMOUNT_PATTERN
    matches = re.findall(pattern, text)
    amounts = []
    for m in matches:
        try:
            amounts.append(float(m.replace(',', '')))
        except ValueError:
            pass
    return amounts


def _extract_labeled_amount(text: str, labels: List[str]) -> Optional[float]:
    """Find an amount associated with any of the given labels."""
    for label in labels:
        pattern = r'(?i)' + re.escape(label) + r'[:\s]*' + _CURRENCY_PREFIX + r'\s*' + _AMOUNT_PATTERN
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
    return None


def _extract_tax_rate(text: str) -> Optional[float]:
    """Extract the first tax rate (%) from text."""
    pattern = r'(?:GST|VAT|TAX|SGST|CGST|IGST)[^\d]*(\d+(?:\.\d+)?)\s*%?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def check_invoice_math(text: str) -> Dict[str, Any]:
    """Attempt to verify invoice/receipt mathematics."""
    findings: List[Dict[str, Any]] = []
    score = 0.0

    subtotal = _extract_labeled_amount(
        text, [
            'subtotal', 'sub total', 'sub-total', 'subtotat', 'subtota', 'sub tot',
            'taxable amount', 'taxable value', 'net amount', 'net total', 'amount'
        ]
    )
    tax_rate = _extract_tax_rate(text)
    discount = _extract_labeled_amount(
        text, ['discount', 'disc', 'less discount']
    )
    total = _extract_labeled_amount(
        text,
        [
            'grand total', 'total amount', 'total due', 'amount due',
            'total payable', 'invoice total', 'net payable', 'totals', 'total'
        ],
    )

    has_enough_data = subtotal is not None and total is not None

    if not has_enough_data:
        return {
            'has_data': False,
            'findings': [],
            'score': 0.0,
            'details': 'Insufficient structured financial data to perform mathematical validation.',
            'extracted': {
                'subtotal': subtotal,
                'tax_rate': tax_rate,
                'discount': discount,
                'total': total,
            },
        }

    # Calculate expected total
    expected = subtotal
    if discount is not None:
        expected -= discount
    if tax_rate is not None:
        tax_amount = expected * (tax_rate / 100)
        expected += tax_amount
    else:
        # Try to infer: total - subtotal might be the tax
        implied_tax = total - subtotal
        if 0 < implied_tax < subtotal * 0.5:
            implied_rate = (implied_tax / subtotal) * 100
            logger.debug('Implied tax rate: %.2f%%', implied_rate)

    mismatch = abs(expected - total) > TOLERANCE
    if mismatch:
        score = min(1.0, abs(expected - total) / max(total, 1) * 3)
        findings.append({
            'type': 'total_mismatch',
            'severity': 'high',
            'message': (
                'Potential total amount mismatch detected. '
                'Calculated expected total: {:,.2f}, '
                'Document total: {:,.2f}. '
                'Difference: {:,.2f}'.format(expected, total, abs(expected - total))
            ),
            'expected': expected,
            'document_value': total,
        })

    return {
        'has_data': True,
        'findings': findings,
        'score': score,
        'mismatch': mismatch,
        'extracted': {
            'subtotal': subtotal,
            'tax_rate': tax_rate,
            'discount': discount,
            'total': total,
            'expected_total': expected,
        },
        'details': 'Mathematical validation completed. Subtotal={}, Tax={}%, Total={}'.format(subtotal, tax_rate, total),
    }


def check_quantity_price(text: str) -> Dict[str, Any]:
    """Look for line-item quantity x unit price = line total mismatches."""
    # Pattern: item description, qty, unit price, total
    # e.g.: "Widget A    5    200.00    1000.00"
    pattern = r'(\d+(?:\.\d+)?)\s+(\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?)\s+(\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?)'
    matches = re.findall(pattern, text)
    mismatches = []
    for qty_s, unit_s, total_s in matches:
        try:
            qty = float(qty_s)
            unit = float(unit_s.replace(',', ''))
            total = float(total_s.replace(',', ''))
            expected = qty * unit
            if abs(expected - total) > TOLERANCE and qty > 0 and unit > 0:
                mismatches.append({
                    'qty': qty, 'unit': unit,
                    'expected': expected, 'document': total,
                })
        except ValueError:
            pass
    score = min(1.0, len(mismatches) * 0.4)
    return {
        'mismatches': mismatches,
        'score': score,
        'details': (
            ['{} potential quantity x price line mismatch(es) detected.'.format(len(mismatches))]
            if mismatches else []
        ),
    }


def run_mathematical_checks(text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """Run all mathematical checks and combine results."""
    invoice_result = check_invoice_math(text)
    qty_result = check_quantity_price(text)

    all_findings = invoice_result.get('findings', []) + [
        {'type': 'qty_price_mismatch', 'severity': 'medium', 'message': d}
        for d in qty_result.get('details', [])
    ]

    composite_score = (
        invoice_result.get('score', 0.0) * 0.7
        + qty_result.get('score', 0.0) * 0.3
    )

    return {
        'invoice': invoice_result,
        'quantity_price': qty_result,
        'findings': all_findings,
        'composite_score': min(1.0, composite_score),
        'has_data': invoice_result.get('has_data', False),
        'total_mismatch': invoice_result.get('mismatch', False),
    }
