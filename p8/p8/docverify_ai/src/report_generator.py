"""Downloadable report generation module."""
import io
import logging
from datetime import datetime
from typing import Dict, Any

from config import APP_NAME, APP_VERSION, DISCLAIMER

logger = logging.getLogger(__name__)


def generate_text_report(result: Dict[str, Any]) -> str:
    """Generate a plain-text analysis report."""
    lines = []
    sep = '=' * 60
    thin = '-' * 60

    lines.append(sep)
    lines.append(f'{APP_NAME} v{APP_VERSION}')
    lines.append('Document Analysis Report')
    lines.append(sep)
    lines.append('')
    lines.append(f"File         : {result.get('filename', 'Unknown')}")
    lines.append(f"Date         : {result.get('timestamp', datetime.now().isoformat())}")
    lines.append(f"Document Type: {result.get('document_type', 'Unknown')}")
    lines.append('')
    lines.append(thin)
    lines.append('RISK ASSESSMENT')
    lines.append(thin)
    risk_score = result.get('risk_score', 0)
    risk_level = result.get('risk_level', 'LOW')
    lines.append(f'Risk Score : {risk_score:.1f} / 100')
    lines.append(f'Risk Level : {risk_level}')
    lines.append('')
    lines.append(
        'NOTE: This is a project-defined risk score, not a certified probability of fraud.'
    )
    lines.append('')
    lines.append(thin)
    lines.append('DETECTED ISSUES')
    lines.append(thin)
    findings = result.get('findings', [])
    if not findings:
        lines.append('No significant issues detected.')
    else:
        for i, f in enumerate(findings, 1):
            severity = f.get('severity', 'low').upper()
            message = f.get('message', '')
            lines.append(f"{i}. [{severity}] {message}")
    lines.append('')

    lines.append(thin)
    lines.append('COMPONENT SCORES (0-100)')
    lines.append(thin)
    component_scores = result.get('component_scores', {})
    for comp, score in component_scores.items():
        lines.append(f"  {comp.capitalize():<20}: {score:.1f}")
    lines.append('')

    lines.append(thin)
    lines.append('MATHEMATICAL CHECKS')
    lines.append(thin)
    math = result.get('math_result', {})
    if math.get('has_data'):
        extracted = math.get('invoice', {}).get('extracted', {})
        lines.append(f"  Subtotal        : {extracted.get('subtotal', 'N/A')}")
        lines.append(f"  Tax Rate        : {extracted.get('tax_rate', 'N/A')}%")
        lines.append(f"  Discount        : {extracted.get('discount', 'N/A')}")
        lines.append(f"  Document Total  : {extracted.get('total', 'N/A')}")
        lines.append(f"  Expected Total  : {extracted.get('expected_total', 'N/A')}")
        lines.append(f"  Total Mismatch  : {math.get('total_mismatch', False)}")
    else:
        lines.append('  Insufficient structured financial data for mathematical validation.')
    lines.append('')

    lines.append(thin)
    lines.append('IMAGE ANALYSIS SUMMARY')
    lines.append(thin)
    img = result.get('image_result', {})
    ela_diff = img.get('ela', {}).get('compression_diff', 0)
    noise_count = len(img.get('noise', {}).get('suspicious_blocks', []))
    copy_move_count = img.get('copy_move', {}).get('pair_count', 0)
    lines.append(f"  ELA Compression Difference : {ela_diff:.2f}")
    lines.append(f"  Suspicious Noise Blocks    : {noise_count}")
    lines.append(f"  Potential Copy-Move Pairs  : {copy_move_count}")
    lines.append('')

    lines.append(thin)
    lines.append('OCR SUMMARY')
    lines.append(thin)
    ocr = result.get('ocr_result', {})
    lines.append(f"  OCR Engine   : {ocr.get('engine', 'Unknown')}")
    lines.append(f"  Confidence   : {ocr.get('confidence', 0):.1f}%")
    lines.append(f"  Word Count   : {ocr.get('word_count', 0)}")
    extracted_text = (ocr.get('text') or '')[:500]
    if extracted_text:
        lines.append('  Extracted Text (first 500 chars):')
        for text_line in extracted_text.splitlines()[:15]:
            lines.append(f'    {text_line}')
    lines.append('')

    lines.append(thin)
    lines.append('DISCLAIMER')
    lines.append(thin)
    # Word-wrap disclaimer
    words = DISCLAIMER.split()
    current_line = []
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 60:
            lines.append('  ' + ' '.join(current_line[:-1]))
            current_line = [word]
    if current_line:
        lines.append('  ' + ' '.join(current_line))
    lines.append('')
    lines.append(sep)

    return '\n'.join(lines)


def generate_html_report(result: Dict[str, Any]) -> str:
    """Generate an HTML analysis report."""
    risk_level = result.get('risk_level', 'LOW')
    risk_score = result.get('risk_score', 0)
    color_map = {
        'LOW': '#28a745', 'MEDIUM': '#ffc107',
        'HIGH': '#fd7e14', 'VERY HIGH': '#dc3545',
    }
    risk_color = color_map.get(risk_level, '#6c757d')

    findings_html = ''
    for i, f in enumerate(result.get('findings', []), 1):
        sev = f.get('severity', 'low')
        sev_color = {'high': '#dc3545', 'medium': '#fd7e14', 'low': '#6c757d'}.get(sev, '#6c757d')
        findings_html += (
            f'<li><span style="color:{sev_color};font-weight:bold">'
            f'[{sev.upper()}]</span> {f.get("message", "")} </li>\n'
        )

    timestamp = result.get('timestamp', datetime.now().isoformat())
    filename = result.get('filename', 'Unknown')

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DocVerify AI Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px;
          margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #4f46e5; padding-bottom: 8px; }}
  h2 {{ color: #4f46e5; margin-top: 30px; }}
  .risk-badge {{ display: inline-block; padding: 8px 20px; border-radius: 6px;
                 font-size: 1.4em; font-weight: bold; color: white;
                 background: {risk_color}; }}
  .score {{ font-size: 2em; font-weight: bold; color: {risk_color}; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f5; }}
  .disclaimer {{ background: #fff3cd; border: 1px solid #ffc107;
                 padding: 15px; border-radius: 6px; font-size: 0.9em; }}
  .footer {{ text-align: center; color: #999; font-size: 0.85em; margin-top: 40px; }}
</style>
</head>
<body>
<h1>DocVerify AI &mdash; Document Analysis Report</h1>
<p><strong>File:</strong> {filename}<br>
   <strong>Date:</strong> {timestamp}</p>

<h2>Risk Assessment</h2>
<div class="score">{risk_score:.1f} / 100</div>&nbsp;
<div class="risk-badge">{risk_level}</div>
<p><em>This is a project-defined risk score, not a certified probability of fraud.</em></p>

<h2>Detected Issues ({len(result.get('findings', []))} total)</h2>
<ul>
{findings_html if findings_html else '<li>No significant issues detected.</li>'}
</ul>

<h2>Component Scores</h2>
<table>
  <tr><th>Component</th><th>Score (0–100)</th></tr>
  {''.join(f"<tr><td>{k.capitalize()}</td><td>{v:.1f}</td></tr>" for k, v in result.get('component_scores', {}).items())}
</table>

<div class="disclaimer">
<strong>Disclaimer:</strong> {DISCLAIMER}
</div>

<div class="footer">
  Generated by {APP_NAME} v{APP_VERSION}
</div>
</body>
</html>
"""
    return html
