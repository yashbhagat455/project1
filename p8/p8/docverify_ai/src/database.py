"""SQLite database module for DocVerify AI."""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import DB_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """Create tables if they do not exist."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                issue_count INTEGER NOT NULL,
                document_type TEXT,
                major_findings TEXT,
                full_result TEXT
            )
        """)
        conn.commit()
        logger.info('Database initialized at %s', DB_PATH)
    finally:
        conn.close()


def save_analysis(result: Dict[str, Any]) -> int:
    """Save an analysis result to the database. Return the new row id."""
    conn = get_connection()
    try:
        major_findings = json.dumps(result.get('major_findings', []))
        full_result = json.dumps(result, default=str)
        cur = conn.execute(
            """
            INSERT INTO analyses
                (filename, timestamp, risk_score, risk_level, issue_count,
                 document_type, major_findings, full_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get('filename', 'Unknown'),
                result.get('timestamp', datetime.now().isoformat()),
                result.get('risk_score', 0),
                result.get('risk_level', 'LOW'),
                result.get('issue_count', 0),
                result.get('document_type', 'Unknown'),
                major_findings,
                full_result,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_analyses() -> List[Dict[str, Any]]:
    """Return all analyses ordered by most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            'SELECT id, filename, timestamp, risk_score, risk_level, issue_count, document_type, major_findings '
            'FROM analyses ORDER BY id DESC'
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """Return the full result for a single analysis."""
    conn = get_connection()
    try:
        row = conn.execute(
            'SELECT * FROM analyses WHERE id = ?', (analysis_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        if data.get('full_result'):
            data['full_result'] = json.loads(data['full_result'])
        if data.get('major_findings'):
            data['major_findings'] = json.loads(data['major_findings'])
        return data
    finally:
        conn.close()


def delete_analysis(analysis_id: int) -> bool:
    """Delete a single analysis record."""
    conn = get_connection()
    try:
        conn.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
        conn.commit()
        return True
    except Exception as exc:
        logger.error('Failed to delete analysis %d: %s', analysis_id, exc)
        return False
    finally:
        conn.close()


def clear_all_analyses() -> None:
    """Delete all analysis records."""
    conn = get_connection()
    try:
        conn.execute('DELETE FROM analyses')
        conn.commit()
    finally:
        conn.close()


def get_statistics() -> Dict[str, Any]:
    """Return aggregate statistics."""
    conn = get_connection()
    try:
        rows = conn.execute(
            'SELECT risk_level, COUNT(*) as cnt FROM analyses GROUP BY risk_level'
        ).fetchall()
        stats: Dict[str, Any] = {
            'total': 0,
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0,
            'VERY HIGH': 0,
        }
        for r in rows:
            stats[r['risk_level']] = r['cnt']
            stats['total'] += r['cnt']
        issue_row = conn.execute('SELECT SUM(issue_count) as total_issues FROM analyses').fetchone()
        stats['total_issues'] = issue_row['total_issues'] or 0
        return stats
    finally:
        conn.close()
