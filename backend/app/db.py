

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

DB_PATH = Path("data/search_logs.db")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist — v2 schema."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id   TEXT NOT NULL,
            query        TEXT NOT NULL,
            latency_ms   REAL,
            top_k        INTEGER,
            alpha        REAL,
            result_count INTEGER,
            error        TEXT,
            created_at   TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialised.")


def log_request(data: Dict[str, Any]):
    """Insert a request log entry."""
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO request_logs
              (request_id, query, latency_ms, top_k, alpha, result_count, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["request_id"],
            data["query"],
            data["latency_ms"],
            data["top_k"],
            data["alpha"],
            data["result_count"],
            data.get("error"),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()


def get_metrics() -> Dict[str, Any]:
    """Compute basic metrics from request logs."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT latency_ms, result_count FROM request_logs"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"total_requests": 0, "p50_ms": 0.0, "p95_ms": 0.0, "zero_results": 0}

    latencies    = sorted([r["latency_ms"] for r in rows])
    zero_results = sum(1 for r in rows if r["result_count"] == 0)
    n            = len(latencies)

    return {
        "total_requests": n,
        "p50_ms":         latencies[int(n * 0.50)],
        "p95_ms":         latencies[int(n * 0.95)],
        "zero_results":   zero_results,
    }


def get_all_logs():
    """Return all request logs as list of dicts."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM request_logs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()