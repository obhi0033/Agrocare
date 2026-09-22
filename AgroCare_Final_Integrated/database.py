
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "agrocare.db"

def get_connection():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with get_connection() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_time TEXT NOT NULL,
            image_name TEXT,
            crop TEXT,
            disease TEXT,
            confidence REAL,
            severity TEXT,
            status TEXT,
            sharpness REAL
        )
        """)
        con.commit()

def save_scan(image_name, crop, disease, confidence, severity, status, sharpness):
    with get_connection() as con:
        con.execute("""
        INSERT INTO scan_history
        (scan_time,image_name,crop,disease,confidence,severity,status,sharpness)
        VALUES (?,?,?,?,?,?,?,?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name, crop, disease, confidence, severity, status, sharpness
        ))
        con.commit()

def get_history(limit=100):
    with get_connection() as con:
        rows = con.execute("""
        SELECT id,scan_time,image_name,crop,disease,confidence,severity,status,sharpness
        FROM scan_history ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]

def get_summary():
    with get_connection() as con:
        total = con.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
        detected = con.execute("SELECT COUNT(*) FROM scan_history WHERE status='Detected'").fetchone()[0]
        unknown = con.execute("SELECT COUNT(*) FROM scan_history WHERE status='Unknown / Unsupported'").fetchone()[0]
        avg = con.execute("SELECT AVG(confidence) FROM scan_history WHERE confidence IS NOT NULL").fetchone()[0]
    return {"total": total, "detected": detected, "unknown": unknown, "avg_confidence": avg or 0.0}

def clear_history():
    with get_connection() as con:
        con.execute("DELETE FROM scan_history")
        con.commit()
