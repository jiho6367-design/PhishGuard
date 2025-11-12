from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

DB_PATH = Path("data/feedback.db")
DEFAULT_THRESHOLD = 0.30


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_feedback_store(
    db_path: Path = DB_PATH,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   email_id TEXT NOT NULL,
                   model_label TEXT NOT NULL CHECK(model_label IN ('phishing','normal')),
                   model_score REAL NOT NULL,
                   user_label TEXT NOT NULL CHECK(user_label IN ('phishing','normal')),
                   threshold_before REAL NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS threshold_state (
                   id INTEGER PRIMARY KEY CHECK(id = 1),
                   threshold REAL NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """INSERT INTO threshold_state (id, threshold, updated_at)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO NOTHING""",
            (default_threshold, _iso_now()),
        )


def record_user_feedback(
    *,
    email_id: str,
    model_label: Literal["phishing", "normal"],
    model_score: float,
    user_feedback: Literal["phishing", "normal"],
    db_path: Path = DB_PATH,
    learning_rate: float = 0.07,
) -> float:
    if not (0.0 <= model_score <= 1.0):
        raise ValueError("model_score must be between 0 and 1")

    init_feedback_store(db_path)
    user_feedback = user_feedback.lower()
    model_label = model_label.lower()
    ts = _iso_now()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        (current_threshold,) = conn.execute(
            "SELECT threshold FROM threshold_state WHERE id = 1"
        ).fetchone()

        conn.execute(
            """INSERT INTO feedback (email_id, model_label, model_score,
                                     user_label, threshold_before, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (email_id, model_label, model_score, user_feedback, current_threshold, ts),
        )

        target_score = 0.75 if user_feedback == "phishing" else 0.25
        delta = learning_rate * (target_score - model_score)
        new_threshold = float(min(0.95, max(0.05, current_threshold + delta)))

        conn.execute(
            "UPDATE threshold_state SET threshold = ?, updated_at = ? WHERE id = 1",
            (new_threshold, ts),
        )

    return new_threshold


def generate_report(db_path: Path = DB_PATH) -> Path:
    init_feedback_store(db_path)
    db_path = Path(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS c FROM feedback").fetchone()["c"]
        phishing = (
            conn.execute(
                "SELECT COUNT(*) AS c FROM feedback WHERE user_label = 'phishing'"
            ).fetchone()["c"]
            or 0
        )
        false_positive = (
            conn.execute(
                """SELECT COUNT(*) AS c
                   FROM feedback
                   WHERE model_label = 'phishing' AND user_label = 'normal'"""
            ).fetchone()["c"]
            or 0
        )
        threshold_row = conn.execute(
            "SELECT threshold FROM threshold_state WHERE id = 1"
        ).fetchone()
        current_threshold = threshold_row["threshold"] if threshold_row else DEFAULT_THRESHOLD

    phishing_pct = (phishing / total * 100) if total else 0
    false_positive_pct = (false_positive / total * 100) if total else 0

    report_dir = db_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = report_dir / f"report_{datetime.utcnow().strftime('%Y%m%d')}.pdf"

    c = canvas.Canvas(str(filename), pagesize=letter)
    width, height = letter
    y = height - 72
    line_height = 20

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "PhishGuard Feedback Report")
    y -= line_height * 2

    c.setFont("Helvetica", 12)
    c.drawString(72, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= line_height
    c.drawString(72, y, f"Total Entries: {total}")
    y -= line_height
    c.drawString(72, y, f"Phishing Decisions: {phishing} ({phishing_pct:.2f}%)")
    y -= line_height
    c.drawString(
        72,
        y,
        f"False Positives: {false_positive} ({false_positive_pct:.2f}%)",
    )
    y -= line_height
    c.drawString(72, y, f"Current Threshold: {current_threshold:.2f}")

    c.showPage()
    c.save()
    return filename
