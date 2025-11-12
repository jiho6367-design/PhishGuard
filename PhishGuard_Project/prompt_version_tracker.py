import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any
from dataclasses import dataclass

DB_PATH = Path("data/prompt_experiments.db")

def _connect(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(
            """CREATE TABLE IF NOT EXISTS prompts (
                   version TEXT PRIMARY KEY,
                   prompt TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
               );
               CREATE TABLE IF NOT EXISTS prompt_runs (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   version TEXT NOT NULL REFERENCES prompts(version) ON DELETE CASCADE,
                   email_id TEXT NOT NULL,
                   predicted TEXT NOT NULL,
                   actual TEXT NOT NULL,
                   correct INTEGER NOT NULL,
                   latency_ms REAL,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
               );"""
        )

@dataclass
class PromptRun:
    version: str
    email_id: str
    predicted: str
    actual: str
    latency_ms: float | None = None

class PromptVersionTracker:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    def register_prompt(self, version: str, prompt_text: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO prompts(version, prompt)
                   VALUES(?, ?)
                   ON CONFLICT(version) DO UPDATE SET prompt = excluded.prompt""",
                (version, prompt_text.strip()),
            )

    def log_run(self, run: PromptRun) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO prompt_runs(version, email_id, predicted, actual, correct, latency_ms)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (run.version, run.email_id, run.predicted, run.actual,
                 int(run.predicted == run.actual), run.latency_ms),
            )

    def build_report(self) -> Iterable[Dict[str, Any]]:
        query = """
            SELECT p.version,
                   COUNT(r.id) AS total,
                   COALESCE(SUM(r.correct), 0) AS correct,
                   ROUND(CAST(COALESCE(SUM(r.correct), 0) AS FLOAT) / NULLIF(COUNT(r.id), 0), 4) AS accuracy,
                   ROUND(AVG(r.latency_ms), 1) AS avg_latency_ms,
                   SUM(CASE WHEN r.predicted='phishing' AND r.actual='phishing' THEN 1 ELSE 0 END) AS true_positive,
                   SUM(CASE WHEN r.predicted='phishing' AND r.actual='normal' THEN 1 ELSE 0 END) AS false_positive
            FROM prompts p
            LEFT JOIN prompt_runs r ON r.version = p.version
            GROUP BY p.version
            ORDER BY accuracy DESC NULLS LAST;
        """
        with _connect(self.db_path) as conn:
            for row in conn.execute(query):
                yield dict(row)

    def print_report(self) -> None:
        print("\n=== Prompt Performance ===")
        for row in self.build_report():
            print(
                f"{row['version']:<12} "
                f"runs={row['total']:<4} acc={row['accuracy'] or 0:.2%} "
                f"tp={row['true_positive'] or 0} fp={row['false_positive'] or 0} "
                f"lat={row['avg_latency_ms'] or 0} ms"
            )

if __name__ == "__main__":
    tracker = PromptVersionTracker()
    tracker.print_report()
