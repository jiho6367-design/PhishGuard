"""SQLite history; old records and human review IDs survive migration."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


DATA_DIR = Path(os.environ.get("PHISH_DATA_DIR") or Path(__file__).resolve().parent / "data").expanduser().resolve()
HISTORY_DB = DATA_DIR / "phishguard.db"
_LEGACY_COLUMNS = ("id", "subject", "label", "confidence", "feedback", "latency_ms", "timestamp", "date", "gmail_id", "gmail_labels", "auth_results", "body")
_NEW_COLUMNS = {"analysis_key": "TEXT", "settings_version": "TEXT", "extra_json": "TEXT", "rule_score": "REAL", "base_prob": "REAL", "score_kind": "TEXT"}
_ALL_COLUMNS = (*_LEGACY_COLUMNS, *_NEW_COLUMNS)
_SELECT = "SELECT a.*, h.gt_label AS human_label, h.noted_at AS reviewed_at FROM email_analysis a LEFT JOIN human_labels h ON h.id=a.id"


@contextmanager
def _connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(Path(db_path).expanduser().resolve(), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path: str | Path = HISTORY_DB) -> None:
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""CREATE TABLE IF NOT EXISTS email_analysis (
            id TEXT PRIMARY KEY, subject TEXT, label TEXT, confidence REAL,
            feedback TEXT, latency_ms REAL, timestamp TEXT, date TEXT,
            gmail_id TEXT, gmail_labels TEXT, auth_results TEXT, body TEXT
        )""")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(email_analysis)")}
        missing = set(_LEGACY_COLUMNS) - columns
        if missing:
            raise ValueError(f"기존 기록 DB에 필수 열이 없습니다: {', '.join(sorted(missing))}")
        for name, type_name in _NEW_COLUMNS.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE email_analysis ADD COLUMN {name} {type_name}")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_email_analysis_gmail_id ON email_analysis(gmail_id)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_email_analysis_key ON email_analysis(analysis_key) WHERE analysis_key IS NOT NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_email_analysis_timestamp ON email_analysis(timestamp DESC)")
        conn.execute("""CREATE TABLE IF NOT EXISTS human_labels (
            id TEXT PRIMARY KEY,
            gt_label TEXT NOT NULL CHECK(gt_label IN ('phishing','normal')),
            noted_at TEXT
        )""")


def _decode(value: str | None, expected: type, default: Any, name: str, warnings: list[str]) -> Any:
    if not value:
        return default
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, expected):
            raise ValueError("잘못된 저장 형식")
        return parsed
    except (ValueError, TypeError, RecursionError):
        warnings.append(f"기록의 {name} 메타데이터를 읽을 수 없어 기본값을 사용했습니다.")
        return default


def _record(row: sqlite3.Row) -> dict[str, Any]:
    warnings: list[str] = []
    result = _decode(row["extra_json"], dict, {}, "추가", warnings)
    result.update({name: row[name] for name in _ALL_COLUMNS if name not in {"extra_json", "gmail_labels", "auth_results"}})
    result["gmail_labels"] = _decode(row["gmail_labels"], list, [], "Gmail 라벨", warnings)
    result["auth_results"] = _decode(row["auth_results"], dict, {}, "메일 인증", warnings)
    result["score_kind"] = row["score_kind"] or "legacy_score"
    result["human_label"] = row["human_label"]
    result["reviewed_at"] = row["reviewed_at"]
    result.setdefault("gpt_feedback", row["feedback"])
    if warnings:
        existing = result.get("warnings")
        result["warnings"] = (existing if isinstance(existing, list) else []) + warnings
    return result


def get_history_item(id_: str, db_path: str | Path = HISTORY_DB) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT + " WHERE a.id=?", (id_,)).fetchone()
        return _record(row) if row else None


def get_by_analysis_key(key: str, db_path: str | Path = HISTORY_DB) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT + " WHERE a.analysis_key=?", (key,)).fetchone()
        return _record(row) if row else None


def get_existing_gmail_ids(gmail_ids: list[str], db_path: str | Path = HISTORY_DB) -> set[str]:
    if not gmail_ids:
        return set()
    found: set[str] = set()
    with _connect(db_path) as conn:
        for start in range(0, len(gmail_ids), 500):
            chunk = gmail_ids[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            found.update(row[0] for row in conn.execute(f"SELECT gmail_id FROM email_analysis WHERE gmail_id IN ({placeholders})", chunk) if row[0])
    return found


def load_history_by_gmail_ids(gmail_ids: list[str], db_path: str | Path = HISTORY_DB) -> list[dict[str, Any]]:
    if not gmail_ids:
        return []
    found: dict[str, dict[str, Any]] = {}
    with _connect(db_path) as conn:
        for start in range(0, len(gmail_ids), 500):
            chunk = gmail_ids[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(_SELECT + f" WHERE a.gmail_id IN ({placeholders})", chunk):
                found[row["gmail_id"]] = {**_record(row), "from_cache": True}
    return [found[id_] for id_ in dict.fromkeys(gmail_ids) if id_ in found]


def save_analysis_result(result: dict[str, Any], db_path: str | Path = HISTORY_DB, *, force: bool = False) -> str:
    """Return the stable stored ID; propagate write failures to the caller.

    Only real gmail_id values occupy the Gmail unique index. Reanalysis keeps
    the old ID so human labels still reference the same message. Duplicate
    analysis keys reuse their first stored record, unless an explicit force
    reanalysis replaces the result while preserving its stored ID.
    """
    if not isinstance(result, dict) or not isinstance(result.get("id"), str) or not result["id"].strip():
        raise ValueError("분석 기록에 비어 있지 않은 문자열 ID가 필요합니다.")
    if not isinstance(force, bool):
        raise ValueError("force는 true 또는 false여야 합니다.")
    row = {name: result.get(name) for name in _ALL_COLUMNS if name != "extra_json"}
    row["gmail_id"] = result.get("gmail_id") or None
    row["analysis_key"] = result.get("analysis_key") or None
    row["feedback"] = result.get("feedback") or result.get("gpt_feedback")
    row["gmail_labels"] = json.dumps(result.get("gmail_labels") or [], ensure_ascii=False, allow_nan=False)
    row["auth_results"] = json.dumps(result.get("auth_results") or {}, ensure_ascii=False, allow_nan=False)
    excluded = set(_ALL_COLUMNS) | {"text", "snippet", "from_cache", "human_label", "reviewed_at"}
    row["extra_json"] = json.dumps({key: value for key, value in result.items() if key not in excluded}, ensure_ascii=False, allow_nan=False)
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        gmail_row = conn.execute("SELECT id FROM email_analysis WHERE gmail_id=?", (row["gmail_id"],)).fetchone() if row["gmail_id"] else None
        key_row = conn.execute("SELECT id FROM email_analysis WHERE analysis_key=?", (row["analysis_key"],)).fetchone() if row["analysis_key"] else None
        if gmail_row and key_row and gmail_row["id"] != key_row["id"]:
            raise ValueError("서로 다른 기록의 Gmail ID와 분석 키가 충돌했습니다. 원본 기록은 보존됩니다.")
        if key_row and not gmail_row and not force:
            return str(key_row["id"])
        stored_id = str(gmail_row["id"]) if gmail_row else str(key_row["id"]) if key_row else row["id"]
        row["id"] = stored_id
        columns = ", ".join(_ALL_COLUMNS)
        placeholders = ", ".join(f":{name}" for name in _ALL_COLUMNS)
        updates = ", ".join(f"{name}=excluded.{name}" for name in _ALL_COLUMNS if name != "id")
        conn.execute(f"INSERT INTO email_analysis ({columns}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}", row)
        return stored_id


def load_history(limit: int = 200, days: int | None = None, db_path: str | Path = HISTORY_DB) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    values: list[Any] = []
    where = ""
    if days is not None:
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError("days는 0 이상의 정수여야 합니다.")
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        where = " WHERE a.date >= ?"
        values.append(cutoff)
    values.append(limit)
    with _connect(db_path) as conn:
        return [_record(row) for row in conn.execute(_SELECT + where + " ORDER BY a.timestamp DESC, a.id DESC LIMIT ?", values)]


def save_review(id_: str, label: str, db_path: str | Path = HISTORY_DB) -> dict[str, Any]:
    if label not in {"phishing", "normal"}:
        raise ValueError("검토 라벨은 phishing 또는 normal이어야 합니다.")
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM email_analysis WHERE id=?", (id_,)).fetchone():
            raise KeyError(id_)
        conn.execute("INSERT INTO human_labels(id, gt_label, noted_at) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET gt_label=excluded.gt_label, noted_at=excluded.noted_at", (id_, label, datetime.now(timezone.utc).isoformat()))
    return get_history_item(id_, db_path)  # type: ignore[return-value]
