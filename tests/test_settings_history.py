"""Disk-level regression tests for migration, recovery, and stable identity."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import history_store as history
import settings_store
from settings_store import SettingsConflict, SettingsStore


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "history.db"
    history.init_db(path)
    return path


def _result(id_="manual-1", **overrides):
    return {
        "id": id_, "subject": "합성 메일", "label": "normal", "confidence": 0.1,
        "risk_score": 0.1, "score_kind": "rule_risk_index", "feedback": "확인하세요",
        "timestamp": "2026-10-11T10:00:00+00:00", "date": "2026-10-11",
        "body": "본문", "rule_signals": ["example_signal"], "signals": [{"name": "example", "score": 0.1}],
        "auth_results": {"spf": None, "dkim": True}, **overrides,
    }


def _legacy_db(path, *, duplicate_gmail=False):
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE email_analysis (
          id TEXT PRIMARY KEY, subject TEXT, label TEXT, confidence REAL,
          feedback TEXT, latency_ms REAL, timestamp TEXT, date TEXT,
          gmail_id TEXT, gmail_labels TEXT, auth_results TEXT, body TEXT)""")
        conn.execute("INSERT INTO email_analysis VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
            "old-id", "이전 제목", "phishing", 0.8, "이전 근거", 500,
            "2025-01-01T00:00:00", "2025-01-01", "gmail-old", '["INBOX"]', '{"spf":false}', "이전 본문"))
        if duplicate_gmail:
            conn.execute("INSERT INTO email_analysis(id,gmail_id) VALUES('duplicate','gmail-old')")
        conn.execute("CREATE TABLE human_labels(id TEXT PRIMARY KEY,gt_label TEXT NOT NULL CHECK(gt_label IN ('phishing','normal')),noted_at TEXT)")
        conn.execute("INSERT INTO human_labels VALUES('old-id','normal','2025-01-02')")


def test_settings_missing_load_has_no_file_side_effect_and_defaults_are_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("PHISH_THRESHOLD", "0.4")
    path = tmp_path / "new-dir" / "settings.json"
    store = SettingsStore(path)
    monkeypatch.setenv("PHISH_THRESHOLD", "0.8")
    settings = store.load()
    assert settings["threshold"] == 0.4
    assert settings["max_results"] == 10 and settings["retain_body"] is True
    assert len(settings["version"]) == 64 and settings["warnings"] == []
    assert not path.parent.exists()


@pytest.mark.parametrize("value", ["nan", "inf", "wrong", "0.01", "2"])
def test_settings_invalid_environment_uses_safe_default(tmp_path, monkeypatch, value):
    monkeypatch.setenv("PHISH_THRESHOLD", value)
    assert SettingsStore(tmp_path / "settings.json").load()["threshold"] == 0.2


@pytest.mark.parametrize("payload", [
    [], {"unexpected": 1}, {"version": None}, {"threshold": True}, {"threshold": "0.3"},
    {"threshold": float("nan")}, {"threshold": float("inf")}, {"threshold": 10**400},
    {"threshold": 0.049}, {"threshold": 0.951}, {"max_results": True}, {"max_results": 1.0},
    {"max_results": 0}, {"max_results": 51}, {"retain_body": "false"}, {"retain_body": 0},
])
def test_settings_invalid_update_never_changes_existing_bytes(tmp_path, payload):
    store = SettingsStore(tmp_path / "settings.json")
    store.update({"threshold": 0.35})
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.update(payload)
    assert store.path.read_bytes() == before


def test_settings_atomic_save_stale_version_and_new_process_other_cwd(tmp_path):
    path = tmp_path / "data" / "settings.json"
    store = SettingsStore(path)
    first = store.load()
    saved = store.update({"version": first["version"], "threshold": 0.65, "max_results": 37, "retain_body": False})
    assert saved["version"] != first["version"]
    with pytest.raises(SettingsConflict):
        store.update({"version": first["version"], "threshold": 0.3})
    assert json.loads(path.read_text("utf-8")) == {"threshold": 0.65, "max_results": 37, "retain_body": False}
    other = tmp_path / "other-cwd"
    other.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(settings_store.__file__).parent)}
    result = subprocess.run([sys.executable, "-c", "import json,sys; from settings_store import SettingsStore; print(json.dumps(SettingsStore(sys.argv[1]).load()))", str(path)], cwd=other, env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == saved
    assert SettingsStore(path).load() == saved
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("raw", [b"{broken", b"\xff\xfe\x00", b'null', b'{"threshold":0.3}', b'{"threshold":NaN,"max_results":10,"retain_body":true}'])
def test_settings_corruption_preserved_then_backed_up_on_explicit_save(tmp_path, raw):
    path = tmp_path / "settings.json"
    path.write_bytes(raw)
    store = SettingsStore(path)
    defaults = store.load()
    assert defaults["warnings"] and defaults["threshold"] == store._defaults["threshold"]
    assert path.read_bytes() == raw and not list(tmp_path.glob("*.bak"))
    saved = store.update({"version": defaults["version"], "max_results": 12})
    backups = list(tmp_path.glob("settings.json.corrupt-*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == raw
    assert saved["warnings"] and store.load()["max_results"] == 12


def test_settings_write_error_propagates_and_leaves_original(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "settings.json")
    store.update({"threshold": 0.4})
    before = store.path.read_bytes()
    def fail_replace(*_args):
        raise OSError("simulated disk replacement failure")
    monkeypatch.setattr(settings_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.update({"threshold": 0.7})
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_settings_corruption_changes_version_and_old_edit_conflicts(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    before = store.update({"threshold": 0.3})
    store.path.write_bytes(b"damaged")
    assert store.load()["version"] != before["version"]
    with pytest.raises(SettingsConflict):
        store.update({"version": before["version"], "threshold": 0.7})
    assert store.path.read_bytes() == b"damaged"


def test_settings_deeply_nested_damaged_json_does_not_crash_load(tmp_path):
    path = tmp_path / "settings.json"
    raw = b"[" * 2000 + b"0" + b"]" * 2000
    path.write_bytes(raw)
    assert SettingsStore(path).load()["warnings"]
    assert path.read_bytes() == raw


def test_settings_two_instances_same_version_allow_one_update_only(tmp_path):
    path = tmp_path / "settings.json"
    stores = [SettingsStore(path), SettingsStore(path)]
    version = stores[0].load()["version"]
    ready = threading.Barrier(2)
    def save(index):
        ready.wait(timeout=5)
        changes = {"threshold": 0.31} if index == 0 else {"max_results": 20}
        try:
            return "saved", stores[index].update({"version": version, **changes})
        except SettingsConflict:
            return "conflict", None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))
    assert sorted(status for status, _ in results) == ["conflict", "saved"]
    saved = next(settings for status, settings in results if status == "saved")
    assert stores[0].load() == saved


def test_settings_two_processes_same_version_allow_one_update_only(tmp_path):
    path = tmp_path / "settings.json"
    version = SettingsStore(path).load()["version"]
    gate = tmp_path / "start-gate"
    env = {**os.environ, "PYTHONPATH": str(Path(settings_store.__file__).parent)}
    code = '''import json,sys,time
from pathlib import Path
from settings_store import SettingsStore, SettingsConflict
deadline=time.monotonic()+5
while not Path(sys.argv[2]).exists():
    if time.monotonic()>deadline: raise RuntimeError("gate timeout")
    time.sleep(0.01)
try:
    saved=SettingsStore(sys.argv[1]).update(json.loads(sys.argv[3]))
    print(json.dumps({"status":"saved","settings":saved}))
except SettingsConflict:
    print(json.dumps({"status":"conflict"}))
'''
    processes = [subprocess.Popen([sys.executable, "-c", code, str(path), str(gate), json.dumps(payload)], env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for payload in [
        {"version": version, "threshold": 0.31}, {"version": version, "max_results": 20}]]
    try:
        gate.write_text("start", encoding="utf-8")
        outputs = [process.communicate(timeout=10) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        results = [json.loads(stdout) for stdout, _stderr in outputs]
        assert sorted(result["status"] for result in results) == ["conflict", "saved"]
        saved = next(result["settings"] for result in results if result["status"] == "saved")
        assert SettingsStore(path).load() == saved
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_settings_file_lock_has_bounded_timeout_and_preserves_original(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.update({"threshold": 0.3})
    before = path.read_bytes()
    store.LOCK_TIMEOUT_SECONDS = 0.15
    env = {**os.environ, "PYTHONPATH": str(Path(settings_store.__file__).parent)}
    code = '''import sys
from pathlib import Path
from settings_store import _file_lock
with _file_lock(Path(sys.argv[1]), 2):
    print("locked",flush=True)
    sys.stdin.readline()
'''
    process = subprocess.Popen([sys.executable, "-c", code, str(path.with_name(path.name + ".lock"))], env=env, cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(OSError, match="저장 중"):
            store.update({"threshold": 0.7})
        assert path.read_bytes() == before
        process.communicate(input="release\n", timeout=5)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_history_migration_preserves_old_row_and_human_review(tmp_path):
    path = tmp_path / "old.db"
    _legacy_db(path)
    history.init_db(path)
    history.init_db(path)
    old = history.get_history_item("old-id", path)
    assert old["id"] == "old-id" and old["body"] == "이전 본문"
    assert old["confidence"] == 0.8 and old["score_kind"] == "legacy_score"
    assert old["human_label"] == "normal" and old["reviewed_at"] == "2025-01-02"
    assert old["auth_results"] == {"spf": False} and old["gmail_labels"] == ["INBOX"]
    with sqlite3.connect(path) as conn:
        assert len(conn.execute("PRAGMA table_info(email_analysis)").fetchall()) == 18
        assert conn.execute("SELECT count(*) FROM email_analysis").fetchone()[0] == 1


def test_history_migration_rolls_back_if_unique_index_fails(tmp_path):
    path = tmp_path / "duplicates.db"
    _legacy_db(path, duplicate_gmail=True)
    with pytest.raises(sqlite3.IntegrityError):
        history.init_db(path)
    with sqlite3.connect(path) as conn:
        assert len(conn.execute("PRAGMA table_info(email_analysis)").fetchall()) == 12
        assert conn.execute("SELECT count(*) FROM email_analysis").fetchone()[0] == 2
        assert conn.execute("SELECT gt_label FROM human_labels WHERE id='old-id'").fetchone()[0] == "normal"


def test_history_manual_ids_not_misclassified_as_gmail_and_metadata_roundtrip(db):
    result = _result(analysis_key="key-1", settings_version="policy-1", attachments=["report.pdf"], threshold=0.2)
    assert history.save_analysis_result(result, db) == "manual-1"
    loaded = history.get_by_analysis_key("key-1", db)
    assert loaded["gmail_id"] is None and "from_cache" not in loaded
    assert loaded["signals"] == result["signals"] and loaded["attachments"] == ["report.pdf"]
    assert loaded["auth_results"]["spf"] is None and loaded["settings_version"] == "policy-1"
    assert history.get_existing_gmail_ids(["manual-1"], db) == set()


def test_history_duplicate_key_and_gmail_update_keep_review_identity(db):
    history.save_analysis_result(_result("first", analysis_key="same"), db)
    assert history.save_analysis_result(_result("second", analysis_key="same"), db) == "first"
    assert len(history.load_history(db_path=db)) == 1
    history.save_analysis_result(_result("gmail-first", gmail_id="g-1", analysis_key="g-1-policy-1"), db)
    history.save_review("gmail-first", "normal", db)
    assert history.save_analysis_result(_result("gmail-reanalysis", gmail_id="g-1", analysis_key="g-1-policy-2", label="phishing"), db) == "gmail-first"
    loaded = history.get_by_analysis_key("g-1-policy-2", db)
    assert loaded["id"] == "gmail-first" and loaded["human_label"] == "normal"
    assert loaded["label"] == "phishing" and history.get_by_analysis_key("g-1-policy-1", db) is None


def test_history_force_manual_reanalysis_replaces_cached_data_keeps_review(db):
    history.save_analysis_result(_result("original", analysis_key="manual-key", warnings=["old warning"]), db)
    history.save_review("original", "normal", db)
    refreshed = _result("new-analysis-id", analysis_key="manual-key", warnings=["fresh warning"], timestamp="2026-10-11T12:00:00+00:00", feedback="새 분석 근거")
    assert history.save_analysis_result(refreshed, db, force=True) == "original"
    cached = history.get_by_analysis_key("manual-key", db)
    assert cached["id"] == "original" and cached["human_label"] == "normal"
    assert cached["timestamp"] == refreshed["timestamp"] and cached["warnings"] == ["fresh warning"]
    assert cached["feedback"] == "새 분석 근거" and len(history.load_history(db_path=db)) == 1


def test_history_concurrent_duplicate_requests_store_one_record(db):
    def save(index):
        return history.save_analysis_result(_result(f"id-{index}", analysis_key="shared-key"), db)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(save, range(16)))
    assert len(set(ids)) == 1
    assert len(history.load_history(db_path=db)) == 1


def test_history_conflicting_gmail_and_key_does_not_merge_originals(db):
    history.save_analysis_result(_result("first", gmail_id="g-1", analysis_key="key-1"), db)
    history.save_analysis_result(_result("second", gmail_id="g-2", analysis_key="key-2"), db)
    with pytest.raises(ValueError, match="충돌"):
        history.save_analysis_result(_result("third", gmail_id="g-1", analysis_key="key-2"), db)
    assert history.get_history_item("first", db)["analysis_key"] == "key-1"
    assert history.get_history_item("second", db)["gmail_id"] == "g-2"


def test_history_body_privacy_no_duplicate_text_in_extra_json(db):
    history.save_analysis_result(_result(body="", text="SHOULD_NOT_RETAIN", snippet="SHOULD_NOT_RETAIN", retain_body=False), db)
    with sqlite3.connect(db) as conn:
        body, extra = conn.execute("SELECT body,extra_json FROM email_analysis").fetchone()
    assert body == "" and "SHOULD_NOT_RETAIN" not in extra
    assert "body" not in json.loads(extra) and "text" not in json.loads(extra)
    assert history.get_history_item("manual-1", db)["body"] == ""


def test_history_corrupted_metadata_recovers_with_warning_not_changed_score(db):
    history.save_analysis_result(_result(), db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE email_analysis SET extra_json='{broken',auth_results='[]',gmail_labels='null'")
    loaded = history.load_history(db_path=db)[0]
    assert loaded["confidence"] == 0.1 and loaded["auth_results"] == {} and loaded["gmail_labels"] == []
    assert len(loaded["warnings"]) == 3


def test_history_deeply_nested_metadata_is_reported_as_damage(db):
    history.save_analysis_result(_result(), db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE email_analysis SET extra_json=?", ("[" * 2000 + "0" + "]" * 2000,))
    loaded = history.get_history_item("manual-1", db)
    assert loaded["warnings"] and loaded["confidence"] == 0.1


def test_history_new_process_other_cwd_reads_same_persisted_record(db, tmp_path):
    history.save_analysis_result(_result("persisted", analysis_key="persisted-key"), db)
    history.save_review("persisted", "normal", db)
    cwd = tmp_path / "different-cwd"
    cwd.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(history.__file__).parent)}
    code = "import json,sys; from history_store import get_by_analysis_key; print(json.dumps(get_by_analysis_key('persisted-key',sys.argv[1])))"
    result = subprocess.run([sys.executable, "-c", code, str(db)], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    persisted = json.loads(result.stdout)
    assert persisted["id"] == "persisted" and persisted["human_label"] == "normal"
    assert persisted["body"] == "본문" and not (cwd / "data").exists()


def test_history_selected_order_review_validation_and_disk_failures(db):
    history.save_analysis_result(_result("a", gmail_id="g-a"), db)
    history.save_analysis_result(_result("b", gmail_id="g-b"), db)
    selected = history.load_history_by_gmail_ids(["g-b", "missing", "g-a", "g-b"], db)
    assert [item["id"] for item in selected] == ["b", "a"]
    assert all(item["from_cache"] for item in selected)
    assert history.save_review("a", "phishing", db)["human_label"] == "phishing"
    with pytest.raises(ValueError):
        history.save_review("a", "unknown", db)
    with pytest.raises(KeyError):
        history.save_review("absent", "normal", db)
    with pytest.raises(sqlite3.OperationalError):
        history.save_analysis_result(_result(), db.parent / "missing-dir" / "missing.db")


def test_history_import_has_no_cwd_data_creation_and_env_absolute_path(tmp_path):
    cwd = tmp_path / "other"
    cwd.mkdir()
    env = {**os.environ, "PHISH_DATA_DIR": str(tmp_path / "private-data"), "PYTHONPATH": str(Path(history.__file__).parent)}
    code = "import history_store; print(history_store.HISTORY_DB); assert not history_store.HISTORY_DB.parent.exists()"
    result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == tmp_path / "private-data" / "phishguard.db"
    assert not (cwd / "data").exists()
