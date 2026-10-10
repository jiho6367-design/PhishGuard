"""Legacy entry points must preserve local-only defaults and real service contracts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import dashboard
import ingestion_workflow
import phishing_analysis
import test_gmail as gmail_diagnostic


def test_legacy_imports_do_not_load_optional_sdks_or_connect():
    code = '''
import builtins
import socket
original = builtins.__import__
blocked = {"openai", "torch", "transformers", "streamlit", "requests", "google", "googleapiclient"}
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("Optional SDK imported: " + name)
    return original(name, *args, **kwargs)
def no_connect(*args, **kwargs):
    raise AssertionError("Network connection on import")
builtins.__import__ = guarded
socket.socket.connect = no_connect
import dashboard, demo_v2, phishing_analysis, ingestion_workflow, test_gmail
print("SAFE_IMPORTS")
'''
    env = {**os.environ, "OPENAI_API_KEY": "synthetic-must-not-be-used"}
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=15,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "SAFE_IMPORTS"


def test_paid_legacy_symbol_fails_explicitly_without_touching_client():
    class ForbiddenClient:
        def __getattribute__(self, name):
            raise AssertionError("Remote client used")
    with pytest.raises(RuntimeError, match="비활성화"):
        phishing_analysis.call_openai_analysis("input", client=ForbiddenClient())


def test_legacy_analysis_uses_original_links_and_rule_risk(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-must-not-be-used")
    class ForbiddenClient:
        def __getattribute__(self, name):
            raise AssertionError("Remote client used")
    result = phishing_analysis.analyze_email_content(
        "계정 확인", "demo@example.invalid",
        "로그인 확인 https://account-check.xyz/login https://example.invalid:broken/test", client=ForbiddenClient(),
    )
    assert result["backend"] == "rules"
    assert result["score_kind"] == "rule_risk_index"
    assert result["is_phishing"] is True
    assert 0 < result["risk_score"] <= 100
    assert result["reasons"]
    assert "확률" in result["summary"]


def test_gmail_diagnostic_default_only_reports_presence(monkeypatch, capsys):
    values = ["synthetic-client-id", "synthetic-client-secret", "synthetic-refresh-token"]
    for name, value in zip(gmail_diagnostic.REQUIRED, values):
        monkeypatch.setenv(name, value)
    assert gmail_diagnostic.main([]) == 0
    printed = capsys.readouterr().out
    assert "외부 연결은 실행하지 않았습니다" in printed
    assert all(value not in printed for value in values)


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8080", "http://example.com:8080",
    "http://user:secret@127.0.0.1:8080", "http://127.0.0.1:broken",
    "http://127.0.0.1:8080/path", "http://127.0.0.1:8080?redirect=1",
])
def test_streamlit_compat_never_sends_mail_to_external_or_ambiguous_url(monkeypatch, url):
    monkeypatch.setenv("PHISH_API_URL", url)
    with pytest.raises(ValueError):
        dashboard._local_base_url()


def test_streamlit_compat_one_request_and_useful_error(monkeypatch):
    import requests
    monkeypatch.setenv("PHISH_API_URL", "http://127.0.0.1:8080")
    calls = []
    class Response:
        status_code = 400
        def json(self):
            return {"error": "invalid_input", "detail": "제목을 확인해 주세요."}
    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()
    monkeypatch.setattr(requests, "request", request)
    with pytest.raises(RuntimeError, match="제목을 확인"):
        dashboard.request_api("POST", "/api/analyze", payload={"subject": 1})
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == (3, 10)
    assert calls[0][1]["allow_redirects"] is False


def test_ingestion_uses_saved_settings_cache_and_all_parser_metadata(tmp_path, monkeypatch):
    import api_service
    import mail_service
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    app = api_service.create_app(tmp_path)
    app.extensions["settings_store"].update({"threshold": 0.95, "retain_body": False})
    email = {
        "id": "gmail-synthetic-id", "subject": "검증", "body": "계정 확인",
        "sender": "demo@example.invalid", "gmail_labels": ["INBOX"],
        "auth_results": {"spf_pass": None, "dkim_pass": True, "spf_result": "unknown"},
        "attachments": [{"filename": "note.txt", "attachmentId": None, "size": None}],
        "links": ["https://account-check.xyz/login"],
        "parser_warnings": ["합성 파서 경고"], "body_format": "html",
    }
    calls = []
    def fetch(max_results):
        calls.append(max_results)
        return [email]
    monkeypatch.setattr(mail_service, "get_unread_emails", fetch)
    monkeypatch.setattr(api_service, "create_app", lambda data_dir=None: app)
    first = ingestion_workflow.fetch_and_analyze_unread(3, str(tmp_path))[0]
    second = ingestion_workflow.fetch_and_analyze_unread(3, str(tmp_path))[0]
    assert calls == [3, 3]
    assert first["threshold"] == 0.95 and first["label"] == "normal"
    assert first["body"] == ""
    assert first["links"] == email["links"]
    assert first["auth_results"]["spf_pass"] is None
    assert first["attachments"][0]["filename"] == "note.txt"
    assert "합성 파서 경고" in first["warnings"]
    assert first["from_cache"] is False and second["from_cache"] is True
    assert first["id"] == second["id"]


def test_ingestion_does_not_hide_service_failure(monkeypatch):
    import api_service
    import mail_service
    class FailingService:
        def analyze(self, item):
            raise OSError("synthetic disk failure")
    class App:
        extensions = {"analysis_service": FailingService()}
    monkeypatch.setattr(api_service, "create_app", lambda data_dir=None: App())
    monkeypatch.setattr(mail_service, "get_unread_emails", lambda max_results: [{"body": "test"}])
    with pytest.raises(OSError, match="disk failure"):
        ingestion_workflow.fetch_and_analyze_unread()


@pytest.mark.parametrize("value", [0, 51, True, 1.5, "5"])
def test_ingestion_rejects_invalid_limits_before_mail_fetch(value):
    with pytest.raises(ValueError):
        ingestion_workflow.fetch_and_analyze_unread(value)
