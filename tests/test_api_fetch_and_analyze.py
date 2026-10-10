"""API regressions use the real offline analyzer and a fake Gmail source."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

import analysis_service
import api_service
from mail_service import GmailAuthError
from server.gmail_client import GmailClientError


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.delenv("PHISH_API_TOKEN", raising=False)
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    monkeypatch.setenv("PHISH_THRESHOLD", "0.20")
    app = api_service.create_app(tmp_path)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(local_app):
    return local_app.test_client()


def _mail(id_, body="Please verify https://example.net"):
    return {"id": id_, "subject": "검증용 메일", "body": body, "sender": "test@example.invalid", "attachments": [{"filename": "alert.docm", "mimeType": "application/msword"}], "auth_results": {"spf_pass": None, "dkim_pass": True}, "gmail_labels": ["INBOX"]}


def test_fetch_and_analyze_endpoint_uses_metadata_and_cache(client, monkeypatch):
    sample = [_mail("abc123")]
    monkeypatch.setattr(api_service, "get_unread_emails", lambda max_results: sample)
    original = analysis_service.analyze_emails
    calls = []
    def spy(texts, metas=None, threshold=None):
        calls.append((texts, metas, threshold))
        return original(texts, metas=metas, threshold=threshold)
    monkeypatch.setattr(analysis_service, "analyze_emails", spy)
    first = client.post("/api/fetch_and_analyze", json={"max_results": 1})
    second = client.post("/api/fetch_and_analyze", json={"max_results": 1})
    assert first.status_code == second.status_code == 200
    result = first.json["results"][0]
    assert result["label"] == "phishing"
    assert result["attachments"][0]["filename"] == "alert.docm"
    assert result["auth_results"]["spf_pass"] is None
    assert result["rule_signals"]
    assert len(calls) == 1  # Previously this route analyzed twice before caching.
    assert second.json["results"][0]["from_cache"] is True


def test_selected_preserves_order_partial_error_and_force(client, monkeypatch):
    def detail(id_):
        if id_ == "broken":
            raise GmailClientError("mock provider unavailable")
        return _mail(id_)
    monkeypatch.setattr(api_service, "fetch_message_detail", detail)
    client.post("/api/analyze_selected", json={"message_ids": ["cached"]})
    response = client.post("/api/analyze_selected", json={"message_ids": ["new", "broken", "cached"]})
    assert response.status_code == 200
    rows = response.json["results"]
    assert [row.get("gmail_id", row.get("id")) for row in rows] == ["new", "broken", "cached"]
    assert rows[2]["from_cache"] is True and rows[1]["error"] == "gmail_fetch_failed"
    forced = client.post("/api/analyze_selected", json={"message_ids": ["cached"], "force": True}).json["results"][0]
    assert forced["from_cache"] is False
    assert forced["id"] == rows[2]["id"]


@pytest.mark.parametrize("payload", [None, [], {}, {"body": 12}, {"title": []}, {"sender": None, "body": "hi"}, {"body": "hi", "auth_results": {"spf_pass": "false"}}, {"body": "hi", "attachments": [None]}, {"body": "hi", "attachments": [{"filename": "a", "size": True}]}, {"body": "hi", "links": [4]}, {"body": "hi", "body_format": {}}, {"body": "x" * 100_001}, {"body": "hi", "force": 1}])
def test_invalid_analyze_inputs_are_client_errors(client, payload):
    response = client.post("/api/analyze", json=payload) if payload is not None else client.post("/api/analyze", data="null", content_type="application/json")
    assert response.status_code == 400
    assert "detail" in response.json


def test_malformed_json_and_oversized_request(client):
    assert client.post("/api/analyze", data='{"body":', content_type="application/json").status_code == 400
    assert client.post("/api/analyze", data="x" * 1_048_577, content_type="application/json").status_code == 413


def test_batch_reports_per_item_error_in_original_positions(client):
    response = client.post("/api/analyze_batch", json={"items": ["invalid", {"body": "안녕하세요."}, {"body": 42}, {"body": "login https://example.com:not-a-port/login"}]})
    assert response.status_code == 200
    assert [row["index"] for row in response.json["results"]] == [0, 1, 2, 3]
    assert "error" in response.json["results"][0] and "error" in response.json["results"][2]
    assert "label" in response.json["results"][1] and "label" in response.json["results"][3]


def test_batch_invalid_unicode_does_not_drop_later_items(client):
    payload = {"items": [{"body": "first"}, {"body": "\ud800"}, {"body": "third"}]}
    response = client.post("/api/analyze_batch", data=json.dumps(payload, ensure_ascii=True), content_type="application/json")
    assert response.status_code == 200
    assert len(response.json["results"]) == 3
    assert response.json["results"][1]["error"] == "invalid_input"
    assert [row["body"] for row in client.get("/api/history").json["results"]] == ["third", "first"]


@pytest.mark.parametrize("path", ["/api/settings", "/api/history", "/api/list_emails", "/api/emails", "/api/emails/abc", "/metrics/summary"])
def test_every_sensitive_get_requires_same_token(tmp_path, monkeypatch, path):
    monkeypatch.setenv("PHISH_API_TOKEN", "검증-token")
    monkeypatch.setenv("OPENAI_API_KEY", "different-paid-key")
    app = api_service.create_app(tmp_path)
    client = app.test_client()
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-API-Key": "different-paid-key"}).status_code == 401
    assert client.get(path, headers={"X-API-Key": "잘못된 토큰"}).status_code == 401
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/settings", headers={"X-API-Key": "검증-token"}).status_code == 200


@pytest.mark.parametrize("method,path,payload", [("post", "/api/analyze", {"body": "hi"}), ("post", "/api/analyze_batch", {"items": []}), ("put", "/api/settings", {"threshold": .3}), ("post", "/api/fetch_and_analyze", {}), ("post", "/api/analyze_selected", {"message_ids": ["abc"]}), ("post", "/api/emails/abc/analyze", {}), ("post", "/api/history/abc/review", {"label": "normal"})])
def test_every_mutation_requires_token_before_processing(tmp_path, monkeypatch, method, path, payload):
    monkeypatch.setenv("PHISH_API_TOKEN", "test-only")
    client = api_service.create_app(tmp_path).test_client()
    assert getattr(client, method)(path, json=payload).status_code == 401


def test_cross_site_and_rebound_host_are_rejected(client):
    assert client.post("/api/analyze", json={"body": "hi"}, headers={"Origin": "https://example.invalid"}).status_code == 403
    assert client.get("/api/history", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/analyze", json={"body": "hi"}, headers={"Origin": "http://localhost"}).status_code == 200


def test_policy_change_invalidates_cache_and_review_survives(client, local_app):
    item = {"body": "Please verify https://example.net"}
    first = client.post("/api/analyze", json=item).json
    assert first["label"] == "phishing" and 0 < first["risk_score"] < .95
    assert client.post("/api/analyze", json=item).json["from_cache"] is True
    reviewed = client.post(f'/api/history/{first["id"]}/review', json={"label": "normal"})
    assert reviewed.json["human_label"] == "normal"
    settings = client.get("/api/settings").json
    assert client.put("/api/settings", json={"threshold": .95, "version": settings["version"]}).status_code == 200
    second = client.post("/api/analyze", json=item).json
    assert second["from_cache"] is False and second["label"] == "normal"
    assert second["settings_version"] != first["settings_version"]
    assert client.get(f'/api/history/{first["id"]}').json["human_label"] == "normal"
    metrics = client.get("/metrics/summary").json
    assert metrics["total"] == 2 and metrics["reviewed"] == metrics["disagreements"] == 1
    assert metrics["avg_latency_ms"] is not None


def test_body_retention_off_really_omits_text_in_sqlite(client, local_app):
    client.put("/api/settings", json={"retain_body": False})
    marker = "PRIVATE-BODY-REGRESSION-12345"
    first = client.post("/api/analyze", json={"title": "제목 보관", "sender": "x@example.invalid", "body": marker}).json
    cached = client.post("/api/analyze", json={"title": "제목 보관", "sender": "x@example.invalid", "body": marker}).json
    assert cached["from_cache"] is True and first["body"] == cached["body"] == ""
    with sqlite3.connect(local_app.extensions["history_path"]) as conn:
        row = conn.execute("SELECT body,extra_json FROM email_analysis WHERE id=?", (first["id"],)).fetchone()
        assert marker not in str(row)
    assert client.get("/api/history").json["results"][0]["subject"] == "제목 보관"


def test_write_failure_is_not_reported_success(client, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("mock disk failure")
    monkeypatch.setattr(analysis_service, "save_analysis_result", fail)
    response = client.post("/api/analyze", json={"body": "hello"})
    assert response.status_code == 503 and response.json["error"] == "storage_unavailable"
    assert client.get("/api/history").json["results"] == []


def test_concurrent_repeated_calls_compute_and_store_once(local_app, monkeypatch):
    original = analysis_service.analyze_emails
    calls = []
    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(analysis_service, "analyze_emails", spy)
    service = local_app.extensions["analysis_service"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: service.analyze({"body": "Repeat safely."}), range(8)))
    assert len(calls) == 1 and len({row["id"] for row in rows}) == 1
    assert sum(row["from_cache"] for row in rows) == 7


def test_manual_force_persists_fresh_result_and_preserves_review(client, monkeypatch):
    item = {"body": "ordinary note"}
    first = client.post("/api/analyze", json=item).json
    client.post(f'/api/history/{first["id"]}/review', json={"label": "normal"})
    original = analysis_service.analyze_emails
    def refreshed(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[0]["feedback"] = "새 검토 설명"
        rows[0]["warnings"] = ["새 분석 경고"]
        return rows
    monkeypatch.setattr(analysis_service, "analyze_emails", refreshed)
    second = client.post("/api/analyze", json={**item, "force": True}).json
    third = client.post("/api/analyze", json=item).json
    assert second["from_cache"] is False and third["from_cache"] is True
    assert first["id"] == second["id"] == third["id"]
    assert second["timestamp"] != first["timestamp"] and third["timestamp"] == second["timestamp"]
    assert third["feedback"] == "새 검토 설명" and third["warnings"] == ["새 분석 경고"]
    assert third["human_label"] == "normal"


def test_hf_weight_is_part_of_cache_policy(client, monkeypatch):
    pipeline = analysis_service.pipeline
    monkeypatch.setenv("PHISH_BACKEND", "hf")
    monkeypatch.setenv("PHISH_HF_MODEL", "synthetic-local-model")
    monkeypatch.setattr(pipeline, "classify_batch", lambda values: [{"phish_probability": .9, "base_label": "phishing", "base_confidence": .9, "model": "synthetic-local-model"} for _ in values])
    monkeypatch.setattr(pipeline, "MODEL_WEIGHT", .1)
    first = client.post("/api/analyze", json={"body": "ordinary note"}).json
    monkeypatch.setattr(pipeline, "MODEL_WEIGHT", .9)
    second = client.post("/api/analyze", json={"body": "ordinary note"}).json
    assert first["risk_score"] == pytest.approx(.09)
    assert second["risk_score"] == pytest.approx(.81) and second["from_cache"] is False


def test_parser_warning_changes_are_not_hidden_by_cache(client):
    item = {"body": "ordinary note", "parser_warnings": ["첫 파서 경고"]}
    first = client.post("/api/analyze", json=item).json
    second = client.post("/api/analyze", json={**item, "parser_warnings": ["다른 파서 경고"]}).json
    assert second["from_cache"] is False and second["id"] != first["id"]
    assert "다른 파서 경고" in second["warnings"]


def test_gmail_configuration_error_leaves_local_analyze_usable(client, monkeypatch):
    def missing(*args, **kwargs):
        raise GmailAuthError("missing test configuration")
    monkeypatch.setattr(api_service, "fetch_recent_messages", missing)
    response = client.get("/api/list_emails")
    assert response.status_code == 400 and response.json["error"] == "gmail_configuration"
    assert client.post("/api/analyze", json={"body": "ordinary note"}).status_code == 200


def test_saved_settings_survive_new_application_and_different_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("PHISH_API_TOKEN", raising=False)
    first = api_service.create_app(tmp_path / "stable")
    client = first.test_client()
    assert client.put("/api/settings", json={"threshold": .65, "max_results": 3, "retain_body": False}).status_code == 200
    record = client.post("/api/analyze", json={"body": "hello"}).json
    another = tmp_path / "elsewhere"
    another.mkdir()
    monkeypatch.chdir(another)
    second = api_service.create_app(tmp_path / "stable").test_client()
    assert second.get("/api/settings").json["threshold"] == .65
    assert second.get("/api/settings").json["max_results"] == 3
    assert second.get("/api/history").json["results"][0]["id"] == record["id"]
