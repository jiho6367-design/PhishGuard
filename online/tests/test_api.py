"""Synthetic public-mode regressions. No personal mail or remote services."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import socket
import sqlite3
import sys
import threading
from unittest.mock import patch

import pytest

ONLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONLINE))
import app as api

ORIGIN = "http://127.0.0.1:5000"
PAYLOAD = {"item": {"title": "합성 일정 안내", "body": "다음 주 회의 일정입니다."}, "threshold": 20}


@pytest.fixture
def application():
    return api.create_app([ORIGIN], vercel_runtime=False, limiter=api.InstanceLimiter(limit=1000))


@pytest.fixture
def client(application):
    return application.test_client()


def post(client, payload=PAYLOAD, *, origin=ORIGIN, base_url=ORIGIN, **kwargs):
    headers = {"Origin": origin} if origin is not None else {}
    headers.update(kwargs.pop("headers", {}))
    try:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        # Browser JSON.stringify escapes lone surrogates; Flask's test-client
        # ensure_ascii=False encoder otherwise fails before reaching our API.
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    return client.post("/api/analyze", base_url=base_url, headers=headers, data=raw,
                       content_type="application/json", **kwargs)


def test_actual_rules_contract_and_privacy(client):
    payload = {"item": {"title": "합성 제목 비공개값-A", "body": "본문 고유 비공개값-B", "sender": "person@example.com"}, "threshold": 20}
    response = post(client, payload)
    result = response.get_json()
    assert response.status_code == 200
    assert result["score_kind"] == "rule_risk_index" and result["backend"] == "rules"
    assert result["rules_version"] == "2026-10-10.rules.v2"
    assert result["risk_score"] == 0 and result["label"] == "normal" and result["threshold"] == 20
    assert result["persistent_storage"] is False and result["external_api_calls"] is False
    assert result["phish_probability"] is None
    assert set(result) == {"backend", "score_kind", "rules_version", "risk_score", "label", "threshold", "signals", "reasons", "warnings", "phish_probability", "persistent_storage", "external_api_calls", "feedback", "request_ms"}
    assert "비공개값" not in response.get_data(as_text=True)
    assert "person@example.com" not in response.get_data(as_text=True)
    assert "확률" in result["feedback"] and result["request_ms"] >= 0


def test_health_exposes_real_soft_limit(client):
    response = client.get("/api/health", base_url=ORIGIN)
    data = response.get_json()
    assert response.status_code == 200
    assert data["mode"] == "online_stateless" and data["gmail_enabled"] is False and data["hf_enabled"] is False
    assert data["persistent_storage"] is False and data["user_authentication"] is False
    assert data["limits"] == {"json_bytes": 65536, "body_chars": 16000, "links": 50, "url_chars": 2048, "url_length_unit": "utf16_code_units", "attachments": 20}
    assert data["rate_limit"] == {"scope": "process_instance", "analyses_per_minute": 30, "concurrency": 2, "durable": False, "global_quota": False, "resets_on_restart": True}


@pytest.mark.parametrize("threshold", [False, True, 20.0, "20", None, 0, 100, 21, [], {}])
def test_strict_threshold(client, threshold):
    response = post(client, {**PAYLOAD, "threshold": threshold})
    assert response.status_code == 400 and response.get_json()["error"] == "invalid_input"


@pytest.mark.parametrize("item", [None, [], "secret", {}, {"body": 1}, {"body": "x", "title": 1}, {"body": "x", "sender": []},
    {"body": "x", "body_format": "html"}, {"body": "x", "subject": "secret"}, {"body": "x", "gmail_id": "secret"},
    {"body": "x", "parser_warnings": ["secret"]}, {"body": "x", "auth_results": None},
    {"body": "x", "auth_results": {"spf_pass": "true"}}, {"body": "x", "auth_results": {"spf_pass": 1}},
    {"body": "x", "auth_results": {"spf": True}}, {"body": "x", "auth_results": {"spf_result": "secret"}},
    {"body": "x", "attachments": {}}, {"body": "x", "attachments": [{"filename": "a.zip", "attachmentId": "secret"}]},
    {"body": "x", "attachments": [{}]}, {"body": "x", "attachments": [1]}, {"body": "x", "links": "secret"},
    {"body": "x", "links": [None]}, {"body": "\ud800"}, {"body": "x" * 16001}, {"title": "x" * 301},
    {"body": "x", "sender": "x" * 501}, {"body": "x", "attachments": ["a"] * 21},
    {"body": "x", "attachments": ["a" * 261]}, {"body": "x", "links": ["a"] * 51},
    {"body": "x", "links": ["a" * 2049]}])
def test_strict_manual_input(client, item):
    response = post(client, {"item": item, "threshold": 20})
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_input"
    assert "secret" not in response.get_data(as_text=True)


@pytest.mark.parametrize("payload", [None, [], "secret", {}, {"item": {"body": "x"}, "force": True}, {"item": {"body": "x"}, "batch": []}])
def test_outer_contract(client, payload):
    assert post(client, payload).status_code == 400


@pytest.mark.parametrize("raw", [b'{"item":{"body":"a","body":"b"}}', b'{"item":{"body":"x"},"threshold":20,"threshold":25}',
    b'{"item":{"body":"x"},"threshold":NaN}', b'{"item":{"body":"x"},"threshold":Infinity}', b'{"item":{"body":"\xff"}}', b'not-json'])
def test_json_parser_rejects_ambiguous_encoding(client, raw):
    response = client.post("/api/analyze", data=raw, base_url=ORIGIN, headers={"Origin": ORIGIN}, content_type="application/json")
    assert response.status_code == 400


def test_json_size_and_content_encoding(client):
    response = client.post("/api/analyze", data=b" " * 65537, base_url=ORIGIN, headers={"Origin": ORIGIN}, content_type="application/json")
    assert response.status_code == 413 and "64 KiB" in response.get_json()["detail"]
    response = client.post("/api/analyze", data=json.dumps(PAYLOAD), base_url=ORIGIN, headers={"Origin": ORIGIN, "Content-Encoding": "gzip"}, content_type="application/json")
    assert response.status_code == 400
    response = client.post("/api/analyze", data=json.dumps(PAYLOAD), base_url=ORIGIN, headers={"Origin": ORIGIN}, content_type="text/plain")
    assert response.status_code == 400


def test_boundary_values(client):
    payload = {"item": {"title": "a" * 300, "body": "가" * 16000, "sender": "s" * 500,
                         "links": ["https://example.com/" + str(n) for n in range(50)], "attachments": ["a" * 260] * 20}, "threshold": 95}
    response = post(client, payload)
    assert response.status_code == 200


def test_total_extracted_urls_limit(client):
    urls = [f"https://example.com/item/{number}" for number in range(51)]
    assert post(client, {"item": {"body": "\n".join(urls[:50])}, "threshold": 20}).status_code == 200
    response = post(client, {"item": {"body": "\n".join(urls)}, "threshold": 20})
    assert response.status_code == 400 and "합계 50개" in response.get_json()["detail"]
    assert post(client, {"item": {"title": urls[0], "body": "\n".join(urls[1:50]), "links": [urls[50]]}, "threshold": 20}).status_code == 400
    assert post(client, {"item": {"title": urls[0], "body": "\n".join(urls[1:50]), "links": [urls[0]]}, "threshold": 20}).status_code == 200


def test_duplicate_url_normalization_matches_scoring(client):
    urls = [f"https://example.com/item/{number}" for number in range(49)]
    payload = {"item": {"body": "\n".join([*urls, "www.example.com/other."]), "links": ["https://www.example.com/other"]}, "threshold": 20}
    assert post(client, payload).status_code == 200


def test_implicit_url_length_limit(client):
    allowed_url = "https://example.com/" + "a" * (2048 - len("https://example.com/"))
    assert post(client, {"item": {"body": allowed_url}, "threshold": 20}).status_code == 200
    assert post(client, {"item": {"title": "https://" + "a" * 301}, "threshold": 20}).status_code == 400
    response = post(client, {"item": {"body": allowed_url + "a"}, "threshold": 20})
    assert response.status_code == 400 and "2,048자" in response.get_json()["detail"]


def test_many_signal_response_fits_ui_contract(client):
    payload = {"item": {"body": "urgent password login", "sender": "sample@example.top",
        "links": [f"https://xn--synthetic--host-{number}.xyz:8443/verify" for number in range(50)],
        "attachments": [f"synthetic-{number}.exe" for number in range(20)],
        "auth_results": {"spf_pass": False, "dkim_pass": False, "dmarc_pass": False}}, "threshold": 20}
    response = post(client, payload)
    data = response.get_json()
    assert response.status_code == 200 and data["risk_score"] == 1.0
    assert 200 < len(data["signals"]) <= 300
    assert len(data["signals"]) == len(data["reasons"])
    assert all(len(reason) <= 2400 for reason in data["reasons"])
    assert len(data["feedback"].encode("utf-16-le")) // 2 <= 10000
    assert "대표 근거 2개" in data["feedback"] and f"전체 {len(data['reasons'])}개" in data["feedback"]


def test_unicode_url_browser_length_and_reason_limit(client):
    prefix, suffix = "https://", ".xyz"
    url = prefix + "🦄" * ((2048 - len(prefix) - len(suffix)) // 2) + suffix
    assert len(url.encode("utf-16-le")) // 2 == 2048
    response = post(client, {"item": {"body": "login " + url}, "threshold": 20})
    data = response.get_json()
    assert response.status_code == 200
    assert all(len(reason.encode("utf-16-le")) // 2 <= 2400 for reason in data["reasons"])
    assert len(data["feedback"].encode("utf-16-le")) // 2 <= 10000
    longer_url = prefix + "🦄" * ((2048 - len(prefix) - len(suffix)) // 2 + 1) + suffix
    assert len(longer_url) < 2048
    assert post(client, {"item": {"body": longer_url}, "threshold": 20}).status_code == 400


def test_maximum_json_with_long_unicode_url_feedback_bounds(client):
    prefix, suffix = "https://", ".xyz"
    longest_url = prefix + "🦄" * ((2048 - len(prefix) - len(suffix)) // 2) + suffix
    payload = {"item": {"title": "a" * 300, "body": "login ", "sender": "s" * 500,
        "links": [longest_url, *["https://example.com/" + str(number) + "/" + "a" * 1000 for number in range(49)]],
        "attachments": [(f"synthetic-{number}-" + "a" * 260)[:256] + ".exe" for number in range(20)],
        "auth_results": {"spf_pass": False, "dkim_pass": False, "dmarc_pass": False}}, "threshold": 20}
    difference = 65536 - len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    assert 0 < difference <= 15994
    payload["item"]["body"] += " " * difference
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert len(raw) == 65536
    response = post(client, payload)
    data = response.get_json()
    assert response.status_code == 200
    assert len(data["feedback"].encode("utf-16-le")) // 2 <= 10000
    assert all(len(reason.encode("utf-16-le")) // 2 <= 2400 for reason in data["reasons"])


def test_query_input_rejected_without_reflection(client):
    response = client.post("/api/analyze?mail=secret-url", json=PAYLOAD, base_url=ORIGIN, headers={"Origin": ORIGIN})
    assert response.status_code == 400 and "secret-url" not in response.get_data(as_text=True)


@pytest.mark.parametrize("origin,host,extra", [
    (None, ORIGIN, {}), ("null", ORIGIN, {}), ("https://evil.example", ORIGIN, {}),
    (ORIGIN + "/", ORIGIN, {}), ("http://localhost:5000", ORIGIN, {}),
    (ORIGIN, "http://evil.example", {}), (ORIGIN, "http://127.0.0.1:5001", {}),
    (ORIGIN, ORIGIN, {"Sec-Fetch-Site": "cross-site"}),
    ("https://evil.example", "http://evil.example", {"X-Forwarded-Host": "127.0.0.1:5000"})])
def test_exact_origin_and_host(client, origin, host, extra):
    response = post(client, origin=origin, base_url=host, headers=extra)
    assert response.status_code == 403
    assert "Access-Control-Allow-Origin" not in response.headers


def test_other_allowed_host_is_not_cross_origin_permission():
    application = api.create_app([ORIGIN, "http://localhost:5000"], vercel_runtime=False)
    response = post(application.test_client(), origin="http://localhost:5000")
    assert response.status_code == 403


def test_forwarded_proto_ignored_off_vercel(client):
    assert post(client, headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "evil.example"}).status_code == 200
    application = api.create_app(["https://phish.example"], vercel_runtime=False)
    response = post(application.test_client(), origin="https://phish.example", base_url="http://phish.example", headers={"X-Forwarded-Proto": "https"})
    assert response.status_code == 403


def test_vercel_tls_terminating_proxy_contract():
    application = api.create_app(["https://phish.example"], vercel_runtime=True)
    client = application.test_client()
    args = {"origin": "https://phish.example", "base_url": "http://phish.example"}
    assert post(client, **args, headers={"X-Forwarded-Proto": "https"}).status_code == 200
    assert post(client, **args).status_code == 403
    assert post(client, **args, headers={"X-Forwarded-Proto": "http"}).status_code == 403
    assert post(client, **args, headers={"X-Forwarded-Proto": "https,http"}).status_code == 403


def test_system_domain_configuration(monkeypatch):
    monkeypatch.delenv("PHISH_PUBLIC_ORIGINS", raising=False)
    monkeypatch.setenv("VERCEL_URL", "unique-test.vercel.app")
    monkeypatch.setenv("VERCEL_BRANCH_URL", "branch-test.vercel.app")
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "phish.example")
    expected = ("https://unique-test.vercel.app", "https://branch-test.vercel.app", "https://phish.example")
    assert api.configured_origins(True) == expected
    assert api.configured_origins(False) == (ORIGIN, "http://localhost:5000")
    monkeypatch.setenv("PHISH_PUBLIC_ORIGINS", "https://extra.example, https://phish.example")
    assert set(api.configured_origins(True)) == {*expected, "https://extra.example"}


@pytest.mark.parametrize("value", ["http://public.example", "https://*.vercel.app", "https://Example.com", "https://example.com/", "https://example.com:443", "https://user@example.com", "https://example.com?x=1", "https://example.com#x", "https://example.com:99999", "https://example.com.", "https://-bad.example", "https://한글.example", " https://example.com"])
def test_configuration_fail_closed(value):
    with pytest.raises(ValueError):
        api.canonical_origin(value)


def test_unconfigured_vercel_fails_closed(monkeypatch):
    for name in ("PHISH_PUBLIC_ORIGINS", "VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL"):
        monkeypatch.delenv(name, raising=False)
    assert post(api.create_app(vercel_runtime=True).test_client(), origin="https://phish.example", base_url="https://phish.example").status_code == 403


@pytest.mark.parametrize("path", ["/api/history", "/api/settings", "/api/gmail", "/api/batch", "/api/auth", "/scoring.py", "/app.py", "/requirements.txt", "/data/results.csv", "/../scoring.py"])
def test_no_shared_or_sensitive_endpoints(client, path):
    assert client.get(path, base_url=ORIGIN).status_code == 404


def test_local_favicon_route_and_security_headers(client):
    response = client.get("/favicon.svg", base_url=ORIGIN)
    assert response.status_code == 200 and response.mimetype == "image/svg+xml"
    assert response.headers["Content-Security-Policy"] == api.CSP
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_response_no_store_and_cors(client):
    for response in [post(client), post(client, {"item": None}), client.get("/api/health", base_url=ORIGIN), client.get("/missing", base_url=ORIGIN)]:
        assert response.headers["Cache-Control"] == "no-store, max-age=0"
        assert response.headers["CDN-Cache-Control"] == "no-store"
        assert response.headers["Vercel-CDN-Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert response.headers["Content-Security-Policy"] == api.CSP
        assert "Access-Control-Allow-Origin" not in response.headers and "Set-Cookie" not in response.headers


def test_options_cross_site_denied(client):
    denied = client.options("/api/analyze", base_url=ORIGIN, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert denied.status_code == 403
    assert "Access-Control-Allow-Origin" not in denied.headers


def test_error_no_private_exception_or_application_log(client, monkeypatch, caplog):
    def bad(*args, **kwargs):
        raise RuntimeError("private-mail-secret URL https://secret.example and internal/path")
    monkeypatch.setattr(api, "compute_rule_score", bad)
    response = post(client)
    assert response.status_code == 500 and response.get_json()["error"] == "internal_error"
    assert "private-mail-secret" not in response.get_data(as_text=True) and not caplog.records


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://localhost:22", "http://169.254.169.254/latest/meta-data", "http://10.0.0.1", "http://[::1]:8080", "file:///etc/passwd", "javascript:alert(1)", "https://xn--test.example", "https://example.com:70000"])
def test_urls_are_parsed_without_network_or_dns(client, url):
    with patch.object(socket, "getaddrinfo", side_effect=AssertionError("Unexpected DNS")), \
         patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network")), \
         patch.object(sqlite3, "connect", side_effect=AssertionError("Unexpected DB")):
        response = post(client, {"item": {"body": "확인 https://example.com", "links": [url]}, "threshold": 20})
        assert response.status_code == 200


def test_auth_unknown_and_fail_differ(client):
    scores = []
    for value in (None, True, False):
        data = post(client, {"item": {"body": "합성 안내", "auth_results": {"spf_pass": value}}, "threshold": 20}).get_json()
        scores.append(data["risk_score"])
        assert ("auth_fail" in [signal["name"] for signal in data["signals"]]) == (value is False)
    assert scores == [0.0, 0.0, 0.25]


EXAMPLES = [
    ({"title": "회의 안내", "body": "다음 주 회의실에서 뵙겠습니다."}, 0.0),
    ({"body": "login https://portal.example.com"}, 0.35),
    ({"body": "Urgent password verify http://192.0.2.20:8080/login", "auth_results": {"spf_pass": False}, "attachments": ["invoice.zip"]}, 1.0),
    ({"body": "https://example.com:70000"}, 0.15),
    ({"body": "<script>globalThis.PHISHGUARD_XSS = true</script>", "auth_results": {"dkim_pass": False}}, 0.25),
]


@pytest.mark.parametrize("item,expected_score", EXAMPLES)
@pytest.mark.parametrize("threshold", range(5, 100, 5))
def test_actual_analysis_95_threshold_combinations(client, item, expected_score, threshold):
    data = post(client, {"item": item, "threshold": threshold}).get_json()
    assert data["risk_score"] == expected_score
    assert data["label"] == ("phishing" if expected_score * 100 + 1e-9 >= threshold else "normal")
    assert data["threshold"] == threshold


def test_changed_new_input_is_recomputed(client):
    first = post(client, {"item": {"body": "회의 일정"}, "threshold": 20}).get_json()
    second = post(client, {"item": {"body": "login https://portal.example.com"}, "threshold": 20}).get_json()
    assert first["risk_score"] == 0 and second["risk_score"] == 0.35
    assert second["signals"] and second["reasons"]


def test_approved_scoring_copy_hash():
    assert sha256((ONLINE / "scoring.py").read_bytes()).hexdigest() == "323e3907ac59a4af023cd66a0ac3feae61b2201c92d8c2ab78183bb88bc7d9cc"


def test_rate_window_limits_retries_and_recovers():
    now = [10.0]
    limiter = api.InstanceLimiter(clock=lambda: now[0])
    application = api.create_app([ORIGIN], vercel_runtime=False, limiter=limiter)
    client = application.test_client()
    assert [post(client).status_code for _ in range(30)] == [200] * 30
    response = post(client)
    assert response.status_code == 429 and response.headers["Retry-After"] == "60"
    assert response.get_json()["retry_after_seconds"] == 60
    assert len(limiter._starts) == 30 and limiter._active == 0
    now[0] = 69.5
    assert post(client).headers["Retry-After"] == "1"
    now[0] = 70.0
    assert post(client).status_code == 200


def test_invalid_requests_are_counted():
    application = api.create_app([ORIGIN], vercel_runtime=False)
    client = application.test_client()
    assert [post(client, {"item": None}).status_code for _ in range(30)] == [400] * 30
    assert post(client).status_code == 429


def test_fresh_instance_bypasses_soft_limit():
    first = api.create_app([ORIGIN], vercel_runtime=False).test_client()
    second = api.create_app([ORIGIN], vercel_runtime=False).test_client()
    assert [post(first).status_code for _ in range(30)] == [200] * 30
    assert post(first).status_code == 429
    assert post(second).status_code == 200
    # This limitation is deliberate evidence, not a durable/global quota claim.


def test_xff_does_not_bypass_one_instance_limit():
    application = api.create_app([ORIGIN], vercel_runtime=False)
    client = application.test_client()
    assert [post(client, headers={"X-Forwarded-For": f"192.0.2.{n}"}).status_code for n in range(30)] == [200] * 30
    assert post(client, headers={"X-Forwarded-For": "198.51.100.4"}).status_code == 429


def test_parallel_load_limit_and_finally_cleanup(monkeypatch):
    entered = threading.Barrier(3)
    release = threading.Event()
    real_score = api.compute_rule_score
    def slow(*args, **kwargs):
        entered.wait(timeout=5)
        assert release.wait(timeout=5)
        return real_score(*args, **kwargs)
    monkeypatch.setattr(api, "compute_rule_score", slow)
    application = api.create_app([ORIGIN], vercel_runtime=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lambda: post(application.test_client())) for _ in range(2)]
        entered.wait(timeout=5)
        response = post(application.test_client())
        assert response.status_code == 429 and response.headers["Retry-After"] == "1"
        release.set()
        assert [future.result(timeout=5).status_code for future in futures] == [200, 200]
    assert application.extensions["instance_limiter"]._active == 0


def test_per_request_threshold_independent_under_parallel(application):
    def analyze(threshold):
        return post(application.test_client(), {"item": {"body": "login https://portal.example.com"}, "threshold": threshold}).get_json()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(analyze, [20, 50] * 5))
    assert [data["label"] for data in results] == ["phishing", "normal"] * 5
    assert [data["threshold"] for data in results] == [20, 50] * 5


def test_hf_and_paid_env_never_selects_other_backend(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "hf")
    monkeypatch.setenv("PHISH_HF_MODEL", "no-such-model")
    data = post(api.create_app([ORIGIN], vercel_runtime=False).test_client()).get_json()
    assert data["backend"] == "rules" and data["external_api_calls"] is False


def test_no_app_files_written_during_analysis(client, monkeypatch):
    attempted = []
    def deny(*args, **kwargs):
        attempted.append("write")
        raise AssertionError("Unexpected file write")
    with patch.object(Path, "write_text", deny), patch.object(Path, "write_bytes", deny), patch.object(sqlite3, "connect", deny):
        assert post(client).status_code == 200
    assert attempted == []
