"""Synthetic MIME/Gmail fixtures: never contact a mailbox or use real credentials."""
import base64
from types import SimpleNamespace

import pytest
import mail_service
import server.gmail_client as gmail_client
from scoring import compute_rule_score


def _encode_text(value: str, charset: str = "utf-8") -> str:
    return base64.urlsafe_b64encode(value.encode(charset)).decode("ascii").rstrip("=")


def _part(mime: str, text: str, **extra):
    return {"mimeType": mime, "body": {"data": _encode_text(text)}, **extra}


def _message():
    return {
        "id": "synthetic-id",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "synthetic preview",
        "payload": {
            "headers": [
                {"name": "sUbJeCt", "value": "계정 확인"},
                {"name": "From", "value": "Demo <demo@example.invalid>"},
                {"name": "Date", "value": "Sat, 10 Oct 2026 10:00:00 +0000"},
            ],
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "multipart/alternative", "parts": [
                    _part("text/plain", "회의 자료를 확인하세요."),
                    _part("text/html", '<p>중복 HTML 본문</p><a href="https://account-check.xyz/login">로그인</a>'),
                ]},
                {"filename": "danger.docm", "mimeType": "application/msword",
                 "body": {"attachmentId": "synthetic-attachment", "size": 123}},
            ],
        },
    }


class DummyRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class DummyMessages:
    def __init__(self, list_payload, detail_payload):
        self.list_payload = list_payload
        self.detail_payload = detail_payload
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return DummyRequest(self.list_payload)

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        payload = self.detail_payload
        if isinstance(payload, dict) and kwargs["id"] in payload:
            payload = payload[kwargs["id"]]
        return DummyRequest(payload)


class DummyService:
    def __init__(self, list_payload, detail_payload):
        self._messages = DummyMessages(list_payload, detail_payload)

    def users(self):
        return SimpleNamespace(messages=lambda: self._messages)


def _mock_mailbox(monkeypatch, service):
    monkeypatch.setattr(mail_service, "_get_credentials", lambda: object())
    monkeypatch.setattr(mail_service, "build", lambda *args, **kwargs: service)
    monkeypatch.setattr(gmail_client, "get_gmail_service", lambda: service)


def test_get_unread_emails_returns_shared_structure_and_readonly_calls(monkeypatch):
    service = DummyService({"messages": [{"id": "synthetic-id"}]}, _message())
    _mock_mailbox(monkeypatch, service)
    email = mail_service.get_unread_emails(max_results=5)[0]
    assert email["subject"] == "계정 확인"
    assert email["sender"] == "Demo <demo@example.invalid>"
    assert email["body"] == "회의 자료를 확인하세요."
    assert email["body_format"] == "plain"
    assert email["links"] == ["https://account-check.xyz/login"]
    assert email["attachments"] == [{
        "filename": "danger.docm", "mimeType": "application/msword",
        "attachmentId": "synthetic-attachment", "size": 123,
    }]
    assert email["auth_results"]["spf_pass"] is None
    assert email["auth_results"]["spf_result"] == "unknown"
    assert service._messages.calls == [
        ("list", {"userId": "me", "labelIds": ["INBOX"], "q": "is:unread", "maxResults": 5}),
        ("get", {"userId": "me", "id": "synthetic-id", "format": "full"}),
    ]
    assert mail_service.SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]


@pytest.mark.parametrize("status,flag", [
    ("pass", True), ("fail", False), ("softfail", False),
    ("neutral", None), ("none", None), ("temperror", None), ("permerror", None),
])
def test_authentication_status_keeps_unknown_distinct(status, flag):
    result = mail_service._parse_authentication_results([
        {"name": "Authentication-Results", "value": f"mx.example.invalid; SPF={status}; dkim=pass"},
    ])
    assert result["spf_pass"] is flag
    assert result["spf_result"] == status
    assert result["dkim_pass"] is True
    assert result["dmarc_pass"] is None
    assert result["dmarc_result"] == "unknown"


def test_missing_or_conflicting_auth_headers_never_become_failures():
    missing = mail_service._parse_authentication_results([])
    assert all(missing[key] is None for key in ("spf_pass", "dkim_pass", "dmarc_pass"))
    assert compute_rule_score("회의 일정", {"auth_results": missing})[0] == 0
    conflict = mail_service._parse_authentication_results([
        {"name": "Authentication-Results", "value": "mx.first; spf=pass"},
        {"name": "authentication-results", "value": "mx.second; spf=fail"},
    ])
    assert conflict["spf_pass"] is None and conflict["spf_result"] == "mixed"


def test_real_auth_failure_produces_a_named_indicator():
    auth = mail_service._parse_authentication_results([
        {"name": "Authentication-Results", "value": "mx.example.invalid; spf=fail"},
    ])
    score, signals = compute_rule_score("회의 일정", {"auth_results": auth})
    assert score == pytest.approx(0.25)
    assert [signal.name for signal in signals] == ["auth_fail"]


def test_html_is_safe_text_and_keeps_unique_href_metadata():
    result = mail_service.parse_gmail_payload(_part("text/html", """
<head><title>hidden heading</title></head><style>hidden style</style>
<script>dangerousFunction()</script><p>회의 &amp; 일정</p>
<a href="https://bad.xyz/login">확인</a><a href="https://bad.xyz/login">다시 확인</a>
<a href="javascript:fake()">의심 링크</a>
"""))
    assert result["body_format"] == "html"
    assert "회의 & 일정" in result["body"]
    assert "dangerousFunction" not in result["body"]
    assert "hidden" not in result["body"]
    assert "<" not in result["body"]
    assert result["links"] == ["https://bad.xyz/login", "javascript:fake()"]
    score, signals = compute_rule_score(result["body"], result)
    assert score > 0 and {"suspicious_tld", "bad_scheme"} <= {signal.name for signal in signals}


def test_plain_alternative_preferred_even_when_html_comes_first():
    result = mail_service.parse_gmail_payload({"mimeType": "multipart/alternative", "parts": [
        _part("text/html", '<p>HTML duplication</p><a href="https://bad.xyz">link</a>'),
        _part("text/plain", "Original plain body"),
    ]})
    assert result["body"] == "Original plain body"
    assert result["links"] == ["https://bad.xyz"]


def test_html_is_used_if_plain_alternative_is_empty():
    result = mail_service.parse_gmail_payload({"mimeType": "multipart/alternative", "parts": [
        _part("text/plain", ""),
        _part("text/html", "<p>HTML body</p>"),
    ]})
    assert result["body"] == "HTML body" and result["body_format"] == "html"


def test_text_attachment_is_metadata_and_not_main_body():
    result = mail_service.parse_gmail_payload({"mimeType": "multipart/mixed", "parts": [
        _part("text/plain", "Main body"),
        _part("text/plain", "password verify urgent https://bad.xyz", filename="notes.txt"),
    ]})
    assert result["body"] == "Main body"
    assert result["attachments"][0]["filename"] == "notes.txt"
    assert result["attachments"][0]["attachmentId"] is None


def test_declared_charset_is_used():
    result = mail_service.parse_gmail_payload({
        "mimeType": "text/plain",
        "headers": [{"name": "Content-Type", "value": 'text/plain; charset="iso-8859-1"'}],
        "body": {"data": _encode_text("café", "iso-8859-1")},
    })
    assert result["body"] == "café" and result["parser_warnings"] == []


def test_bad_base64_is_reported_not_mistaken_for_empty_inbox():
    result = mail_service.parse_gmail_payload({"mimeType": "text/plain", "body": {"data": "%%%invalid"}})
    assert result["body"] == ""
    assert "base64" in result["parser_warnings"][0]


def test_unknown_charset_is_reported_and_falls_back_to_utf8():
    part = _part("text/plain", "한글 본문", headers=[
        {"name": "Content-Type", "value": "text/plain; charset=no-such-encoding"},
    ])
    result = mail_service.parse_gmail_payload(part)
    assert result["body"] == "한글 본문"
    assert result["parser_warnings"]


def test_legacy_walk_adapter_preserves_hidden_href_and_attachment():
    body, attachments = [], []
    mail_service._walk_parts(_message()["payload"], body, attachments)
    assert "https://account-check.xyz/login" in "\n".join(body)
    assert attachments[0]["filename"] == "danger.docm"


def test_server_detail_uses_same_mime_metadata_without_optional_sdk(monkeypatch):
    service = DummyService({}, _message())
    _mock_mailbox(monkeypatch, service)
    email = gmail_client.fetch_message_detail("synthetic-id")
    assert email["body"] == "회의 자료를 확인하세요."
    assert email["links"] == ["https://account-check.xyz/login"]
    assert email["attachments"][0]["filename"] == "danger.docm"
    assert email["auth_results"]["spf_result"] == "unknown"
    assert service._messages.calls[-1][1]["format"] == "full"


def test_server_metadata_list_does_not_fetch_full_bodies(monkeypatch):
    message = _message()
    message["internalDate"] = "1791626400000"
    service = DummyService({"messages": [{"id": "synthetic-id"}]}, message)
    _mock_mailbox(monkeypatch, service)
    result = gmail_client.fetch_recent_messages(limit=200, label="INBOX")
    assert result[0]["id"] == "synthetic-id"
    assert "body" not in result[0]
    assert service._messages.calls[0][1]["maxResults"] == 100
    assert service._messages.calls[1][1]["format"] == "metadata"


@pytest.mark.parametrize("message_id", ["", " ", None])
def test_server_empty_message_id_rejected_before_connection(monkeypatch, message_id):
    monkeypatch.setattr(gmail_client, "get_gmail_service", lambda: pytest.fail("mailbox contacted"))
    with pytest.raises(ValueError):
        gmail_client.fetch_message_detail(message_id)


def test_gmail_connection_errors_are_sanitized(monkeypatch):
    sensitive = "synthetic-private-token-must-not-leak"
    service = DummyService(RuntimeError(sensitive), _message())
    _mock_mailbox(monkeypatch, service)
    with pytest.raises(RuntimeError) as exc:
        mail_service.get_unread_emails()
    assert sensitive not in str(exc.value)
    with pytest.raises(gmail_client.GmailClientError) as exc:
        gmail_client.fetch_recent_messages()
    assert sensitive not in str(exc.value)


def test_partial_gmail_failure_survives_with_warning(monkeypatch):
    service = DummyService({"messages": [{"id": "ok"}, {"id": "bad"}]},
                           {"ok": _message(), "bad": RuntimeError("synthetic-failure")})
    _mock_mailbox(monkeypatch, service)
    result = mail_service.get_unread_emails()
    assert len(result) == 1 and result[0]["parser_warnings"]


def test_all_gmail_detail_failures_raise_visible_error(monkeypatch):
    service = DummyService({"messages": [{"id": "bad"}]}, RuntimeError("synthetic-failure"))
    _mock_mailbox(monkeypatch, service)
    with pytest.raises(RuntimeError, match="가져오지"):
        mail_service.get_unread_emails()


def test_parser_to_real_analysis_service_preserves_metadata_and_history(tmp_path, monkeypatch):
    from analysis_service import AnalysisService
    from history_store import init_db, get_by_analysis_key
    from settings_store import SettingsStore
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    db = tmp_path / "history.db"
    init_db(db)
    service = AnalysisService(db, SettingsStore(tmp_path / "settings.json"))
    message = _message()
    # Optional attachment values are absent on this synthetic inline payload.
    message["payload"]["parts"][1]["body"] = {}
    record = mail_service._message_record(message)
    item = {**record, "gmail_id": record["id"]}
    first = service.analyze(item)
    assert first["label"] == "phishing"
    assert first["auth_results"]["spf_pass"] is None
    assert first["auth_results"]["spf_result"] == "unknown"
    assert first["attachments"][0]["filename"] == "danger.docm"
    assert "attachmentId" not in first["attachments"][0]
    assert first["links"] == record["links"]
    assert first["body_format"] == "plain"
    assert "auth_fail" not in {signal["name"] for signal in first["signals"]}
    saved = get_by_analysis_key(first["analysis_key"], db_path=db)
    assert saved["links"] == record["links"] and saved["attachments"] == first["attachments"]
    second = service.analyze(item)
    assert second["from_cache"] is True and second["id"] == first["id"]
