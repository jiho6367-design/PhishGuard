"""Read-only Gmail service, sharing the dependency-free MIME parser."""
from __future__ import annotations
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from mail_service import (GmailAuthError, SCOPES, TOKEN_URI, _get_credentials, _extract_header,
                          _parse_authentication_results, _decode_data, _message_record,
                          parse_gmail_payload, build)

logger = logging.getLogger(__name__)

class GmailClientError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code

def _read_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise GmailAuthError(f"Gmail 설정이 없습니다: {name}")
    return value

def _build_credentials():
    # The authorized transport refreshes credentials when a read is executed.
    return _get_credentials()

def get_gmail_service():
    try:
        return build("gmail", "v1", credentials=_build_credentials(), cache_discovery=False)
    except GmailAuthError:
        raise
    except Exception as exc:
        raise GmailClientError("Gmail 읽기 서비스를 초기화할 수 없습니다.") from exc

def _status_code(exc: Exception) -> int:
    raw = getattr(exc, "status_code", None) or getattr(getattr(exc, "resp", None), "status", 502)
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return 502
    return number if 400 <= number <= 599 else 502

def _decode_body(data: Optional[str]) -> str:
    return _decode_data(data)

def _collect_text_parts(part: Dict[str, Any], accumulator: List[str]) -> None:
    parsed = parse_gmail_payload(part)
    if parsed["body"]:
        accumulator.append(parsed["body"])
    if parsed["links"]:
        accumulator.append("링크 주소: " + " ".join(parsed["links"]))

def fetch_recent_messages(limit: int = 20, label: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    service = get_gmail_service()
    kwargs: Dict[str, Any] = {"userId": "me", "maxResults": limit}
    if label:
        kwargs["labelIds"] = [label]
    try:
        response = service.users().messages().list(**kwargs).execute() or {}
    except Exception as exc:
        raise GmailClientError("Gmail 목록 조회에 실패했습니다.", _status_code(exc)) from exc
    messages = []
    failures = 0
    for item in response.get("messages", []) or []:
        message_id = item.get("id")
        if not message_id:
            continue
        try:
            message = service.users().messages().get(
                userId="me", id=message_id, format="metadata", metadataHeaders=["Subject", "From", "Date"]
            ).execute() or {}
        except Exception:
            failures += 1
            continue
        headers = (message.get("payload") or {}).get("headers", []) or []
        date_header = _extract_header(headers, "Date")
        iso_date = None
        internal_ms = message.get("internalDate")
        if internal_ms is not None:
            try:
                iso_date = datetime.fromtimestamp(int(internal_ms) / 1000, tz=timezone.utc).isoformat()
            except (ValueError, TypeError, OverflowError, OSError):
                pass
        messages.append({"id": message.get("id", message_id), "threadId": message.get("threadId"),
                         "internalDate": iso_date or date_header, "date": date_header,
                         "from": _extract_header(headers, "From"), "subject": _extract_header(headers, "Subject"),
                         "snippet": message.get("snippet", ""), "gmail_labels": message.get("labelIds", []) or []})
    if failures and not messages:
        raise GmailClientError("Gmail 목록의 상세 정보를 가져오지 못했습니다.")
    return messages

def fetch_message_detail(message_id: str) -> Dict[str, Any]:
    if not isinstance(message_id, str) or not message_id.strip():
        raise ValueError("조회할 Gmail 메일 ID가 필요합니다.")
    service = get_gmail_service()
    try:
        message = service.users().messages().get(userId="me", id=message_id, format="full").execute() or {}
    except Exception as exc:
        raise GmailClientError("Gmail 메일 내용을 가져오지 못했습니다.", _status_code(exc)) from exc
    record = _message_record(message, message_id)
    if not record["body"] and "body" in message and not message.get("payload"):
        record["body"] = _decode_body((message.get("body") or {}).get("data"))
    return record
