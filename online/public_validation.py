"""Small, strict manual-input contract in front of the approved validator."""
from __future__ import annotations

from typing import Any
from validation_snapshot import InputError, _text, prepare_item
from scoring import _all_urls

MAX_JSON_BYTES = 65_536
MAX_BODY_CHARS = 16_000
MAX_LINKS = 50
MAX_ATTACHMENTS = 20
MAX_URL_CHARS = 2048
ITEM_KEYS = {"title", "body", "sender", "auth_results", "attachments", "links"}
AUTH_KEYS = {"spf_pass", "dkim_pass", "dmarc_pass"}


def threshold_percent(value: Any) -> int:
    if type(value) is not int or not 5 <= value <= 95 or value % 5:
        raise InputError("임계값은 5~95 사이의 5 단위 정수여야 합니다.")
    return value


def prepare_public_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) - ITEM_KEYS:
        raise InputError("메일에는 제목·본문·발신자·인증 결과·첨부명·링크만 입력할 수 있습니다.")
    _text(item.get("body", ""), "본문", MAX_BODY_CHARS)
    auth = item.get("auth_results", {})
    if not isinstance(auth, dict) or set(auth) - AUTH_KEYS:
        raise InputError("인증 결과는 spf_pass·dkim_pass·dmarc_pass만 입력할 수 있습니다.")
    links = item.get("links", [])
    if not isinstance(links, list) or len(links) > MAX_LINKS:
        raise InputError("추가 링크는 최대 50개까지 입력할 수 있습니다.")
    attachments = item.get("attachments", [])
    if not isinstance(attachments, list) or len(attachments) > MAX_ATTACHMENTS:
        raise InputError("첨부명은 최대 20개까지 입력할 수 있습니다.")
    for attachment in attachments:
        if isinstance(attachment, dict) and set(attachment) != {"filename"}:
            raise InputError("첨부 항목에는 파일명만 입력할 수 있습니다.")
    prepared = prepare_item(item)
    text = "\n".join(part for part in (prepared["subject"], prepared["body"]) if part)
    # Match exactly the approved scorer's URL normalization/deduplication.
    # Reject rather than silently discard evidence from long pasted mail.
    all_urls = _all_urls(text, prepared)
    if len(all_urls) > MAX_LINKS:
        raise InputError("제목·본문·추가 링크의 중복 제외 URL은 합계 50개 이하여야 합니다.")
    # Match browser maxlength/String.length, including non-BMP characters.
    # This bounds derived host reasons under the UI's 2,400-unit safety cap.
    if any(len(url.encode("utf-16-le")) // 2 > MAX_URL_CHARS for url in all_urls):
        raise InputError("제목·본문·추가 링크의 각 URL은 2,048자 이하여야 합니다.")
    # These are never accepted from a caller; no Gmail/ID/HTML/parser path.
    return {key: prepared[key] for key in ("subject", "body", "sender", "auth_results", "attachments", "links")}
