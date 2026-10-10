"""Gmail read-only integration and a dependency-free MIME parser."""
from __future__ import annotations
import base64
import binascii
import os
import re
from email.message import Message
from html.parser import HTMLParser
from typing import Any, Dict, List

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"

class GmailAuthError(RuntimeError):
    """Gmail credentials or optional integration dependencies are unavailable."""

def build(*args, **kwargs):
    """Import Google transport only when the user explicitly requests Gmail."""
    try:
        from googleapiclient.discovery import build as google_build
        from google_auth_httplib2 import AuthorizedHttp
        from httplib2 import Http
    except ImportError as exc:
        raise GmailAuthError("Gmail 연동의 선택 의존성이 설치되지 않았습니다. 로컬 본문 분석은 계속 사용할 수 있습니다.") from exc
    credentials = kwargs.pop("credentials", None)
    if credentials is not None and "http" not in kwargs:
        kwargs["http"] = AuthorizedHttp(credentials, http=Http(timeout=15))
    return google_build(*args, **kwargs)

def _get_credentials():
    values = [os.getenv(name, "").strip() for name in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")]
    if not all(values):
        raise GmailAuthError("Gmail OAuth 설정이 없습니다. GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET/GOOGLE_REFRESH_TOKEN을 확인하세요.")
    try:
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise GmailAuthError("Gmail 선택 의존성 google-auth가 설치되지 않았습니다.") from exc
    return Credentials(token=None, refresh_token=values[2], token_uri=TOKEN_URI,
                       client_id=values[0], client_secret=values[1], scopes=SCOPES)

def _extract_header(headers: List[Dict[str, str]], name: str) -> str:
    return next((str(header.get("value", "")) for header in headers
                 if isinstance(header, dict) and str(header.get("name", "")).casefold() == name.casefold()), "")

def _parse_authentication_results(headers: List[Dict[str, str]]) -> Dict[str, Any]:
    """Header-derived hints only; missing or indeterminate is None, never failure."""
    auth_headers = " ".join(str(header.get("value", "")) for header in headers
                            if isinstance(header, dict) and str(header.get("name", "")).casefold() == "authentication-results")
    result: Dict[str, Any] = {}
    for protocol in ("spf", "dkim", "dmarc"):
        values = re.findall(rf"(?<![a-z0-9_-]){protocol}\s*=\s*(pass|fail|softfail|neutral|none|temperror|permerror)\b",
                            auth_headers, re.IGNORECASE)
        statuses = {value.casefold() for value in values}
        if statuses == {"pass"}:
            flag, status = True, "pass"
        elif statuses and statuses <= {"fail", "softfail"}:
            flag, status = False, "fail" if "fail" in statuses else "softfail"
        else:
            flag = None
            status = next(iter(statuses)) if len(statuses) == 1 else "mixed" if statuses else "unknown"
        result[protocol + "_pass"] = flag
        result[protocol + "_result"] = status
    return result

def _decode_data(data: Any, charset: str = "utf-8", warnings: List[str] | None = None) -> str:
    if not data:
        return ""
    warnings = warnings if warnings is not None else []
    try:
        encoded = re.sub(r"\s+", "", str(data))
        raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error):
        warnings.append("메일 본문의 일부 base64 데이터를 읽을 수 없습니다.")
        return ""
    try:
        return raw.decode(charset or "utf-8", errors="replace")
    except LookupError:
        warnings.append("알 수 없는 문자 인코딩이 있어 UTF-8로 읽었습니다.")
        return raw.decode("utf-8", errors="replace")

def _decode_body(part: Dict[str, Any]) -> str:
    content_type = _extract_header(part.get("headers", []) or [], "Content-Type")
    header = Message()
    header["Content-Type"] = content_type
    return _decode_data((part.get("body") or {}).get("data"), header.get_content_charset() or "utf-8")

class _HTMLText(HTMLParser):
    _hidden_tags = {"script", "style", "template", "head"}
    _block_tags = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "section", "article"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text: List[str] = []
        self.links: List[str] = []
        self.hidden: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()
        if tag in self._hidden_tags:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag in self._block_tags:
            self.text.append("\n")
        if tag == "a":
            href = next((value for name, value in attrs if name.casefold() == "href"), None)
            if href:
                self.links.append(str(href).strip())

    def handle_endtag(self, tag):
        tag = tag.casefold()
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag in self._block_tags:
            self.text.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)

def html_to_text(html: str) -> tuple[str, List[str]]:
    """Parse HTML as data only: never render markup, execute JS, or visit links."""
    parser = _HTMLText()
    parser.feed(html or "")
    parser.close()
    text = re.sub(r"[ \t]+", " ", "".join(parser.text))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text, list(dict.fromkeys(link for link in parser.links if link))

def parse_gmail_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer plain alternatives while retaining links from every body alternative."""
    attachments: List[Dict[str, Any]] = []
    links: List[str] = []
    warnings: List[str] = []

    def walk(part: Dict[str, Any]) -> tuple[str, str]:
        if not isinstance(part, dict):
            return "", "none"
        mime = str(part.get("mimeType", "")).casefold()
        headers = part.get("headers", []) or []
        disposition = _extract_header(headers, "Content-Disposition").casefold()
        body = part.get("body") or {}
        filename = str(part.get("filename") or "")
        is_attachment = bool(filename or body.get("attachmentId") or disposition.startswith("attachment"))
        if is_attachment:
            attachments.append({"filename": filename, "mimeType": mime,
                                "attachmentId": body.get("attachmentId"), "size": body.get("size")})
            # Even a text/plain attachment is not the message body.
            return "", "none"
        children = part.get("parts") or []
        if children:
            content = [walk(child) for child in children]
            if mime == "multipart/alternative":
                preferred = next(((text, kind) for text, kind in content if text and kind == "plain"), None)
                return preferred or next(((text, kind) for text, kind in content if text), ("", "none"))
            chosen = [(text, kind) for text, kind in content if text]
            return "\n".join(text for text, _ in chosen).strip(), ("plain" if chosen and all(kind == "plain" for _, kind in chosen) else "html" if chosen else "none")
        if mime not in {"text/plain", "text/html"}:
            return "", "none"
        charset_header = Message()
        charset_header["Content-Type"] = _extract_header(headers, "Content-Type")
        text = _decode_data(body.get("data"), charset_header.get_content_charset() or "utf-8", warnings)
        if mime == "text/html":
            text, hrefs = html_to_text(text)
            links.extend(hrefs)
            return text, "html"
        # Plain URL extraction is delegated to scoring, hrefs are separate metadata.
        return text.strip(), "plain"

    text, body_format = walk(payload or {})
    return {"body": text, "body_format": body_format, "attachments": attachments,
            "links": list(dict.fromkeys(links)), "parser_warnings": list(dict.fromkeys(warnings))}

def _walk_parts(part: Dict[str, Any], body_acc: List[str], attachments: List[Dict[str, Any]]) -> None:
    """Legacy adapter retaining HTML href information in plain output too."""
    parsed = parse_gmail_payload(part)
    if parsed["body"]:
        body_acc.append(parsed["body"])
    if parsed["links"]:
        body_acc.append("링크 주소: " + " ".join(parsed["links"]))
    attachments.extend(parsed["attachments"])

def _message_record(message: Dict[str, Any], fallback_id: str = "") -> Dict[str, Any]:
    payload = message.get("payload") or {}
    headers = payload.get("headers", []) or []
    sender = _extract_header(headers, "From")
    return {"id": message.get("id", fallback_id), "subject": _extract_header(headers, "Subject"),
            "sender": sender, "from_header": sender, "date": _extract_header(headers, "Date"),
            "snippet": message.get("snippet", ""), "gmail_labels": message.get("labelIds", []) or [],
            "auth_results": _parse_authentication_results(headers), **parse_gmail_payload(payload)}

def get_unread_emails(max_results: int = 10) -> List[Dict[str, Any]]:
    """Read-only Gmail fetch. Integration must be explicitly configured by the user."""
    limit = max(1, min(int(max_results), 100))
    credentials = _get_credentials()
    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        response = service.users().messages().list(userId="me", labelIds=["INBOX"], q="is:unread", maxResults=limit).execute() or {}
    except GmailAuthError:
        raise
    except Exception as exc:
        raise RuntimeError("Gmail 목록 조회에 실패했습니다. 연결과 읽기 권한을 확인하세요.") from exc
    emails = []
    failures = 0
    for item in response.get("messages", []) or []:
        message_id = item.get("id")
        if not message_id:
            continue
        try:
            message = service.users().messages().get(userId="me", id=message_id, format="full").execute() or {}
        except Exception:
            failures += 1
            continue
        emails.append(_message_record(message, str(message_id)))
    if failures and not emails:
        raise RuntimeError("Gmail 메일 내용을 가져오지 못했습니다. 잠시 후 다시 시도하세요.")
    if failures:
        for record in emails:
            record["parser_warnings"].append(f"선택된 메일 중 {failures}개의 내용을 가져오지 못했습니다.")
    return emails
