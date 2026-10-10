"""Validated, offline analysis with policy-aware persistence and deduplication."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from history_store import get_by_analysis_key, save_analysis_result
from optimized_pipeline import RULES_VERSION, analyze_emails
import optimized_pipeline as pipeline


class InputError(ValueError):
    """An input the caller can correct without retrying the same request."""


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise InputError(f"{name}은 문자열이어야 합니다.")
    if len(value) > limit:
        raise InputError(f"{name}은 {limit:,}자 이하여야 합니다.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputError(f"{name}에 올바르지 않은 유니코드 문자가 있습니다.") from exc
    return value.strip()


def prepare_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise InputError("메일 항목은 JSON 객체여야 합니다.")
    title = _text(item.get("title", item.get("subject", "")), "제목", 300)
    body = _text(item.get("body", ""), "본문", 100_000)
    sender = _text(item.get("sender", item.get("from", "")), "발신자", 500)
    if not title and not body:
        raise InputError("제목 또는 본문을 입력해 주세요.")
    auth = item.get("auth_results", {})
    if not isinstance(auth, dict):
        raise InputError("auth_results는 JSON 객체여야 합니다.")
    if any(key not in {f"{name}_{kind}" for name in ("spf", "dkim", "dmarc") for kind in ("pass", "result")} for key in auth):
        raise InputError("지원하지 않는 인증 결과 항목입니다.")
    auth_results = {}
    for key in ("spf_pass", "dkim_pass", "dmarc_pass"):
        value = auth.get(key)
        if value is not None and type(value) is not bool:
            raise InputError("인증 결과는 true, false 또는 null이어야 합니다.")
        auth_results[key] = value
    for name in ("spf", "dkim", "dmarc"):
        key = f"{name}_result"
        if key in auth and auth[key] is not None:
            auth_results[key] = _text(auth[key], "인증 결과 설명", 100)
    attachments = item.get("attachments", [])
    if not isinstance(attachments, list) or len(attachments) > 50:
        raise InputError("첨부 목록은 최대 50개의 배열이어야 합니다.")
    parsed_attachments = []
    for attachment in attachments:
        if isinstance(attachment, str):
            parsed_attachments.append({"filename": _text(attachment, "첨부 파일명", 260)})
        elif isinstance(attachment, dict):
            parsed = {"filename": _text(attachment.get("filename", ""), "첨부 파일명", 260)}
            for key in ("mimeType", "attachmentId"):
                if key in attachment and attachment[key] is not None:
                    parsed[key] = _text(attachment[key], "첨부 정보", 300)
            if "size" in attachment and attachment["size"] is not None:
                size = attachment["size"]
                if type(size) is not int or size < 0:
                    raise InputError("첨부 크기는 0 이상의 정수여야 합니다.")
                parsed["size"] = size
            parsed_attachments.append(parsed)
        else:
            raise InputError("첨부 항목은 파일명 또는 JSON 객체여야 합니다.")
    links = item.get("links", [])
    if not isinstance(links, list) or len(links) > 200:
        raise InputError("링크 목록은 최대 200개의 배열이어야 합니다.")
    parsed_links = sorted(set(_text(link, "링크", 2048) for link in links if link != ""))
    labels = item.get("gmail_labels", [])
    if not isinstance(labels, list) or len(labels) > 100:
        raise InputError("메일 라벨 목록 형식이 올바르지 않습니다.")
    parsed_labels = sorted(set(_text(label, "메일 라벨", 100) for label in labels))
    gmail_id = item.get("gmail_id")
    if gmail_id is not None:
        gmail_id = _text(gmail_id, "메일 ID", 128)
        if not gmail_id:
            gmail_id = None
    warnings = item.get("parser_warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    body_format = item.get("body_format", "plain")
    if not isinstance(body_format, str):
        raise InputError("본문 형식은 문자열이어야 합니다.")
    return {
        "subject": title, "body": body, "sender": sender,
        "auth_results": auth_results, "attachments": parsed_attachments,
        "links": parsed_links, "gmail_labels": parsed_labels, "gmail_id": gmail_id,
        "parser_warnings": [_text(warning, "파서 경고", 500) for warning in warnings[:50]],
        "body_format": body_format if body_format in {"plain", "html", "none", "empty"} else "plain",
    }


class AnalysisService:
    def __init__(self, db_path: Path, settings_store):
        self.db_path = Path(db_path).resolve()
        self.settings_store = settings_store
        # One runtime worker serializes check/compute/save to make repeated calls idempotent.
        self._lock = threading.RLock()

    def analyze(self, item: Any, force: bool = False) -> dict[str, Any]:
        if type(force) is not bool:
            raise InputError("force는 true 또는 false여야 합니다.")
        started = time.perf_counter()
        prepared = prepare_item(item)
        settings = self.settings_store.load()
        policy = {
            **pipeline.get_analysis_policy(),
            "threshold": settings["threshold"],
            "retain_body": settings["retain_body"],
        }
        fingerprint = prepared
        canonical = json.dumps({"input": fingerprint, "policy": policy}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        analysis_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            cached = get_by_analysis_key(analysis_key, db_path=self.db_path)
            if cached is not None and not force:
                cached.update(from_cache=True, settings_version=settings["version"], request_ms=round((time.perf_counter() - started) * 1000, 3))
                return cached
            text = "\n".join(part for part in (prepared["subject"], prepared["body"]) if part)
            try:
                output = analyze_emails([text], metas=[prepared], threshold=settings["threshold"])
            except ValueError as exc:
                raise RuntimeError("분석 백엔드 설정을 확인해 주세요.") from exc
            if len(output) != 1 or output[0].get("label") not in {"phishing", "normal"}:
                raise RuntimeError("분석 결과 형식이 올바르지 않습니다.")
            now = datetime.now(timezone.utc)
            result = {
                **prepared, **output[0],
                "id": cached["id"] if cached else uuid.uuid4().hex,
                "subject": prepared["subject"], "body": prepared["body"],
                "analysis_key": analysis_key, "settings_version": settings["version"],
                "rules_version": RULES_VERSION, "timestamp": now.isoformat(), "date": now.date().isoformat(),
                "from_cache": False, "needs_human_review": output[0]["label"] == "phishing",
                "feedback": output[0].get("feedback", ""), "gpt_feedback": output[0].get("feedback", ""),
            }
            # The source text must not leak into extra_json when body retention is disabled.
            result.pop("text", None)
            if not settings["retain_body"]:
                result["body"] = ""
            stored_id = save_analysis_result(result, db_path=self.db_path, force=force)
            if isinstance(stored_id, str):
                result["id"] = stored_id
            result["request_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return result
