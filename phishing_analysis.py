"""Compatibility helpers for the former remote-analysis interface.

Analysis now uses the local pipeline. No external client is imported or called.
The 0-100 score is a rule risk index, not a calibrated probability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional

BODY_PREVIEW_CHARS = 1200
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
PROMPT_TEMPLATE = """[로컬 참고 미리보기]
제목: {subject}
발신자: {sender}
본문 ({max_chars}자 이내):
{body_preview}
이 문자열은 외부 모델로 전송하지 않습니다.
"""


def sanitize_body(body: str, max_chars: int = BODY_PREVIEW_CHARS) -> str:
    """Redact email addresses/URLs in a display preview only.

    This is not a complete PII detector. The analyzer receives the original text
    so that URL signals are retained; it processes that text locally.
    """
    if not isinstance(body, str):
        raise ValueError("본문은 문자열이어야 합니다.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("미리보기 길이는 0 이상의 정수여야 합니다.")
    redacted = URL_PATTERN.sub("[URL]", body)
    redacted = EMAIL_PATTERN.sub("[EMAIL]", redacted)
    return redacted.replace("\r", " ").strip()[:max_chars]


def build_prompt(subject: str, sender: str, body_preview: str) -> str:
    return PROMPT_TEMPLATE.format(
        subject=subject or "(제목 없음)",
        sender=sender or "(발신자 미상)",
        body_preview=body_preview or "(본문 없음)",
        max_chars=BODY_PREVIEW_CHARS,
    )


def call_openai_analysis(prompt: str, client: Any = None, model: Optional[str] = None) -> Dict[str, Any]:
    """Retain the old symbol while explicitly preventing a paid request."""
    raise RuntimeError("유료 외부 분석은 비활성화되어 있습니다. analyze_email_content로 로컬 분석을 사용하세요.")


def analyze_email_content(
    subject: str,
    sender: str,
    body: str,
    client: Any = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the old response shape using local analysis.

    ``client`` and ``model`` are accepted for call compatibility and never used.
    This helper does not save history; the primary API/ingestion service does.
    """
    if not all(isinstance(value, str) for value in (subject, sender, body)):
        raise ValueError("제목, 발신자, 본문은 문자열이어야 합니다.")
    if not (subject.strip() or body.strip()):
        raise ValueError("제목 또는 본문을 입력하세요.")
    from optimized_pipeline import analyze_emails

    result = analyze_emails(
        [f"{subject}\n{body}"],
        metas=[{"subject": subject, "sender": sender, "body": body}],
    )[0]
    return {
        "is_phishing": result["label"] == "phishing",
        "risk_score": round(float(result["risk_score"]) * 100),
        "reasons": result.get("rule_signals", []),
        "summary": result.get("feedback", ""),
        "score_kind": result.get("score_kind", "rule_risk_index"),
        "backend": result.get("backend", "rules"),
    }


@dataclass
class StoredAnalysis:
    message_id: str
    is_phishing: bool
    risk_score: int
    summary: str
    reasons: List[str]


class PhishingAnalysisStore:
    """Legacy in-memory store; records may contain user text in explanations."""

    def __init__(self):
        self._data: Dict[str, StoredAnalysis] = {}
        self._lock = Lock()

    def save(self, record: Dict[str, Any]) -> None:
        message_id = record.get("messageId")
        if not message_id:
            return
        stored = StoredAnalysis(
            message_id=str(message_id),
            is_phishing=bool(record.get("isPhishing")),
            risk_score=max(0, min(100, int(record.get("riskScore", 0)))),
            summary=str(record.get("summary", "")),
            reasons=list(record.get("reasons", [])),
        )
        with self._lock:
            self._data[str(message_id)] = stored

    def get(self, message_id: str) -> Optional[StoredAnalysis]:
        with self._lock:
            return self._data.get(message_id)

    def all(self) -> List[StoredAnalysis]:
        with self._lock:
            return list(self._data.values())
