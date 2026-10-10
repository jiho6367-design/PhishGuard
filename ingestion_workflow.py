"""Explicit Gmail ingestion using the same local analysis/settings/history service.

No Gmail calls happen at import. Running this CLI explicitly reads unread mail
using existing read-only OAuth credentials; it never issues credentials.
"""
from __future__ import annotations

import argparse
from typing import Any


def _auth_status(value: Any) -> str:
    return "pass" if value is True else "fail" if value is False else "unknown"


def _format_signals(email: dict[str, Any]) -> str:
    auth = email.get("auth_results") or {}
    labels = email.get("gmail_labels") or []
    attachments = email.get("attachments") or []
    filenames = [item if isinstance(item, str) else item.get("filename", "") for item in attachments]
    return (
        f"- Gmail labels: {', '.join(labels) or 'none'}\n"
        f"- Auth: SPF={_auth_status(auth.get('spf_pass'))}, "
        f"DKIM={_auth_status(auth.get('dkim_pass'))}, DMARC={_auth_status(auth.get('dmarc_pass'))}\n"
        f"- Attachments: {', '.join(filenames) or 'none'}"
    )


def _as_prompt_text(email: dict[str, Any]) -> str:
    """Compatibility text: metadata is passed separately, not scored as prose."""
    return f"{email.get('subject', '').strip()}\n{email.get('body', '').strip()}"


def fetch_and_analyze_unread(max_results: int = 5, data_dir: str | None = None) -> list[dict[str, Any]]:
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 50:
        raise ValueError("메일 조회 개수는 1~50 사이의 정수여야 합니다.")
    # Late imports avoid model/SDK/server initialization during discovery or pytest collection.
    from api_service import create_app
    from mail_service import get_unread_emails

    app = create_app(data_dir=data_dir)
    service = app.extensions["analysis_service"]
    emails = get_unread_emails(max_results=max_results)
    results: list[dict[str, Any]] = []
    for email in emails:
        auth = email.get("auth_results") or {}
        attachments = [
            {key: value for key, value in attachment.items() if value is not None}
            if isinstance(attachment, dict) else attachment
            for attachment in email.get("attachments") or []
        ]
        item = {
            "subject": email.get("subject", ""),
            "body": email.get("body", ""),
            "sender": email.get("sender", ""),
            "gmail_id": email.get("gmail_id") or email.get("id"),
            "gmail_labels": email.get("gmail_labels") or [],
            "auth_results": {
                key: auth.get(key)
                for key in ("spf_pass", "dkim_pass", "dmarc_pass", "spf_result", "dkim_result", "dmarc_result")
                if key in auth
            },
            "attachments": attachments,
            "links": email.get("links") or [],
            "parser_warnings": email.get("parser_warnings") or [],
            "body_format": email.get("body_format", "plain"),
        }
        # Validation/storage/backend failures propagate instead of claiming success.
        results.append(service.analyze(item))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="기존 Gmail 자격 증명으로 읽지 않은 메일을 로컬 분석합니다.")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    try:
        results = fetch_and_analyze_unread(args.max_results, args.data_dir)
    except Exception as exc:
        # Provider exception text may contain identifiers. Do not print credentials/bodies.
        print(f"메일 분석을 완료하지 못했습니다 ({type(exc).__name__}). 자격 증명·선택 의존성·저장 경로를 확인하세요.")
        return 1
    print(f"분석 완료: {len(results)}건. 상세 내용은 로컬 이력 화면에서 확인하세요.")
    for index, report in enumerate(results, 1):
        print(f"{index}. {report['label']} · 위험 지표 {float(report['risk_score']):.2f} · 캐시 {bool(report.get('from_cache'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
