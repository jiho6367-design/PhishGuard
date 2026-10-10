"""Deterministic Korean feedback, no model or external API import."""
from __future__ import annotations
from typing import Sequence


def make_feedback(label: str, risk: float, threshold: float, reasons: Sequence[str]) -> str:
    conclusion = "현재 규칙 기준으로 피싱이 의심됩니다." if label == "phishing" else "현재 입력에서 피싱 의심 임계값 미만입니다."
    evidence = "\n".join("- " + reason for reason in reasons[:2]) or "- 현재 규칙에서 뚜렷한 위험 신호를 찾지 못했습니다."
    if len(reasons) > 2:
        evidence += f"\n- 대표 근거 2개를 요약했습니다. 전체 {len(reasons)}개는 상세 근거 목록에서 확인하세요."
    return (f"{conclusion}\n위험 지수: {risk * 100:.1f}/100 · 임계값: {threshold * 100:.1f}/100"
            "\n이 값은 규칙의 위험 지수이며 실제 피싱 확률이나 정확도가 아닙니다."
            f"\n근거:\n{evidence}"
            "\n권장 조치: 메일 속 링크 대신 공식 사이트에 직접 접속하고 발신자를 다른 경로로 확인하세요. 예상치 못한 첨부는 실행하지 마세요."
            "\n규칙이 놓치는 피싱과 정상 메일의 오탐이 모두 가능합니다.")
