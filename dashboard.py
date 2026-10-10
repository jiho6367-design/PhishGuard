"""Optional Streamlit compatibility screen; primary UI is served by Flask.

Importing this module is safe: it does not load models, connect to Gmail, or make
HTTP requests. Each operation requires an explicit form/button action.
"""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


def _local_base_url() -> str:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    value = os.getenv("PHISH_API_URL", "http://127.0.0.1:8080").strip().rstrip("/")
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        port = parts.port
        loopback = host.lower() == "localhost"
        if not loopback:
            loopback = ipaddress.ip_address(host).is_loopback
    except ValueError as exc:
        raise ValueError("PHISH_API_URL에 올바른 로컬 주소를 지정하세요.") from exc
    if (
        parts.scheme != "http"
        or not loopback
        or parts.username is not None
        or parts.password is not None
        or parts.path
        or parts.query
        or parts.fragment
        or port == 0
    ):
        raise ValueError("보조 화면은 http://127.0.0.1:8080 같은 로컬 API 주소만 사용합니다.")
    return value


def request_api(method: str, path: str, token: str = "", payload: dict | None = None) -> dict:
    """One bounded local request; mutations are never automatically retried."""
    import requests

    base = _local_base_url()
    if path not in {"/api/analyze", "/api/history", "/api/health"}:
        raise ValueError("허용되지 않은 API 경로입니다.")
    try:
        response = requests.request(
            method,
            base + path,
            headers={"X-API-Key": token} if token else {},
            json=payload,
            timeout=(3, 10),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError("로컬 API에 연결하지 못했습니다. 서버를 확인한 뒤 버튼으로 다시 시도하세요.") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("API가 올바른 JSON 응답을 보내지 않았습니다.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("API 응답 형식을 확인할 수 없습니다.")
    if not 200 <= response.status_code < 300:
        error = data.get("detail") or data.get("error", "요청을 처리하지 못했습니다.")
        if isinstance(error, dict):
            error = error.get("message", "요청을 처리하지 못했습니다.")
        raise RuntimeError(f"API 오류 ({response.status_code}): {error}")
    return data


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("선택 화면은 requirements-streamlit.txt 설치 후 streamlit run dashboard.py로 실행하세요.") from exc

    st.set_page_config(page_title="PhishGuard 보조 화면", layout="centered")
    st.title("PhishGuard · 보조 화면")
    st.caption("무료 로컬 분석 · 위험 지표는 피싱 확률이나 안전 보증이 아닙니다.")
    try:
        base = _local_base_url()
    except ValueError as exc:
        st.error(str(exc))
        return
    st.link_button("기본 PhishGuard 화면 열기", base)
    st.info("기본 화면에서 분석, 이력, 설정, 취소 기능을 사용할 수 있습니다. 이 화면은 이전 Streamlit 실행 명령을 위한 보조 화면입니다.")
    token = st.text_input("로컬 API 토큰 (설정한 경우만)", type="password", key="local_api_token")
    token = token.strip() or os.getenv("PHISH_API_TOKEN", "").strip()
    with st.form("local_analysis", clear_on_submit=False):
        subject = st.text_input("메일 제목", max_chars=300, key="subject")
        sender = st.text_input("발신자", max_chars=500, key="sender")
        body = st.text_area("메일 본문", height=220, max_chars=100000, key="body")
        uploaded = st.file_uploader("또는 UTF-8 텍스트 파일", type=["txt"], key="text_file")
        submitted = st.form_submit_button("로컬 분석", type="primary")
    if submitted:
        if uploaded is not None:
            if uploaded.size > 400000:
                st.error("텍스트 파일은 400 KB 이내로 선택하세요. 입력은 유지됩니다.")
                return
            try:
                body = uploaded.getvalue().decode("utf-8-sig")
            except UnicodeDecodeError:
                st.error("UTF-8 텍스트 파일을 선택하세요. 입력은 유지됩니다.")
                return
        if not (subject.strip() or body.strip()):
            st.error("제목 또는 본문을 입력하세요.")
        else:
            try:
                with st.spinner("로컬에서 분석 중…"):
                    st.session_state["analysis_result"] = request_api(
                        "POST", "/api/analyze", token,
                        {"subject": subject, "body": body, "sender": sender},
                    )
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
    result = st.session_state.get("analysis_result")
    if result:
        st.subheader("마지막 분석 결과")
        st.write("검토 필요" if result.get("label") == "phishing" else "낮은 위험")
        st.caption("규칙 위험 지표이며 낮은 점수도 메일의 안전을 보장하지 않습니다.")
        st.text(result.get("feedback") or result.get("gpt_feedback") or "근거 없음")
        st.json(result)
    if st.button("이력 조회"):
        try:
            st.session_state["history_result"] = request_api("GET", "/api/history", token)
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    if "history_result" in st.session_state:
        records = st.session_state["history_result"].get("results", [])
        if records:
            st.json(records)
        else:
            st.info("아직 분석 이력이 없습니다.")


if __name__ == "__main__":
    main()
