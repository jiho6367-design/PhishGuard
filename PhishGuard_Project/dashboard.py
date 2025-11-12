import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st
import altair as alt
import requests

# PowerShell 예시 (한 줄) - <TOKEN>을 발급받은 토큰으로 교체하세요.
# curl.exe -X POST "http://127.0.0.1:8080/api/analyze" -H "Content-Type: application/json" -H "X-API-Key: <TOKEN>" -d "{\"title\":\"Invoice overdue\",\"body\":\"Click here to pay now: http://phish.example\"}"
# $token = "<TOKEN>"
# Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/api/analyze" -Headers @{ "X-API-Key" = $token } -Body (@{title="Invoice overdue"; body="Click here to pay now: http://phish.example"} | ConvertTo-Json) -ContentType "application/json"

API_BASE = os.getenv("PHISH_API_URL", "http://localhost:8080")

st.set_page_config(page_title="PhishGuard Dashboard", layout="wide")
st.title("PhishGuard Operations Dashboard")

env_token = os.getenv("PHISH_API_TOKEN", "").strip()
ui_token = st.text_input("API token (optional) — paste here to use for this session", value="")
API_TOKEN = ui_token.strip() or env_token  # UI 입력을 우선 적용하고, 없으면 환경변수 사용
headers = {"X-API-Key": API_TOKEN} if API_TOKEN else {}
if not API_TOKEN:
    st.warning("API 토큰이 설정되어 있지 않습니다. billing 서비스에서 토큰을 발급받아 위 입력란에 붙여넣으세요.")

@st.cache_data(ttl=300)
def fetch_summary(token: str):
    fallback = {
        "phishing_today": 0,
        "false_positives": 0,
        "avg_feedback_latency_ms": 0,
        "monthly_trend": [],
    }
    headers = {"X-API-Key": token} if token else {}
    try:
        resp = requests.get(f"{API_BASE}/metrics/summary", headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {**fallback, **data}
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", "n/a")
        st.error(f"API 오류: {e} (상태 {status})")
    except requests.RequestException as e:
        st.error("API 연결 실패: " + str(e))
    except ValueError:
        st.error("API 응답을 JSON으로 파싱하지 못했습니다.")
    return fallback

summary = fetch_summary(API_TOKEN)
col1, col2, col3 = st.columns(3)
col1.metric("Phishing Detected (Today)", summary.get("phishing_today", 0))
col2.metric("False Positives (Today)", summary.get("false_positives", 0))
col3.metric(
    "Avg GPT Latency (ms)",
    round(summary.get("avg_feedback_latency_ms") or 0, 1),
)

trend_data = summary.get("monthly_trend") or []
if not trend_data:
    trend_data = [
        {"month": "2024-08-01", "phishing": 42},
        {"month": "2024-09-01", "phishing": 57},
        {"month": "2024-10-01", "phishing": 63},
    ]
trend_df = pd.DataFrame(trend_data)
trend_chart = (
    alt.Chart(trend_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("phishing:Q", title="Phishing Count"),
        tooltip=["month", "phishing"],
    )
    .properties(height=280)
)
st.altair_chart(trend_chart, use_container_width=True)

st.subheader("Upload Suspicious Email")
uploaded = st.file_uploader("Drop .eml/.txt files", type=["txt", "eml"])
subject = st.text_input("Subject override (optional)")
if uploaded:
    email_body = uploaded.read().decode(errors="ignore")
    payload = {"title": subject, "body": email_body}
    with st.spinner("Analyzing..."):
        try:
            resp = requests.post(f"{API_BASE}/api/analyze", json=payload, headers=headers, timeout=15)
        except requests.RequestException as e:
            st.error("API 연결 실패: " + str(e))
        else:
            try:
                data = resp.json()
            except ValueError:
                st.error("API 응답을 JSON으로 파싱하지 못했습니다.")
            else:
                if resp.status_code >= 400:
                    if data.get("error") == "missing_token":
                        st.error("API 토큰이 필요합니다. billing 서비스에서 발급받은 토큰을 입력하거나 환경변수를 설정하세요.")
                    elif data.get("error") == "limit":
                        st.error("무료 호출 한도를 초과했습니다. 요금제를 업그레이드하세요.")
                    else:
                        st.error(f"API 오류({resp.status_code}): {data}")
                else:
                    st.json(data)
