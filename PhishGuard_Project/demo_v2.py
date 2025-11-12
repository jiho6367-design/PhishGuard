# demo_v2.py
import os
import re
import textwrap
from datetime import datetime

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from feedback_loop import record_user_feedback, generate_report

# =========================
# 0) Config
# =========================
THRESHOLD = 0.30  # 이 이상일 때만 GPT 피드백 호출
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# =========================
# 1) Load classifier (DistilBERT)
# =========================
@st.cache_resource(show_spinner=False)
def load_classifier():
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
    return tok, mdl

tokenizer, model = load_classifier()
id2label = model.config.id2label  # 예: {0:'NEGATIVE', 1:'POSITIVE'}

def classify(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs))
        conf = float(probs[0, idx])
    raw = id2label[idx]
    label = "phishing" if raw.upper().startswith("NEG") else "normal"
    return label, conf, [float(p) for p in probs[0]]


# =========================
# 2) PII 마스킹 유틸 (GPT 호출 전 개인정보 최소화)
# =========================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+")

def redact(text: str) -> str:
    text = EMAIL_RE.sub("[MASKED_EMAIL]", text)
    text = URL_RE.sub("[MASKED_URL]", text)
    return text


# =========================
# 3) GPT 피드백 (OpenAI 최신 SDK용)
# =========================
def get_openai_client_or_error():
    """
    성공 시 (client, None)
    실패 시 (None, 오류메시지)
    """
    try:
        from openai import OpenAI
    except Exception as e:
        return None, f"❌ openai 패키지 임포트 실패: {e}\n→ pip install 'openai>=1.0.0' 로 확인하세요."

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None, "❌ OPENAI_API_KEY가 설정되지 않았습니다. PowerShell에서 setx 후 새 터미널 열기!"

    base_url = os.getenv("OPENAI_BASE_URL") or None
    try:
        if base_url:
            client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"❌ OpenAI 클라이언트 생성 실패: {e}"


def generate_feedback(email_text: str, label: str, score: float) -> str:
    client, err = get_openai_client_or_error()
    if err:
        return f"⚠️ GPT 클라이언트 오류: {err}"

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    sys_msg = "You are a cybersecurity analyst. Respond in Korean, concise and practical."
    user_prompt = f"""
[Email Content]
{redact(email_text)}

[Model Result]
Label: {label}, Score: {score:.2f}

[Task]
1) 왜 이렇게 분류됐는지 2~3문장으로 설명  
2) 사용자가 취할 조치를 3가지 불릿으로 제시  
3) 과도한 공포 유발 금지, 실제 행동 중심
"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT 호출 오류: {e}"


# =========================
# 4) Streamlit UI
# =========================
st.set_page_config(page_title="PhishGuard Demo v2", layout="centered")
st.title("🛡️ PhishGuard Demo v2 – 피싱 메일 탐지 + GPT 피드백")

with st.form("phish_form"):
    subject = st.text_input("이메일 제목")
    body = st.text_area("이메일 본문", height=160)
    submitted = st.form_submit_button("Analyze")

if submitted:
    text = (subject or "") + "\n" + (body or "")
    if not text.strip():
        st.warning("⚠️ 제목/본문을 입력하세요.")
        st.stop()

    with st.spinner("🤖 DistilBERT 모델 분석 중..."):
        label, score, probs = classify(text)

    st.subheader("📊 탐지 결과")
    st.write(f"**Label:** {label}")
    st.write(f"**Confidence:** {score:.2f}")
    st.markdown("#### User Feedback Placeholder")
    st.caption("UI ???�서 ???�자 판정??받을 예정입니다. 현재??모델 판정??임시 사용 중.")

    feedback_email_id = "ui-" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    try:
        THRESHOLD = record_user_feedback(
            email_id=feedback_email_id,
            model_label=label,
            model_score=score,
            user_feedback=label,
        )
        st.caption(f"Adaptive threshold: {THRESHOLD:.2f}")
    except Exception as exc:
        st.warning(f"Feedback logging unavailable: {exc}")

    need_gpt = (label != "normal") and (score >= THRESHOLD)
    if need_gpt:
        with st.spinner("💬 GPT 피드백 생성 중..."):
            fb = generate_feedback(text, label, score)
        st.markdown("### 🧠 GPT 피드백")
        st.write(fb)
    else:
        st.info("✅ 정상 메일이거나 신뢰도 낮음 → GPT 피드백 생략 (임계치 조정 가능)")

# =========================
# 5) 연결 진단 패널
# =========================
with st.expander("🔧 GPT 연결 진단", expanded=False):
    import sys
    import importlib.metadata as im
    try:
        version = im.version("openai")
    except:
        version = "Not Installed"
    st.write({
        "Python Version": sys.version.split()[0],
        "OpenAI SDK": version,
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "API Key 존재 여부": bool(os.getenv("OPENAI_API_KEY")),
    })


REPORT_STATE_KEY = "feedback_report_path"
if st.button("Generate Feedback Report PDF"):
    try:
        report_path = generate_report()
        st.session_state[REPORT_STATE_KEY] = str(report_path)
        st.success("PDF report generated.")
    except Exception as exc:
        st.error(f"Report generation failed: {exc}")

saved_report = st.session_state.get(REPORT_STATE_KEY)
if saved_report and os.path.exists(saved_report):
    with open(saved_report, "rb") as fh:
        st.download_button(
            "Download latest report",
            data=fh.read(),
            file_name=os.path.basename(saved_report),
            mime="application/pdf",
        )
