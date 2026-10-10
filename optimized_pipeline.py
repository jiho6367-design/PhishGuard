"""No-cost, offline analysis. Optional HF models must already exist locally."""
from __future__ import annotations
import logging
import math
import os
import time
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Sequence
from scoring import RULES_VERSION, compute_rule_score, fuse_scores, decide_label, clamp_confidence

logger = logging.getLogger(__name__)

def _env_unit_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        if math.isfinite(value) and 0 <= value <= 1:
            return value
    except ValueError:
        pass
    logger.warning("%s is invalid; using the documented default.", name)
    return default

PHISH_THRESHOLD = _env_unit_float("PHISH_THRESHOLD", 0.20)
MODEL_WEIGHT = _env_unit_float("MODEL_WEIGHT", 0.7)
FAST_MODEL = os.getenv("PHISH_HF_MODEL", "").strip()
DEVICE = "cpu"

class ModelConfigurationError(RuntimeError):
    """The explicitly selected local phishing classifier is unavailable or invalid."""

def get_analysis_backend() -> str:
    backend = os.getenv("PHISH_BACKEND", "rules").strip().casefold()
    if backend not in {"rules", "hf"}:
        raise ModelConfigurationError("PHISH_BACKEND는 rules 또는 hf만 지원합니다. 유료 API 분석은 제공하지 않습니다.")
    return backend

def get_analysis_policy() -> Dict[str, Any]:
    """Expose the actual runtime policy for history cache invalidation."""
    backend = get_analysis_backend()
    return {
        "backend": backend,
        "model": os.getenv("PHISH_HF_MODEL", "").strip() if backend == "hf" else None,
        "model_weight": MODEL_WEIGHT if backend == "hf" else None,
        "rules_version": RULES_VERSION,
    }

def _validate_threshold(threshold: float | None) -> float:
    try:
        value = PHISH_THRESHOLD if threshold is None else float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("판정 임계값은 0~1 사이의 숫자여야 합니다.") from exc
    if isinstance(threshold, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("판정 임계값은 0~1 사이의 유한한 숫자여야 합니다.")
    return value

def _validate_texts(texts: Sequence[str]) -> List[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("분석 본문은 문자열의 목록이어야 합니다.")
    values = list(texts)
    if any(not isinstance(text, str) for text in values):
        raise TypeError("각 분석 본문은 문자열이어야 합니다.")
    return values

def _validate_label_mapping(mapping: Any) -> Dict[int, str]:
    try:
        labels = {int(index): str(label).strip().upper() for index, label in mapping.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelConfigurationError("모델의 id2label 설정을 읽을 수 없습니다.") from exc
    if set(labels) != {0, 1} or set(labels.values()) != {"PHISHING", "NORMAL"}:
        raise ModelConfigurationError("로컬 모델은 PHISHING/NORMAL 2개 라벨을 명시해야 합니다. 감정분류나 라벨 번호를 피싱으로 추정하지 않습니다.")
    return labels

@lru_cache(maxsize=1)
def _load_local_hf(model_name: str):
    if not model_name:
        raise ModelConfigurationError("HF 분석에는 PHISH_HF_MODEL로 이미 준비된 로컬 모델을 지정해야 합니다.")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ModelConfigurationError("선택 HF 분석에는 별도로 설치한 torch와 transformers가 필요합니다.") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=False, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True, trust_remote_code=False).to("cpu").eval()
        labels = _validate_label_mapping(model.config.id2label)
    except ModelConfigurationError:
        raise
    except Exception as exc:
        raise ModelConfigurationError("로컬 캐시에 모델을 찾을 수 없거나 읽을 수 없습니다. 자동 다운로드는 실행하지 않습니다.") from exc
    return torch, tokenizer, model, labels

def classify_batch(texts: Sequence[str]) -> Sequence[Dict[str, Any]]:
    """Legacy component interface; rules never masquerade as model probabilities."""
    values = _validate_texts(texts)
    backend = get_analysis_backend()
    if not values:
        return []
    if backend == "rules":
        return [{"text": text, "base_label": None, "base_confidence": None,
                 "phish_probability": None, "risk_score": compute_rule_score(text)[0],
                 "backend": "rules", "score_kind": "rule_risk_index"} for text in values]
    model_name = os.getenv("PHISH_HF_MODEL", "").strip()
    torch, tokenizer, model, labels = _load_local_hf(model_name)
    phish_index = next(index for index, label in labels.items() if label == "PHISHING")
    with torch.inference_mode():
        inputs = tokenizer(values, padding=True, truncation=True, max_length=256, return_tensors="pt")
        logits = model(**inputs).logits
        if logits.shape[-1] != 2:
            raise ModelConfigurationError("로컬 모델의 출력은 PHISHING/NORMAL 2개여야 합니다.")
        probabilities = torch.softmax(logits, dim=-1).cpu()
    outputs = []
    for text, probability in zip(values, probabilities):
        index = int(probability.argmax())
        outputs.append({"text": text, "base_label": labels[index].lower(),
                        "base_confidence": clamp_confidence(float(probability[index])),
                        "phish_probability": clamp_confidence(float(probability[phish_index])),
                        "backend": "hf", "model": model_name})
    if len(outputs) != len(values):
        raise ModelConfigurationError("로컬 모델이 요청한 수와 다른 결과 수를 반환했습니다.")
    return outputs

def _make_feedback(label: str, risk: float, threshold: float, reasons: Sequence[str]) -> str:
    conclusion = "현재 규칙 기준으로 피싱이 의심됩니다." if label == "phishing" else "현재 기준으로 피싱 의심 임계값 미만입니다."
    evidence = "\n".join("- " + reason for reason in reasons) or "- 현재 규칙에서 뚜렷한 위험 신호를 찾지 못했습니다."
    return (f"{conclusion}\n위험 지수: {risk * 100:.1f}/100 · 임계값: {threshold * 100:.1f}/100"
            "\n이 값은 규칙 또는 모델 결합 위험 지수이며 실제 피싱 확률·판정 정확도가 아닙니다."
            f"\n근거:\n{evidence}"
            "\n권장 조치: 메일 속 링크 대신 공식 사이트를 직접 열고, 발신자를 다른 경로로 확인하세요. 예기치 않은 첨부는 실행하지 마세요."
            "\n규칙에 잡히지 않는 피싱과 정상 메일의 오탐이 모두 가능합니다.")

def final_judge_with_llm(*args, **kwargs):
    """Compatibility guard: this runtime deliberately has no paid LLM path."""
    raise RuntimeError("유료 LLM 판정은 이 로컬 런타임에서 지원하지 않습니다.")

async def feedback_async(items: Sequence[Dict[str, Any]], detection_policy: str = ""):
    """Legacy async interface with deterministic local feedback only."""
    return [{"content": _make_feedback(str(item.get("label", "normal")),
                                      clamp_confidence(item.get("risk_score", item.get("confidence", 0))),
                                      PHISH_THRESHOLD, item.get("reasons", item.get("rule_signals", []))),
             "latency_ms": 0.0} for item in items]

def analyze_emails(texts: Sequence[str], metas: Sequence[Dict[str, Any]] | None = None,
                   threshold: float | None = None) -> Sequence[Dict[str, Any]]:
    values = _validate_texts(texts)
    decision_threshold = _validate_threshold(threshold)
    if metas is None:
        metadata = [{} for _ in values]
    else:
        if isinstance(metas, (str, bytes)) or not isinstance(metas, Sequence):
            raise TypeError("메일 메타데이터는 객체의 목록이어야 합니다.")
        metadata = list(metas)
        if len(metadata) != len(values) or any(not isinstance(meta, dict) for meta in metadata):
            raise ValueError("메일 본문과 메타데이터 개수는 같고 각 메타데이터는 객체여야 합니다.")
    backend = get_analysis_backend()
    started = time.perf_counter()
    base = classify_batch(values) if backend == "hf" and values else [None] * len(values)
    model_latency = (time.perf_counter() - started) * 1000 / max(1, len(values))
    records = []
    for text, meta, component in zip(values, metadata, base):
        item_started = time.perf_counter()
        rule_score, rule_signals = compute_rule_score(text, meta)
        model_probability = component["phish_probability"] if component else None
        risk = rule_score if component is None else fuse_scores(model_probability, rule_score, MODEL_WEIGHT)
        label = decide_label(risk, decision_threshold)
        reasons = [signal.reason for signal in rule_signals]
        if component:
            reasons.append("명시적 PHISHING/NORMAL 라벨을 가진 로컬 분류기의 출력과 규칙을 결합했습니다.")
        reason = reasons[0] if reasons else "현재 규칙에서 뚜렷한 위험 신호를 찾지 못했습니다."
        auth = meta.get("auth_results") or {}
        warnings = []
        if not isinstance(auth, dict) or any(auth.get(field) is None for field in ("spf_pass", "dkim_pass", "dmarc_pass")):
            warnings.append("일부 메일 인증 결과가 없거나 확인되지 않았습니다. 인증 실패로 가산하지 않았습니다.")
        warnings.extend(str(warning) for warning in meta.get("parser_warnings", []) if warning)
        records.append({"text": text, "label": label, "confidence": risk, "risk_score": risk,
                        "score_kind": "rule_risk_index" if backend == "rules" else "hybrid_risk_index",
                        "rule_score": rule_score, "rule_signals": [signal.reason for signal in rule_signals],
                        "signals": [asdict(signal) for signal in rule_signals], "reasons": reasons,
                        "reason": reason, "brief_reason": reason,
                        "feedback": _make_feedback(label, risk, decision_threshold, reasons),
                        "threshold": decision_threshold, "backend": backend, "rules_version": RULES_VERSION,
                        "base_label": component["base_label"] if component else None,
                        "base_confidence": component["base_confidence"] if component else None,
                        "base_phish_probability": model_probability,
                        "model": component.get("model") if component else None, "warnings": warnings,
                        "latency_ms": model_latency + (time.perf_counter() - item_started) * 1000})
    return records
