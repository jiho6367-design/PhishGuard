import asyncio
import builtins
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
import optimized_pipeline as pipeline
from scoring import compute_rule_score

@pytest.fixture(autouse=True)
def rules_mode(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    monkeypatch.delenv("PHISH_HF_MODEL", raising=False)

def test_risk_index_not_inverted_or_reduced_for_normal_labels():
    result = pipeline.analyze_emails(["회의 일정입니다."])[0]
    assert result["risk_score"] == result["confidence"] == result["rule_score"] == 0
    assert result["label"] == "normal" and result["score_kind"] == "rule_risk_index"

def test_rules_risk_is_not_scaled_by_missing_model():
    text = "https://bad.xyz"
    result = pipeline.analyze_emails([text])[0]
    assert result["risk_score"] == compute_rule_score(text)[0] == pytest.approx(0.3)
    assert result["label"] == "phishing"
    assert result["base_phish_probability"] is None

def test_threshold_applies_to_labels_and_feedback():
    result = pipeline.analyze_emails(["https://bad.xyz"], threshold=0.8)[0]
    assert result["label"] == "normal" and result["threshold"] == 0.8
    assert "80.0" in result["feedback"] and "확률" in result["feedback"]
    assert result["signals"][0]["name"] == "suspicious_tld"
    assert result["reason"] == result["brief_reason"]
    assert result["latency_ms"] >= 0

@pytest.mark.parametrize("value", [True, "bad", float("nan"), float("inf"), -0.1, 1.1])
def test_bad_threshold_is_rejected(value):
    with pytest.raises(ValueError):
        pipeline.analyze_emails(["hello"], threshold=value)

@pytest.mark.parametrize("texts,metas", [("single string", None), ([None], None), (["one"], []), (["one"], [None])])
def test_invalid_batch_input_is_rejected(texts, metas):
    with pytest.raises((TypeError, ValueError)):
        pipeline.analyze_emails(texts, metas)

def test_empty_batch_is_supported():
    assert pipeline.analyze_emails([]) == []

def test_paid_llm_compatibility_guard():
    with pytest.raises(RuntimeError, match="LLM"):
        pipeline.final_judge_with_llm("text", 0, 0, [])
    # This compatibility coroutine has no await/I/O; exercise it without a Windows
    # event loop that would create a separate socket pair unrelated to analysis.
    coroutine = pipeline.feedback_async([{"label": "normal", "confidence": 0}])
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    feedback = completed.value.value
    assert "위험 지수" in feedback[0]["content"]

def test_import_and_analysis_do_not_load_paid_model_or_google_libraries():
    script = """
import builtins, socket, os
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'openai', 'torch', 'transformers', 'google', 'googleapiclient'}:
        raise AssertionError('forbidden dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def denied(*args, **kwargs):
    raise AssertionError('network access attempted')
socket.socket = denied
socket.create_connection = denied
os.environ['OPENAI_API_KEY'] = 'sentinel-not-real'
os.environ['ENABLE_LLM_JUDGE'] = '1'
os.environ['PHISH_BACKEND'] = 'rules'
import optimized_pipeline, mail_service, server.gmail_client, eval
result = optimized_pipeline.analyze_emails(['verify https://bad.xyz'])[0]
assert result['backend'] == 'rules'
assert result['risk_score'] > 0
print('offline-ok')
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "offline-ok"

@pytest.mark.parametrize("mapping", [{0: "NEGATIVE", 1: "POSITIVE"}, {0: "LABEL_0", 1: "LABEL_1"}, {0: "NOT_PHISHING", 1: "NORMAL"}, {0: "PHISHING", 1: "NORMAL", 2: "OTHER"}])
def test_sentiment_and_ambiguous_labels_are_rejected(mapping):
    with pytest.raises(pipeline.ModelConfigurationError):
        pipeline._validate_label_mapping(mapping)

def test_explicit_phishing_and_normal_labels_are_valid():
    assert pipeline._validate_label_mapping({"0": "phishing", "1": "normal"}) == {0: "PHISHING", 1: "NORMAL"}

def test_optional_hf_loader_forces_local_cache_without_remote_code(monkeypatch):
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "PHISHING", 1: "NORMAL"}))
    model.to = lambda device: model
    model.eval = lambda: model
    tokenizer_cls = SimpleNamespace(from_pretrained=lambda name, **kwargs: calls.append(("tokenizer", kwargs)) or object())
    model_cls = SimpleNamespace(from_pretrained=lambda name, **kwargs: calls.append(("model", kwargs)) or model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer_cls, AutoModelForSequenceClassification=model_cls))
    pipeline._load_local_hf.cache_clear()
    try:
        pipeline._load_local_hf("preinstalled-model")
        assert len(calls) == 2
        assert all(kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False for _, kwargs in calls)
    finally:
        pipeline._load_local_hf.cache_clear()

def test_missing_hf_model_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "hf")
    with pytest.raises(pipeline.ModelConfigurationError, match="PHISH_HF_MODEL"):
        pipeline.analyze_emails(["hello"])

def test_paid_backend_is_not_supported(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "openai")
    with pytest.raises(pipeline.ModelConfigurationError):
        pipeline.analyze_emails(["hello"])

def test_policy_reports_actual_dynamic_model_weight(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "hf")
    monkeypatch.setenv("PHISH_HF_MODEL", " already-local ")
    monkeypatch.setattr(pipeline, "MODEL_WEIGHT", 0.1)
    first = pipeline.get_analysis_policy()
    monkeypatch.setattr(pipeline, "MODEL_WEIGHT", 0.9)
    second = pipeline.get_analysis_policy()
    assert first["model_weight"] == 0.1 and second["model_weight"] == 0.9
    assert second["model"] == "already-local"
    assert second["rules_version"] == pipeline.RULES_VERSION
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    assert pipeline.get_analysis_policy()["model_weight"] is None
