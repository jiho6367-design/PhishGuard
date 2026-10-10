import csv
import subprocess
import sys
import pytest
import eval as evaluator
from optimized_pipeline import analyze_emails

def test_eval_uses_the_same_rule_scores_as_pipeline(monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    rows = [{"text": "https://bad.xyz"}, {"subject": "회의", "body": "일정입니다."}]
    components = evaluator._predict_components(rows)
    expected = analyze_emails([evaluator._combine_text(row) for row in rows], [evaluator._extract_meta(row) for row in rows])
    assert [row["risk_score"] for row in components] == [row["risk_score"] for row in expected]
    assert components[0]["base_prob"] is None
    assert evaluator._metrics([1, 0], [row["risk_score"] for row in components], 0.2)["tp"] == 1

@pytest.mark.parametrize("label", ["", "maybe", None, "spam"])
def test_unknown_ground_truth_is_not_silently_normal(label):
    with pytest.raises(ValueError):
        evaluator._to_bool(label)

def test_eval_unknown_authentication_is_none():
    auth = evaluator._extract_meta({"spf_pass": "unknown", "dkim_pass": "", "dmarc_pass": "neutral"})["auth_results"]
    assert all(value is None for value in auth.values())

def test_eval_cli_outputs_rule_threshold_sweep_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PHISH_BACKEND", "rules")
    source = tmp_path / "samples.csv"
    source.write_text("text,label\nhttps://bad.xyz,phishing\nhello,normal\n", encoding="utf-8")
    output = tmp_path / "results.csv"
    completed = subprocess.run([sys.executable, "eval.py", str(source), "--sweep", "--out", str(output)], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert "Backend: rules" in completed.stdout and "mw=" not in completed.stdout
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["risk_score"] == "0.3000" and rows[0]["base_prob"] == ""
    assert rows[0]["score_kind"] == "rule_risk_index"
