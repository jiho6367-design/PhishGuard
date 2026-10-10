"""Evaluate the same offline scores used by the API; no model downloads or API calls."""
from __future__ import annotations
import argparse
import csv
import math
import re
from typing import Any, Dict, List, Tuple
from optimized_pipeline import analyze_emails, PHISH_THRESHOLD, MODEL_WEIGHT, get_analysis_backend
from scoring import RULES_VERSION, fuse_scores

def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

def _to_bool(label: str) -> int:
    normalized = str(label or "").strip().casefold()
    if normalized in {"phishing", "1", "true"}:
        return 1
    if normalized in {"normal", "0", "false"}:
        return 0
    raise ValueError("정답 label은 phishing/normal 또는 1/0이어야 합니다. 누락된 라벨은 정상으로 간주하지 않습니다.")

def _combine_text(row: Dict[str, Any]) -> str:
    return str(row.get("text") or "\n".join(str(row.get(field) or "") for field in ("subject", "body") if row.get(field)))

def _extract_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    auth = {}
    for key in ("spf_pass", "dkim_pass", "dmarc_pass"):
        value = str(row.get(key) or "").strip().casefold()
        auth[key] = True if value in {"1", "true", "yes", "pass"} else False if value in {"0", "false", "no", "fail", "softfail"} else None
    attachments = [{"filename": token.strip()} for token in re.split(r"[;,]", str(row.get("attachments") or "")) if token.strip()]
    return {"auth_results": auth, "attachments": attachments,
            "from_header": row.get("from_header") or row.get("sender") or ""}

def _predict_components(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs = analyze_emails([_combine_text(row) for row in rows], [_extract_meta(row) for row in rows])
    return [{"text": item["text"], "base_prob": item["base_phish_probability"],
             "rule_score": item["rule_score"], "risk_score": item["risk_score"],
             "score_kind": item["score_kind"], "rule_signals": item["rule_signals"]} for item in outputs]

def _metrics(y_true: List[int], scores: List[float], threshold: float) -> Dict[str, float]:
    if len(y_true) != len(scores):
        raise ValueError("정답과 예측 점수의 개수가 다릅니다.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("임계값은 0~1 사이의 유한한 숫자여야 합니다.")
    tp = fp = fn = tn = 0
    for truth, score in zip(y_true, scores):
        pred = score + 1e-12 >= threshold
        if truth == 1 and pred:
            tp += 1
        elif truth == 0 and pred:
            fp += 1
        elif truth == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def threshold_sweep(y_true: List[int], scores: List[float], thresholds: List[float]) -> List[Tuple[float, Dict[str, float]]]:
    return [(threshold, _metrics(y_true, scores, threshold)) for threshold in thresholds]

def main():
    parser = argparse.ArgumentParser(description="오프라인 피싱 위험 지수 평가: 실제 피싱 확률이나 정확도 인증이 아닙니다.")
    parser.add_argument("csv_path", help="text 또는 subject/body 및 정답 label 컬럼을 가진 CSV")
    parser.add_argument("--threshold", type=float, default=PHISH_THRESHOLD)
    parser.add_argument("--sweep", action="store_true", help="동일 정답셋에 대해 임계값 0.10~0.90 비교")
    parser.add_argument("--out", default=None, help="행별 위험 지수와 정답을 CSV로 저장")
    args = parser.parse_args()
    try:
        rows = _load_rows(args.csv_path)
        if not rows:
            raise ValueError("평가 CSV에 행이 없습니다.")
        labels = [_to_bool(row.get("label", "")) for row in rows]
        components = _predict_components(rows)
        scores = [row["risk_score"] for row in components]
        metrics = _metrics(labels, scores, args.threshold)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    backend = get_analysis_backend()
    print(f"Records: {len(rows)}  Backend: {backend}  Rules: {RULES_VERSION}  Threshold: {args.threshold:.2f}")
    print("Scores are risk indexes, not calibrated phishing probabilities.")
    print(f"Precision: {metrics['precision']:.3f} Recall: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
          f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    if args.out:
        fields = ["base_prob", "rule_score", "risk_score", "score_kind", "label", "gt_label", "rule_signals"]
        with open(args.out, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for component, score, truth in zip(components, scores, labels):
                writer.writerow({"base_prob": "" if component["base_prob"] is None else f"{component['base_prob']:.4f}",
                                 "rule_score": f"{component['rule_score']:.4f}", "risk_score": f"{score:.4f}",
                                 "score_kind": component["score_kind"], "label": "phishing" if score + 1e-12 >= args.threshold else "normal",
                                 "gt_label": "phishing" if truth else "normal", "rule_signals": "; ".join(component["rule_signals"])})
    if args.sweep:
        thresholds = [round(0.1 + index * 0.05, 2) for index in range(17)]
        print("\nThreshold sweep (precision / recall / f1):")
        weights = [MODEL_WEIGHT] if backend == "rules" else [0.6, 0.65, 0.7, 0.75, 0.8]
        for weight in weights:
            sweep_scores = scores if backend == "rules" else [fuse_scores(row["base_prob"], row["rule_score"], weight) for row in components]
            for threshold, result in threshold_sweep(labels, sweep_scores, thresholds):
                prefix = f"mw={weight:.2f} " if backend == "hf" else ""
                print(f"{prefix}t={threshold:.2f}: P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}")

if __name__ == "__main__":
    main()
