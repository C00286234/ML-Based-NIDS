"""Evaluate the saved model on the held-out test split and write a report to reports/.

The test split is used for nothing else. Alert thresholds are chosen on the
validation split and only then measured here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve

from nids import paths
from nids.features import ATTACK_FAMILY, BENIGN, FEATURE_NAMES_84, LABEL_COLUMN
from nids.ml import plots
from nids.ml.dataset import load_split
from nids.ml.model import NIDSModel

log = logging.getLogger(__name__)


MIN_RELIABLE_SUPPORT = 30
FPR_TARGETS = (0.01, 0.001, 0.0001)  # alert budgets: share of benign flows allowed to raise an alert
CONFIDENCE_BANDS = (0.0, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0)
BOOTSTRAP_ROUNDS = 1000
SCAN_CLASSES = ("Portscan", "Infiltration - Portscan")


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    attack_true, attack_pred = y_true != BENIGN, y_pred != BENIGN
    benign = ~attack_true
    labels, counts = np.unique(y_true, return_counts=True)
    reliable = labels[counts >= MIN_RELIABLE_SUPPORT]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        # Same, but only over classes with enough test flows for F1 to mean something.
        "macro_f1_reliable": f1_score(y_true, y_pred, labels=reliable, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        # Attack-vs-benign view: what a SOC analyst cares about first.
        "detection_rate": float((attack_pred & attack_true).sum() / max(attack_true.sum(), 1)),
        "false_positive_rate": float((attack_pred & benign).sum() / max(benign.sum(), 1)),
    }


def _confidence_intervals(y: np.ndarray, pred: np.ndarray, classes: list[str], seed: int = 0) -> dict:
    """95% bootstrap intervals for the headline numbers: how much they would move on another sample of flows."""
    k = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    yi = np.fromiter((index[v] for v in y), dtype=np.int64, count=len(y))
    pi = np.fromiter((index[v] for v in pred), dtype=np.int64, count=len(y))
    cell = yi * k + pi
    benign = index[BENIGN]
    reliable = np.bincount(yi, minlength=k) >= MIN_RELIABLE_SUPPORT
    rng = np.random.default_rng(seed)
    out = {name: [] for name in ("accuracy", "macro_f1_reliable", "detection_rate", "false_positive_rate")}
    for _ in range(BOOTSTRAP_ROUNDS):
        cm = np.bincount(cell[rng.integers(0, len(cell), len(cell))], minlength=k * k).reshape(k, k)
        tp = np.diag(cm)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
        attacks = cm.sum() - cm[benign].sum()
        out["accuracy"].append(tp.sum() / cm.sum())
        out["macro_f1_reliable"].append(f1[reliable].mean())
        out["detection_rate"].append((attacks - (cm[:, benign].sum() - cm[benign, benign])) / max(attacks, 1))
        out["false_positive_rate"].append((cm[benign].sum() - cm[benign, benign]) / max(cm[benign].sum(), 1))
    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for name, v in out.items()}


def _operating_points(score_val, benign_val, score_test, y_test, pred_test) -> list[dict]:
    """Alert thresholds on the attack score (1 - P(benign)), chosen on validation, measured on test."""
    attack = y_test != BENIGN
    classes = sorted(set(y_test[attack]))

    def measure(flagged: np.ndarray) -> dict:
        return {
            "false_positive_rate": float(flagged[~attack].mean()),
            "detection_rate": float(flagged[attack].mean()),
            "recall_by_class": {c: float(flagged[y_test == c].mean()) for c in classes},
        }

    points = [{"name": "Model's own decision (most likely class)", "short": "Most likely class", "threshold": None,
               "target_false_positive_rate": None, **measure(pred_test != BENIGN)}]
    for target in FPR_TARGETS:
        threshold = float(np.quantile(score_val[benign_val], 1 - target))
        points.append({"name": f"Alert budget {100 * target:g}% of benign flows", "short": f"{100 * target:g}% budget",
                       "threshold": threshold, "target_false_positive_rate": target, **measure(score_test > threshold)})
    return points


def _calibration(confidence: np.ndarray, correct: np.ndarray) -> dict:
    bins = []
    for lo, hi in zip(CONFIDENCE_BANDS[:-1], CONFIDENCE_BANDS[1:], strict=True):
        rows = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        n = int(rows.sum())
        bins.append({"from": lo, "to": hi, "flows": n,
                     "mean_confidence": float(confidence[rows].mean()) if n else None,
                     "accuracy": float(correct[rows].mean()) if n else None})
    ece = sum(b["flows"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["flows"]) / len(confidence)
    return {"expected_calibration_error": float(ece), "bins": bins}


def _confusions(cm: np.ndarray, classes: list[str], top: int = 8) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:top]
    rows = []
    for flat in order:
        i, j = divmod(int(flat), len(classes))
        if off[i, j]:
            rows.append({"true": classes[i], "predicted": classes[j], "flows": int(off[i, j]),
                         "share_of_true_class": float(off[i, j] / cm[i].sum())})
    return rows


def _latency(model: NIDSModel, X: pd.DataFrame) -> dict:
    one = X.iloc[[0]]
    model.predict_flows(one)  # warm up
    single = []
    for i in range(200):
        t = time.perf_counter()
        model.predict_flows(X.iloc[[i % len(X)]])
        single.append(time.perf_counter() - t)
    batch = X.iloc[:1000]
    t = time.perf_counter()
    model.predict_flows(batch)
    batch_s = time.perf_counter() - t
    model.explain(one, model.predict(one)[0])  # build the SHAP explainer once
    explain = []
    for i in range(50):
        row = X.iloc[[i % len(X)]]
        t = time.perf_counter()
        model.explain(row, model.predict(row)[0])
        explain.append(time.perf_counter() - t)
    return {
        "single_flow_ms_p50": 1000 * float(np.median(single)),
        "single_flow_ms_p95": 1000 * float(np.percentile(single, 95)),
        "batch_1000_flows_per_second": len(batch) / batch_s,
        "shap_explain_ms_p50": 1000 * float(np.median(explain)),
    }


def evaluate(model: NIDSModel | None = None, out_dir: Path = paths.REPORTS_DIR) -> dict:
    model = model or NIDSModel.load()
    test, val = load_split("test"), load_split("val")
    X, y = test[FEATURE_NAMES_84], test[LABEL_COLUMN].to_numpy()
    classes = model.classes
    benign_col = classes.index(BENIGN)

    ens, rf_p, xgb_p = model.predict_proba_all(X)
    decode = model.label_encoder.inverse_transform
    preds = {
        f"Ensemble (deployed, RF weight {model.rf_weight:.2f})": decode(ens.argmax(1)),
        "Random Forest": decode(rf_p.argmax(1)),
        "XGBoost": decode(xgb_p.argmax(1)),
    }
    y_ens = next(iter(preds.values()))
    report = classification_report(y, y_ens, output_dict=True, zero_division=0)
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, ([index[v] for v in y], [index[v] for v in y_ens]), 1)

    family = np.vectorize(ATTACK_FAMILY.get)
    merge_scans = lambda a: np.where(np.isin(a, SCAN_CLASSES), "Scan (either kind)", a)  # noqa: E731
    val_score = 1 - model.predict_proba(val[FEATURE_NAMES_84])[:, benign_col]
    test_score = 1 - ens[:, benign_col]
    points = _operating_points(val_score, (val[LABEL_COLUMN] == BENIGN).to_numpy(), test_score, y, y_ens)
    calibration = _calibration(ens.max(axis=1), y_ens == y)

    summary = paths.PROCESSED_DIR / "dataset_summary.json"
    results = {
        "test_rows": len(test),
        "split": json.loads(summary.read_text()).get("split", "random") if summary.exists() else "unknown",
        "test_class_counts": {c: int(n) for c, n in zip(*np.unique(y, return_counts=True), strict=True)},
        "models": {name: _scores(y, p) for name, p in preds.items()},
        "confidence_intervals_95": _confidence_intervals(y, y_ens, classes),
        "per_class": {c: report[c] for c in classes if c in report},
        "largest_confusions": _confusions(cm, classes),
        "by_family": _scores(family(y), family(y_ens)),
        "scans_merged": _scores(merge_scans(y), merge_scans(y_ens)),
        "operating_points": points,
        "calibration": calibration,
        "latency": _latency(model, X),
        "model_metadata": model.metadata,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    plots.confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y != BENIGN, test_score)
    plots.detection_tradeoff(fpr, tpr, points, out_dir / "detection_tradeoff.png")
    plots.calibration(calibration["bins"], out_dir / "calibration.png")
    (out_dir / "REPORT.md").write_text(_markdown(results))
    log.info("Wrote report to %s", out_dir)
    return results


# ------------------------------------------------------------------ report


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def _markdown(r: dict) -> str:
    split_note = {
        "time": "The split is time-ordered: on each day the model trained on the earliest 70% of each class "
                "and is tested here on the last 15%.",
        "random": "The split is random, so test flows have near-identical neighbours in the training set. "
                  "Expect these scores to be optimistic.",
    }.get(r["split"], "")
    deployed = next(iter(r["models"]))
    d, ci = r["models"][deployed], r["confidence_intervals_95"]
    lines = [
        "# Evaluation report", "",
        f"Held-out test set: **{r['test_rows']:,} flows**, never seen during training or model selection. {split_note}",
        "",
        "| Model | Accuracy | Macro F1 (all classes) | Macro F1 (classes with 30+ test flows) "
        "| Detection rate | False positive rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in r["models"].items():
        lines.append(f"| {name} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {s['macro_f1_reliable']:.4f} | "
                     f"{s['detection_rate']:.4f} | {s['false_positive_rate']:.4%} |")
    lines += [
        "",
        "Detection rate is the share of attack flows flagged as any attack. False positive rate is the share",
        "of benign flows flagged as an attack.",
        "",
        "95% intervals for the deployed model (bootstrap over test flows; they show sampling noise only,",
        "not how the model would do on a different network):",
        "",
        "| | Value | 95% interval |",
        "|---|---|---|",
        f"| Accuracy | {_pct(d['accuracy'])} | {_pct(ci['accuracy'][0])} to {_pct(ci['accuracy'][1])} |",
        f"| Macro F1 (30+ flows) | {d['macro_f1_reliable']:.4f} | {ci['macro_f1_reliable'][0]:.4f} to {ci['macro_f1_reliable'][1]:.4f} |",
        f"| Detection rate | {_pct(d['detection_rate'])} | {_pct(ci['detection_rate'][0])} to {_pct(ci['detection_rate'][1])} |",
        f"| False positive rate | {_pct(d['false_positive_rate'], 4)} | {_pct(ci['false_positive_rate'][0], 4)} to {_pct(ci['false_positive_rate'][1], 4)} |",
        "",
        "## Per-class results (deployed model)", "",
        "| Class | Precision | Recall | F1 | Test flows |", "|---|---|---|---|---|",
    ]
    few = False
    for cls, c in r["per_class"].items():
        mark = ""
        if c["support"] < MIN_RELIABLE_SUPPORT:
            mark, few = " *", True
        lines.append(f"| {cls}{mark} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1-score']:.3f} | {int(c['support']):,} |")
    if few:
        lines += ["", "\\* Fewer than 30 test flows: treat these scores as indicative only. "
                  "Macro F1 weights every class equally, so these small classes move it a lot."]
    lines += ["", "## Where it goes wrong", "", "| True class | Predicted as | Flows | Share of the true class |", "|---|---|---|---|"]
    for c in r["largest_confusions"]:
        lines.append(f"| {c['true']} | {c['predicted']} | {c['flows']:,} | {_pct(c['share_of_true_class'], 1)} |")
    fam, scan = r["by_family"], r["scans_merged"]
    lines += [
        "",
        "Most errors are between closely related classes. Scored more coarsely:",
        "",
        "| Scoring | Accuracy | Macro F1 (30+ flows) |",
        "|---|---|---|",
        f"| All {len(r['per_class'])} classes | {_pct(d['accuracy'])} | {d['macro_f1_reliable']:.4f} |",
        f"| The two port-scan classes counted as one | {_pct(scan['accuracy'])} | {scan['macro_f1_reliable']:.4f} |",
        f"| By attack family ({', '.join(sorted(set(ATTACK_FAMILY.values()) - {'Benign'}))}) | {_pct(fam['accuracy'])} | {fam['macro_f1_reliable']:.4f} |",
        "",
        "## Choosing an alert threshold", "",
        "Every flow gets an attack score, 1 - P(benign). Alerting on a score threshold instead of the most",
        "likely class trades false alarms for detections. Thresholds below were set on the validation",
        "split to hit a false-alarm budget, then measured on the test split.",
        "",
        "| Setting | Score threshold | False positive rate (test) | Detection rate (test) | Least-detected class (30+ flows) |",
        "|---|---|---|---|---|",
    ]
    big = [c for c, v in r["per_class"].items() if v["support"] >= MIN_RELIABLE_SUPPORT and c != BENIGN]
    for p in r["operating_points"]:
        worst = min(big, key=p["recall_by_class"].get)
        threshold = "n/a" if p["threshold"] is None else f"{p['threshold']:.4f}"
        lines.append(f"| {p['name']} | {threshold} | {_pct(p['false_positive_rate'], 4)} | {_pct(p['detection_rate'])} | "
                     f"{worst} ({_pct(p['recall_by_class'][worst], 1)}) |")
    cal = r["calibration"]
    lines += [
        "", "![Detection against false alarms](detection_tradeoff.png)", "",
        "## Is the confidence score honest?", "",
        "A prediction made with 90% confidence should be right about 90% of the time.",
        "",
        "| Confidence band | Test flows | Mean confidence | Actually correct |",
        "|---|---|---|---|",
    ]
    for b in cal["bins"]:
        if b["flows"]:
            lines.append(f"| {_pct(b['from'], 1)} to {_pct(b['to'], 1)} | {b['flows']:,} | {_pct(b['mean_confidence'], 1)} | {_pct(b['accuracy'], 1)} |")
    lat = r["latency"]
    lines += [
        "", f"Expected calibration error: {_pct(cal['expected_calibration_error'], 3)}.", "",
        "![Calibration](calibration.png)", "",
        "## Speed (single process, this machine)", "",
        f"- One flow: {lat['single_flow_ms_p50']:.1f} ms median, {lat['single_flow_ms_p95']:.1f} ms p95",
        f"- Batch: {lat['batch_1000_flows_per_second']:,.0f} flows/second",
        f"- SHAP explanation: {lat['shap_explain_ms_p50']:.1f} ms median (on demand)", "",
        "## Confusion matrix", "",
        "![Confusion matrix](confusion_matrix.png)", "",
    ]
    return "\n".join(lines)
