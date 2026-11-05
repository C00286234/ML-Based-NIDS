"""Evaluate the saved model on the held-out test split and write a report to reports/."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from nids import paths
from nids.features import BENIGN, FEATURE_NAMES_84, LABEL_COLUMN
from nids.ml.dataset import load_split
from nids.ml.model import NIDSModel

log = logging.getLogger(__name__)


MIN_RELIABLE_SUPPORT = 30


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


def _plot_confusion(y_true, y_pred, classes, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        log.warning("matplotlib/seaborn not installed; skipping confusion matrix plot")
        return
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1, cbar_kws={"label": "share of true class"}, ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Ensemble confusion matrix (held-out test set, row-normalised)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown(results: dict) -> str:
    lines = ["# Evaluation report", "", f"Held-out test set: **{results['test_rows']:,} flows** "
             f"(never seen during training or model selection).", "",
             "| Model | Accuracy | Macro F1 (all classes) | Macro F1 (classes with 30+ test flows) "
             "| Detection rate | False positive rate |",
             "|---|---|---|---|---|---|"]
    for name, s in results["models"].items():
        lines.append(f"| {name} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {s['macro_f1_reliable']:.4f} | "
                     f"{s['detection_rate']:.4f} | {s['false_positive_rate']:.4%} |")
    lines += ["", "## Per-class results (ensemble)", "", "| Class | Precision | Recall | F1 | Test flows |",
              "|---|---|---|---|---|"]
    few = False
    for cls, r in results["per_class"].items():
        mark = ""
        if r["support"] < MIN_RELIABLE_SUPPORT:
            mark, few = " *", True
        lines.append(f"| {cls}{mark} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support']):,} |")
    if few:
        lines += ["", "\\* Fewer than 30 test flows: treat these scores as indicative only. "
                  "Macro F1 weights every class equally, so these small classes move it a lot."]
    lat = results["latency"]
    lines += ["", "## Speed (single process, this machine)", "",
              f"- One flow: {lat['single_flow_ms_p50']:.1f} ms median, {lat['single_flow_ms_p95']:.1f} ms p95",
              f"- Batch: {lat['batch_1000_flows_per_second']:,.0f} flows/second",
              f"- SHAP explanation: {lat['shap_explain_ms_p50']:.1f} ms median (on demand)", "",
              "![Confusion matrix](confusion_matrix.png)", ""]
    return "\n".join(lines)


def evaluate(model: NIDSModel | None = None, out_dir: Path = paths.REPORTS_DIR) -> dict:
    model = model or NIDSModel.load()
    test = load_split("test")
    X, y = test[FEATURE_NAMES_84], test[LABEL_COLUMN].to_numpy()

    ens, rf_p, xgb_p = model.predict_proba_all(X)
    decode = model.label_encoder.inverse_transform
    preds = {
        f"Ensemble (deployed, RF weight {model.rf_weight:.2f})": decode(ens.argmax(1)),
        "Random Forest": decode(rf_p.argmax(1)),
        "XGBoost": decode(xgb_p.argmax(1)),
    }
    y_ens = next(iter(preds.values()))
    report = classification_report(y, y_ens, output_dict=True, zero_division=0)
    results = {
        "test_rows": len(test),
        "models": {name: _scores(y, p) for name, p in preds.items()},
        "per_class": {c: report[c] for c in model.classes if c in report},
        "latency": _latency(model, X),
        "model_metadata": model.metadata,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    _plot_confusion(y, y_ens, model.classes, out_dir / "confusion_matrix.png")
    (out_dir / "REPORT.md").write_text(_markdown(results))
    log.info("Wrote report to %s", out_dir)
    return results
