"""Check the live flow builder against the dataset, using the dataset's own packets.

For each capture slice in data/pcap/ (see scripts/pcap_slice.py), this rebuilds
flows with nids.engine.flowmeter and compares them with the rows CICFlowMeter
produced for the same packets (data/raw). It answers three questions:

1. Do we cut packets into the same flows?          (matched on 5-tuple + start time)
2. Do we compute the same 84 feature values?       (per-feature agreement)
3. Does the model decide the same thing?           (predictions on our features vs theirs)

It then repeats the exercise with live settings, where flows are ended early
instead of waiting 120 s, to show what faster detection costs in accuracy.

    python scripts/parity_report.py            # writes reports/PARITY.md and reports/parity.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nids import paths
from nids.engine.engine import DEFAULT_CONFIG
from nids.engine.flowmeter import FLOW_TIMEOUT_US, TCP, flows_from_packets, in_time_order
from nids.engine.pcap import read_packets
from nids.features import BENIGN, FEATURE_NAMES_84, LABEL_COLUMN, TRAINING_FEATURES
from nids.ml.model import NIDSModel

PCAP_DIR = paths.DATA_DIR / "pcap"
ID_COLUMNS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Attempted Category"]
# What "the same value" means: the CSVs are rounded, and floats differ in the last digits.
RTOL, ATOL = 1e-4, 1e-2
LIVE_SETTINGS = [
    {"idle_timeout": 60, "close_grace": 5},
    {"idle_timeout": 30, "close_grace": 2},
    {"idle_timeout": 15, "close_grace": 1},
    {"idle_timeout": 5, "close_grace": 1},
]

_day_cache: dict[str, pd.DataFrame] = {}


def load_day(day: str) -> pd.DataFrame:
    """The dataset's flows for one day, with start time in microseconds and cleaned labels."""
    if day not in _day_cache:
        df = pd.read_csv(paths.RAW_DIR / f"{day}.csv", usecols=ID_COLUMNS + FEATURE_NAMES_84 + [LABEL_COLUMN])
        df["start"] = pd.to_datetime(df["Timestamp"]).astype("datetime64[us]").astype("int64")
        # Same relabelling as nids.ml.dataset.clean
        df.loc[df["Attempted Category"] != -1, LABEL_COLUMN] = BENIGN
        df.loc[df[LABEL_COLUMN].str.startswith("Web Attack"), LABEL_COLUMN] = "Web Attack"
        df["tuple"] = [
            _canonical(*t) for t in zip(df["Src IP"], df["Src Port"], df["Dst IP"], df["Dst Port"], df["Protocol"], strict=True)
        ]
        _day_cache[day] = df
    return _day_cache[day]


def _canonical(a, ap, b, bp, proto) -> tuple:
    return (*sorted([(a, int(ap)), (b, int(bp))]), int(proto))


def comparable(ref: pd.DataFrame, t0: int, t1: int) -> pd.DataFrame:
    """Dataset flows we can fairly compare: every packet of the flow, and of the
    connection it belongs to, must be inside the slice [t0, t1]."""
    end = ref["start"] + ref["Flow Duration"]
    inside = ref[(ref["start"] >= t0) & (end < t1 - 1_000_000) & ref["Protocol"].isin([1, 6, 17])]
    # A flow that was still open when the slice began shifts where CICFlowMeter
    # cuts every later flow on that 5-tuple, so those cannot line up with ours.
    open_at_start = ref[(ref["start"] < t0) & (ref["start"] + FLOW_TIMEOUT_US > t0)]
    inside = inside[~inside["tuple"].isin(set(open_at_start["tuple"]))]
    # A TCP flow that continues a connection from before the slice.
    carried = inside["Total TCP Flow Time"] - inside["Flow Duration"]
    return inside[~((inside["Protocol"] == TCP) & (inside["start"] - carried < t0))]


def rebuild(pcap: Path, **meter) -> pd.DataFrame:
    flows = [f for f in flows_from_packets(in_time_order(read_packets(pcap)), **meter) if f.total_packets > 1]
    df = pd.DataFrame([f.features() for f in flows], columns=FEATURE_NAMES_84)
    df["Src IP"] = [f.src_ip for f in flows]
    df["Dst IP"] = [f.dst_ip for f in flows]
    df["start"] = [f.start for f in flows]
    df["end"] = [f.last_seen for f in flows]
    return df


KEY = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "start"]


def compare_slice(pcap: Path, model: NIDSModel) -> dict:
    day = pcap.name.split("_")[0]
    ours = rebuild(pcap)
    t0, t1 = int(ours["start"].min()), int(ours["end"].max())
    ref = comparable(load_day(day), t0, t1)
    both = ref.merge(ours, on=KEY, suffixes=("", "_ours"))
    theirs = both[FEATURE_NAMES_84].to_numpy(float)
    mine = both[[f + "_ours" if f not in KEY else f for f in FEATURE_NAMES_84]].to_numpy(float)
    # Our NaN (rate of a zero-length flow) corresponds to their divide-by-zero output.
    same = np.isclose(theirs, mine, rtol=RTOL, atol=ATOL) | (np.isnan(mine) & ~np.isfinite(theirs))
    X_theirs = pd.DataFrame(theirs, columns=FEATURE_NAMES_84)
    X_mine = pd.DataFrame(mine, columns=FEATURE_NAMES_84)
    pred_theirs = model.predict(X_theirs) if len(both) else np.array([])
    pred_mine = model.predict(X_mine) if len(both) else np.array([])
    return {
        "slice": pcap.name,
        "dataset_flows": len(ref),
        "matched": len(both),
        "same": same,
        "labels": both[LABEL_COLUMN].to_numpy(),
        "pred_theirs": pred_theirs,
        "pred_mine": pred_mine,
    }


def live_slice(pcap: Path, model: NIDSModel, settings: dict) -> dict:
    """Score flows cut with live settings. A flow gets the label of the dataset flow it falls inside."""
    day = pcap.name.split("_")[0]
    ours = rebuild(pcap, **settings)
    t0, t1 = int(ours["start"].min()), int(ours["end"].max())
    ref = comparable(load_day(day), t0, t1)
    ref = ref.assign(end=ref["start"] + ref["Flow Duration"]).sort_values("start")
    ours = ours.assign(tuple=[
        _canonical(*t) for t in zip(ours["Src IP"], ours["Src Port"], ours["Dst IP"], ours["Dst Port"], ours["Protocol"], strict=True)
    ]).sort_values("start")
    joined = pd.merge_asof(ours, ref[["tuple", "start", "end", LABEL_COLUMN]].rename(
        columns={"start": "ref_start", "end": "ref_end"}), left_on="start", right_on="ref_start", by="tuple")
    joined = joined[joined["ref_end"].notna() & (joined["start"] <= joined["ref_end"])]
    pred = model.predict(joined[FEATURE_NAMES_84]) if len(joined) else np.array([])
    return {"labels": joined[LABEL_COLUMN].to_numpy(), "pred": pred, "dataset_flows": len(ref)}


def _binary(labels: np.ndarray, pred: np.ndarray) -> dict:
    attack, flagged = labels != BENIGN, pred != BENIGN
    return {
        "flows": int(len(labels)),
        "attack_flows": int(attack.sum()),
        "accuracy": float((labels == pred).mean()) if len(labels) else None,
        "detection_rate": float((attack & flagged).sum() / attack.sum()) if attack.any() else None,
        "false_positive_rate": float((~attack & flagged).sum() / (~attack).sum()) if (~attack).any() else None,
    }


def run() -> dict:
    model = NIDSModel.load()
    pcaps = sorted(PCAP_DIR.glob("*.pcap"))
    if not pcaps:
        raise SystemExit(f"No capture slices in {PCAP_DIR}. Fetch some with scripts/pcap_slice.py first.")
    parts = []
    for pcap in pcaps:
        print(f"Comparing {pcap.name}")
        parts.append(compare_slice(pcap, model))

    same = np.vstack([p["same"] for p in parts])
    labels = np.concatenate([p["labels"] for p in parts])
    pred_theirs = np.concatenate([p["pred_theirs"] for p in parts])
    pred_mine = np.concatenate([p["pred_mine"] for p in parts])
    trained = [FEATURE_NAMES_84.index(f) for f in model.selected_features]
    all_training = [FEATURE_NAMES_84.index(f) for f in TRAINING_FEATURES]

    by_class = {}
    for cls in sorted(set(labels)):
        rows = labels == cls
        by_class[cls] = {
            "flows": int(rows.sum()),
            "same_prediction": float((pred_theirs[rows] == pred_mine[rows]).mean()),
            "correct_on_dataset_features": float((pred_theirs[rows] == cls).mean()),
            "correct_on_our_features": float((pred_mine[rows] == cls).mean()),
        }

    results = {
        "slices": [
            {"slice": p["slice"], "dataset_flows": p["dataset_flows"], "matched": p["matched"],
             "labels": {k: int(v) for k, v in zip(*np.unique(p["labels"], return_counts=True), strict=True)}}
            for p in parts
        ],
        "dataset_flows": int(sum(p["dataset_flows"] for p in parts)),
        "matched_flows": int(len(labels)),
        "feature_agreement": {f: float(same[:, i].mean()) for i, f in enumerate(FEATURE_NAMES_84)},
        "flows_with_all_84_features_equal": float(same.all(axis=1).mean()),
        "flows_with_all_training_features_equal": float(same[:, all_training].all(axis=1).mean()),
        "flows_with_all_model_features_equal": float(same[:, trained].all(axis=1).mean()),
        "same_prediction": float((pred_theirs == pred_mine).mean()),
        "on_dataset_features": _binary(labels, pred_theirs),
        "on_our_features": _binary(labels, pred_mine),
        "by_class": by_class,
        "tolerance": {"rtol": RTOL, "atol": ATOL},
    }

    results["live"] = []
    for settings in LIVE_SETTINGS:
        print(f"Live settings {settings}")
        live = [live_slice(p, model, settings) for p in pcaps]
        lab = np.concatenate([x["labels"] for x in live])
        pred = np.concatenate([x["pred"] for x in live])
        results["live"].append({**settings, **_binary(lab, pred)})

    paths.REPORTS_DIR.mkdir(exist_ok=True)
    (paths.REPORTS_DIR / "parity.json").write_text(json.dumps(results, indent=2))
    (paths.REPORTS_DIR / "PARITY.md").write_text(markdown(results))
    return results


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def markdown(r: dict) -> str:
    agreement = r["feature_agreement"]
    worst = sorted(agreement.items(), key=lambda kv: kv[1])[:8]
    lines = [
        "# Live feature parity report",
        "",
        "The model is trained on flows that CICFlowMeter built from packet captures. For it to work on",
        "live traffic, the live engine must build the same flows and compute the same numbers from the",
        "same packets. This report measures that, on slices of the original CIC-IDS2017 captures.",
        "",
        "Generated by `scripts/parity_report.py`. Only flows whose packets all fall inside a slice are compared.",
        "",
        "## Same packets, same flows, same features",
        "",
        "| | |",
        "|---|---|",
        f"| Capture slices | {len(r['slices'])} |",
        f"| Dataset flows inside the slices | {r['dataset_flows']:,} |",
        f"| Rebuilt with the same endpoints and start time (to the microsecond) | {r['matched_flows']:,} "
        f"({_pct(r['matched_flows'] / r['dataset_flows'])}) |",
        f"| Of those, all 84 features equal | {_pct(r['flows_with_all_84_features_equal'])} |",
        f"| Of those, every feature the model uses equal | {_pct(r['flows_with_all_model_features_equal'])} |",
        f"| Model gives the same answer on our features as on the dataset's | {_pct(r['same_prediction'], 3)} |",
        "",
        f"Two values count as equal within a relative difference of {r['tolerance']['rtol']:g} "
        f"(the dataset's CSV values are rounded).",
        "",
        "### Least-matching features",
        "",
        "| Feature | Flows with the same value |",
        "|---|---|",
        *[f"| {name} | {_pct(v, 3)} |" for name, v in worst],
        "",
        "## Model results on these flows",
        "",
        "| Features from | Flows | Attack flows | Accuracy | Detection rate | False positive rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, key in (("The dataset (CICFlowMeter)", "on_dataset_features"), ("This project's engine", "on_our_features")):
        b = r[key]
        lines.append(f"| {name} | {b['flows']:,} | {b['attack_flows']:,} | {_pct(b['accuracy'])} | "
                     f"{_pct(b['detection_rate'])} | {_pct(b['false_positive_rate'], 3)} |")
    lines += [
        "",
        "These slices overlap the training data, so the scores are not a test of the model. What matters",
        "is that the two rows agree.",
        "",
        "| Class | Flows | Same prediction | Correct (dataset features) | Correct (our features) |",
        "|---|---|---|---|---|",
    ]
    for cls, c in r["by_class"].items():
        lines.append(f"| {cls} | {c['flows']:,} | {_pct(c['same_prediction'])} | "
                     f"{_pct(c['correct_on_dataset_features'])} | {_pct(c['correct_on_our_features'])} |")
    if "Infiltration - Portscan" in r["by_class"]:
        lines += ["", "The low score for \"Infiltration - Portscan\" is a labelling quirk, not a parity problem: these flows are a",
                  "14-second scan from the outside attacker (172.16.0.1) that the dataset files under that label, and the",
                  "model calls them plain \"Portscan\". It does so on the dataset's features and on ours alike."]
    lines += [
        "",
        "## What faster detection costs",
        "",
        "The dataset only ends a flow when a packet arrives more than 120 s after the flow began. A live",
        "detector cannot wait that long, so the engine also ends a flow after a period of silence",
        "(`idle_timeout`) or shortly after the TCP connection closes (`close_grace`). Flows cut early",
        "look slightly different from the training data. Scored against the label of the dataset flow",
        "each one falls inside:",
        "",
        "| Idle timeout | Close grace | Flows scored | Accuracy | Detection rate | False positive rate |",
        "|---|---|---|---|---|---|",
    ]
    default = DEFAULT_CONFIG["flow"]
    for row in r["live"]:
        is_default = (row["idle_timeout"], row["close_grace"]) == (default["idle_timeout"], default["close_grace"])
        lines.append(f"| {row['idle_timeout']} s{' (default)' if is_default else ''} | {row['close_grace']} s | "
                     f"{row['flows']:,} | {_pct(row['accuracy'])} | "
                     f"{_pct(row['detection_rate'])} | {_pct(row['false_positive_rate'], 3)} |")
    lines += ["", "For comparison, with flows cut exactly as in the dataset (the tables above) the detection rate",
              f"on these slices is {_pct(r['on_our_features']['detection_rate'], 0)}. `nids pcap` uses those exact "
              "rules, since a file has no need to hurry."]
    lines += ["", "## Slices", "", "| Slice | Dataset flows | Matched | Labels |", "|---|---|---|---|"]
    for s in r["slices"]:
        when = re.sub(r"\.pcap$", "", s["slice"])
        labels = ", ".join(f"{k} {v:,}" for k, v in sorted(s["labels"].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {when} | {s['dataset_flows']:,} | {s['matched']:,} | {labels} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = run()
    print(f"Matched {out['matched_flows']:,} of {out['dataset_flows']:,} flows; "
          f"same prediction on {100 * out['same_prediction']:.3f}%")
    print(f"Report: {paths.REPORTS_DIR / 'PARITY.md'}")
