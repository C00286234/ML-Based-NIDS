"""Replay labelled flows against a running API, so the dashboard has something to show.

The flows come from the held-out test split when it exists, otherwise from the
small sample file shipped in the repo (data/sample/demo_flows.parquet). Either
way the model never saw them during training.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import requests

from nids import paths
from nids.features import BENIGN, FEATURE_NAMES_84, LABEL_COLUMN

SAMPLE_FILE = paths.DATA_DIR / "sample" / "demo_flows.parquet"


def make_sample(out: Path = SAMPLE_FILE, per_class: int = 40, benign: int = 600, seed: int = 7) -> Path:
    """Write a small, balanced slice of the test split for the demo (committed to the repo)."""
    from nids.ml.dataset import load_split

    test = load_split("test")
    parts = [test[test[LABEL_COLUMN] == BENIGN].sample(benign, random_state=seed)]
    for _label, group in test[test[LABEL_COLUMN] != BENIGN].groupby(LABEL_COLUMN):
        parts.append(group.sample(min(per_class, len(group)), random_state=seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts)[FEATURE_NAMES_84 + [LABEL_COLUMN]].to_parquet(out, index=False)
    return out


def _load_flows() -> pd.DataFrame:
    test = paths.PROCESSED_DIR / "test.parquet"
    if test.exists():
        return pd.read_parquet(test)
    if SAMPLE_FILE.exists():
        return pd.read_parquet(SAMPLE_FILE)
    raise FileNotFoundError("No flows to replay: run `nids data build`, or check data/sample/ exists.")


def _fake_flow_id(label: str, rng: random.Random) -> str:
    """A plausible 5-tuple so the dashboard can show source and destination."""
    src = f"192.168.10.{rng.randint(2, 254)}" if label == BENIGN else f"172.16.0.{rng.randint(1, 20)}"
    dst = f"192.168.10.{rng.choice([3, 5, 8, 9, 14, 15, 50, 51])}"
    return f"{src}-{dst}-6-{rng.randint(1024, 65535)}-{rng.choice([21, 22, 80, 443, 8080])}"


def run(
    api_url: str = "http://localhost:8000",
    rate: float = 4.0,
    attack_ratio: float = 0.3,
    duration: float | None = None,
    seed: int | None = None,
    api_key: str | None = None,
) -> dict:
    rng = random.Random(seed)
    flows = _load_flows()
    benign = flows[flows[LABEL_COLUMN] == BENIGN]
    attacks = {k: g for k, g in flows[flows[LABEL_COLUMN] != BENIGN].groupby(LABEL_COLUMN)}
    headers = {"X-API-Key": api_key} if api_key else {}
    session = requests.Session()

    sent = correct = 0
    start = time.time()
    print(f"Replaying flows to {api_url} at {rate:g}/s ({attack_ratio:.0%} attacks). Ctrl+C to stop.")
    try:
        while duration is None or time.time() - start < duration:
            if attacks and rng.random() < attack_ratio:
                label = rng.choice(list(attacks))
                row = attacks[label].sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
            else:
                label, row = BENIGN, benign.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
            # NaN is not valid JSON: send null and let the model impute it.
            features = {f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURE_NAMES_84}
            body = {"features": features, "flow_id": _fake_flow_id(label, rng)}
            r = session.post(f"{api_url}/predict", json=body, headers=headers, timeout=10)
            if r.status_code == 429:
                print("Rate limited by the API; waiting 5 s (set NIDS_RATE_LIMIT_ENABLED=false to disable).")
                time.sleep(5)
                continue
            r.raise_for_status()
            pred = r.json()["prediction"]
            sent += 1
            correct += pred == label
            mark = "ok " if pred == label else "MISS"
            print(f"[{mark}] true={label:<24} predicted={pred:<24} conf={r.json()['confidence']:.2f}")
            time.sleep(max(0.0, 1.0 / rate))
    except KeyboardInterrupt:
        pass
    summary = {"sent": sent, "correct": correct, "accuracy": correct / sent if sent else 0.0}
    print(f"\nSent {sent} flows, {correct} classified correctly ({summary['accuracy']:.1%}).")
    return summary
