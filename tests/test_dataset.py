import numpy as np
import pandas as pd

from nids.features import FEATURE_NAMES_84, LABEL_COLUMN
from nids.ml.dataset import clean, split


def _raw(labels, attempted=None):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(len(labels), len(FEATURE_NAMES_84))), columns=FEATURE_NAMES_84)
    df[LABEL_COLUMN] = labels
    df["Attempted Category"] = attempted if attempted is not None else -1
    return df


def test_attempted_flows_become_benign():
    df = _raw(["DoS Hulk - Attempted", "DoS Hulk"] * 10, attempted=[3, -1] * 10)
    out, stats = clean(df, min_class_size=1)
    assert stats["attempted_rows"] == 10
    assert (out[LABEL_COLUMN] == "BENIGN").sum() == 10


def test_attempted_can_be_merged_into_attack():
    df = _raw(["DoS Hulk - Attempted"] * 5, attempted=[3] * 5)
    out, _ = clean(df, attempted="attack", min_class_size=1)
    assert set(out[LABEL_COLUMN]) == {"DoS Hulk"}


def test_duplicates_and_conflicts_removed():
    df = _raw(["BENIGN"] * 4)
    df.iloc[1] = df.iloc[0]  # exact duplicate
    df.loc[3, FEATURE_NAMES_84] = df.loc[2, FEATURE_NAMES_84].to_numpy()
    df.loc[3, LABEL_COLUMN] = "DDoS"  # same features, different label
    out, stats = clean(df, min_class_size=1)
    assert stats["duplicate_rows_removed"] == 1
    assert stats["conflicting_rows_removed"] == 2
    assert len(out) == 1


def test_web_attacks_merged_and_tiny_classes_dropped():
    df = _raw(["Web Attack - XSS"] * 6 + ["Web Attack - Brute Force"] * 6 + ["Heartbleed"] * 2)
    out, stats = clean(df, min_class_size=5)
    assert set(out[LABEL_COLUMN]) == {"Web Attack"}
    assert stats["classes_dropped_too_small"] == {"Heartbleed": 2}


def test_infinities_become_nan():
    df = _raw(["BENIGN"] * 3)
    df.loc[0, "Flow Bytes/s"] = np.inf
    out, _ = clean(df, min_class_size=1)
    assert out["Flow Bytes/s"].isna().sum() == 1


def test_split_is_disjoint_and_stratified():
    df = _raw(["BENIGN"] * 80 + ["DDoS"] * 20)
    df["row"] = range(len(df))
    parts = split(df)
    ids = [set(p["row"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(p) for p in parts.values()) == 100
    assert (parts["test"][LABEL_COLUMN] == "DDoS").sum() == 3
