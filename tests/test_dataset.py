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


def test_non_ip_rows_removed():
    df = _raw(["BENIGN"] * 5)
    df.loc[0, "Protocol"] = 0
    out, stats = clean(df, min_class_size=1)
    assert stats["non_ip_rows_removed"] == 1 and len(out) == 4


def test_duplicate_keeps_the_earliest_flow():
    df = _raw(["BENIGN"] * 3)
    df.iloc[2] = df.iloc[0]
    df["Timestamp"] = pd.to_datetime(["2017-07-03 10:00", "2017-07-03 09:00", "2017-07-03 08:00"])
    out, _ = clean(df, min_class_size=1)
    assert sorted(out["Timestamp"].dt.hour) == [8, 9]


def _timed(labels_by_day: dict[str, list[str]]) -> pd.DataFrame:
    frames = []
    for i, (day, labels) in enumerate(labels_by_day.items()):
        df = _raw(labels).drop(columns="Attempted Category")
        df["Day"] = day
        df["Timestamp"] = pd.Timestamp(f"2017-07-0{3 + i}") + pd.to_timedelta(range(len(labels)), unit="s")
        frames.append(df.sample(frac=1, random_state=i))  # shuffled: the split must sort by time itself
    return pd.concat(frames, ignore_index=True)


def test_time_split_trains_on_the_past_and_tests_on_the_future():
    df = _timed({"monday": ["BENIGN"] * 100, "friday": ["BENIGN", "DDoS"] * 50})
    parts = split(df)
    assert [len(parts[name]) for name in ("train", "val", "test")] == [142, 29, 29]  # 15% of each class per day, rounded down
    for _, group in df.groupby(["Day", LABEL_COLUMN]):
        times = {name: parts[name].loc[parts[name].index.intersection(group.index), "Timestamp"] for name in parts}
        assert times["train"].max() < times["val"].min() <= times["val"].max() < times["test"].min()
    assert (parts["test"][LABEL_COLUMN] == "DDoS").sum() == 7  # 15% of the 50 DDoS flows, rounded down
    assert set(parts["test"]["Day"]) == {"monday", "friday"}


def test_time_split_keeps_rare_classes_measurable():
    parts = split(_timed({"wednesday": ["BENIGN"] * 40 + ["Heartbleed"] * 3 + ["Infiltration"] * 2}))
    assert (parts["val"][LABEL_COLUMN] == "Heartbleed").sum() == 1
    assert (parts["test"][LABEL_COLUMN] == "Heartbleed").sum() == 1
    assert (parts["train"][LABEL_COLUMN] == "Infiltration").sum() == 2  # too few to hold any out


def test_random_split_is_disjoint_and_stratified():
    df = _raw(["BENIGN"] * 80 + ["DDoS"] * 20)
    df["row"] = range(len(df))
    parts = split(df, how="random")
    ids = [set(p["row"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(p) for p in parts.values()) == 100
    assert (parts["test"][LABEL_COLUMN] == "DDoS").sum() == 3
