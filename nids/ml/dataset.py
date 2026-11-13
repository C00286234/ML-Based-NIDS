"""Download CIC-IDS2017 (improved) and turn it into clean train / validation / test splits.

The dataset is the re-labelled CIC-IDS2017 from Engelen et al. (KU Leuven, 2021),
which fixes many labelling and flow-construction errors in the original release:
https://intrusion-detection.distrinet-research.be/CNS2022/
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from nids import paths
from nids.features import BENIGN, FEATURE_NAMES_84, LABEL_COLUMN

log = logging.getLogger(__name__)

DATASET_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
DAY_FILES = ["monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv", "friday.csv"]
SPLITS = ("train", "val", "test")
# Kept next to the features for splitting by time and for analysis. Never model inputs.
META_COLUMNS = ["Timestamp", "Day", "Src IP", "Dst IP", "Attempted"]


def download(dest: Path = paths.RAW_DIR, url: str = DATASET_URL) -> Path:
    """Download and unzip the dataset (about 330 MB zipped, 1.1 GB unzipped)."""
    if all((dest / f).exists() for f in DAY_FILES):
        log.info("Dataset already present in %s", dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "CICIDS2017_improved.zip"
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in DAY_FILES:
                (dest / name).write_bytes(z.read(member))
    zip_path.unlink()
    return dest


def load_raw(raw_dir: Path = paths.RAW_DIR) -> pd.DataFrame:
    """Load the five day files: the 84 features, the label, the attempted flag and META_COLUMNS."""
    dtypes = {name: "float32" for name in FEATURE_NAMES_84} | {"Src IP": "category", "Dst IP": "category"}
    frames = []
    for name in DAY_FILES:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `nids data download` first.")
        log.info("Loading %s", name)
        day = pd.read_csv(
            path,
            usecols=FEATURE_NAMES_84 + [LABEL_COLUMN, "Attempted Category", "Timestamp", "Src IP", "Dst IP"],
            dtype=dtypes,
            parse_dates=["Timestamp"],
        )
        day["Day"] = name.removesuffix(".csv")
        frames.append(day)
    df = pd.concat(frames, ignore_index=True)
    for col in ("Day", "Src IP", "Dst IP"):
        df[col] = df[col].astype("category")
    return df


def clean(
    df: pd.DataFrame, attempted: str = "benign", merge_web_attacks: bool = True, min_class_size: int = 10
) -> tuple[pd.DataFrame, dict]:
    """Relabel, de-duplicate and tidy the raw flows.

    attempted: what to do with "Attempted" flows (attack traffic that never did
      anything malicious). "benign" (the dataset authors' recommendation),
      "attack" (merge into the attack label, as the original project did) or "drop".
    merge_web_attacks: combine Brute Force, XSS and SQL Injection into "Web Attack".
      Once attempted flows are removed only ~100 web-attack flows remain, and
      flow statistics cannot tell these three apart (they are all HTTP requests
      to the same server); the payload is what differs.
    min_class_size: classes with fewer unique flows than this are dropped, because
      they cannot be split into train/val/test and still be measured.
    """
    stats: dict = {"raw_rows": len(df)}

    is_attempted = df["Attempted Category"] != -1
    stats["attempted_rows"] = int(is_attempted.sum())
    if attempted == "benign":
        df.loc[is_attempted, LABEL_COLUMN] = BENIGN
    elif attempted == "attack":
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.replace(" - Attempted", "", regex=False)
    elif attempted == "drop":
        df = df[~is_attempted]
    else:
        raise ValueError(f"attempted must be benign, attack or drop, not {attempted!r}")
    # Kept as a flag so results can be broken down by it later.
    df = df.assign(Attempted=is_attempted.reindex(df.index)).drop(columns=["Attempted Category"])
    if merge_web_attacks:
        is_web = df[LABEL_COLUMN].str.startswith("Web Attack")
        df.loc[is_web, LABEL_COLUMN] = "Web Attack"

    # "Protocol 0" rows are ARP frames that CICFlowMeter misread as IP. They are not
    # flows, and the live engine never produces them.
    not_ip = df["Protocol"] == 0
    stats["non_ip_rows_removed"] = int(not_ip.sum())
    df = df[~not_ip].copy()

    # Header lengths are occasionally negative in CICFlowMeter output (an overflow bug).
    for col in ("Fwd Header Length", "Bwd Header Length"):
        df[col] = df[col].abs()
    # Infinities come from divide-by-zero rates; NaN is imputed later, inside the model.
    feats = df[FEATURE_NAMES_84]
    df[FEATURE_NAMES_84] = feats.where(np.isfinite(feats), np.nan)

    # Exact duplicates inflate scores: the same row ends up in train and test.
    # "Duplicate" means same features and label; the earliest occurrence is kept.
    before = len(df)
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp", kind="stable")
    df = df.drop_duplicates(subset=FEATURE_NAMES_84 + [LABEL_COLUMN])
    stats["duplicate_rows_removed"] = before - len(df)

    # Identical features with different labels cannot be learned; drop them all.
    conflicted = df.duplicated(subset=FEATURE_NAMES_84, keep=False)
    stats["conflicting_rows_removed"] = int(conflicted.sum())
    df = df[~conflicted]

    counts = df[LABEL_COLUMN].value_counts()
    too_small = counts[counts < min_class_size]
    stats["classes_dropped_too_small"] = {k: int(v) for k, v in too_small.items()}
    df = df[~df[LABEL_COLUMN].isin(too_small.index)]

    stats["clean_rows"] = len(df)
    stats["class_counts"] = {k: int(v) for k, v in df[LABEL_COLUMN].value_counts().items()}
    return df.reset_index(drop=True), stats


def split(
    df: pd.DataFrame, how: str = "time", val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> dict:
    """Train / validation / test split.

    how="time" (default): within each day, each class is split in time order.
      The model trains on the earliest 70% of an attack and is tested on the
      last 15%, with the validation flows in between. Neighbouring flows of one
      attack burst are near-copies of each other, and this keeps them on the
      same side of the split.
    how="random": stratified random split. Easier, because every test flow has
      near-identical neighbours in training. Kept for comparison.
    """
    if how == "random":
        train, rest = train_test_split(
            df, test_size=val_size + test_size, stratify=df[LABEL_COLUMN], random_state=seed
        )
        val, test = train_test_split(
            rest, test_size=test_size / (val_size + test_size), stratify=rest[LABEL_COLUMN], random_state=seed
        )
        return {"train": train, "val": val, "test": test}
    if how != "time":
        raise ValueError(f"how must be 'time' or 'random', not {how!r}")

    ordered = df.sort_values("Timestamp", kind="stable")
    groups = ordered.groupby(["Day", LABEL_COLUMN], observed=True, sort=False)
    position = groups.cumcount()
    size = groups[LABEL_COLUMN].transform("size")
    # Whole flows per part, at least one in validation and test when the class has 3+ flows that day.
    n_test = np.maximum(np.floor(size * test_size), (size >= 3).astype(int)).astype(int)
    n_val = np.maximum(np.floor(size * val_size), (size >= 3).astype(int)).astype(int)
    part = np.where(position >= size - n_test, "test", np.where(position >= size - n_test - n_val, "val", "train"))
    return {name: ordered[part == name] for name in SPLITS}


def build(
    raw_dir: Path = paths.RAW_DIR,
    out_dir: Path = paths.PROCESSED_DIR,
    attempted: str = "benign",
    merge_web_attacks: bool = True,
    how: str = "time",
    seed: int = 42,
) -> dict:
    """Full pipeline: load raw CSVs, clean, split, and save Parquet files plus a summary."""
    df, stats = clean(load_raw(raw_dir), attempted=attempted, merge_web_attacks=merge_web_attacks)
    parts = split(df, how=how, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats["split_class_counts"] = {}
    for name, part in parts.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        stats[f"{name}_rows"] = len(part)
        stats["split_class_counts"][name] = {k: int(v) for k, v in part[LABEL_COLUMN].value_counts().items()}
    stats.update({"attempted_policy": attempted, "merge_web_attacks": merge_web_attacks, "split": how, "seed": seed})
    (out_dir / "dataset_summary.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_split(name: str, processed_dir: Path = paths.PROCESSED_DIR) -> pd.DataFrame:
    """Load one saved split ("train", "val" or "test")."""
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `nids data build` first.")
    return pd.read_parquet(path)
