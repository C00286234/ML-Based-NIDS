"""Train the Random Forest + XGBoost ensemble on the saved training split."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from nids import paths
from nids.features import BENIGN, FEATURE_NAMES_84, LABEL_COLUMN
from nids.ml.dataset import load_split
from nids.ml.model import NIDSModel

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    n_features: int = 50
    smote: bool = True
    smote_max_multiplier: int = 10  # grow a rare class by at most 10x
    smote_cap: int = 100_000  # and never past this many rows
    rf_trees: int = 200
    rf_max_depth: int = 20
    xgb_rounds: int = 400
    xgb_max_depth: int = 10
    xgb_learning_rate: float = 0.1
    early_stopping: int = 15
    benign_fraction: float = 1.0  # < 1.0 subsamples benign rows for quick runs
    seed: int = 42
    n_jobs: int = -1


def _oversample(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    counts = Counter(y)
    majority = max(counts.values())
    targets = {
        c: min(n * cfg.smote_max_multiplier, cfg.smote_cap, majority)
        for c, n in counts.items()
        if min(n * cfg.smote_max_multiplier, cfg.smote_cap, majority) > n
    }
    if not targets:
        return X, y
    k = max(1, min(5, min(counts.values()) - 1))
    log.info("SMOTE: oversampling %d classes (k_neighbors=%d)", len(targets), k)
    return SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=cfg.seed).fit_resample(X, y)


def train(cfg: TrainConfig | None = None) -> NIDSModel:
    cfg = cfg or TrainConfig()
    t0 = time.time()

    train_df, val_df = load_split("train"), load_split("val")
    if cfg.benign_fraction < 1.0:
        benign = train_df[train_df[LABEL_COLUMN] == BENIGN].sample(frac=cfg.benign_fraction, random_state=cfg.seed)
        train_df = pd.concat([train_df[train_df[LABEL_COLUMN] != BENIGN], benign])
    log.info("Training on %d flows, validating on %d", len(train_df), len(val_df))

    encoder = LabelEncoder().fit(train_df[LABEL_COLUMN])
    y_train = encoder.transform(train_df[LABEL_COLUMN])
    y_val = encoder.transform(val_df[LABEL_COLUMN])

    preprocess = NIDSModel.make_preprocess(cfg.n_features)
    X_train = preprocess.fit_transform(train_df[FEATURE_NAMES_84], y_train)
    X_val = preprocess.transform(val_df[FEATURE_NAMES_84])

    # SMOTE only ever touches the training split.
    if cfg.smote:
        X_train, y_train = _oversample(X_train, y_train, cfg)

    log.info("Training Random Forest (%d trees)", cfg.rf_trees)
    t = time.time()
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        max_depth=cfg.rf_max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
    ).fit(X_train, y_train)
    rf_seconds = time.time() - t

    log.info("Training XGBoost (up to %d rounds, early stopping on validation)", cfg.xgb_rounds)
    t = time.time()
    xgb = XGBClassifier(
        n_estimators=cfg.xgb_rounds,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=cfg.early_stopping,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
    ).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_seconds = time.time() - t

    # Pick the ensemble weight on the validation split (never on test). Log-loss is
    # used rather than macro-F1 because macro-F1 can swing on a single flow of a
    # tiny class (Heartbleed has 2 validation flows).
    rf_p, xgb_p = rf.predict_proba(X_val), xgb.predict_proba(X_val)
    labels = np.arange(len(encoder.classes_))
    weight_scores = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = w * rf_p + (1 - w) * xgb_p
        weight_scores[w] = {
            "log_loss": log_loss(y_val, np.clip(p, 1e-7, 1), labels=labels),
            "macro_f1": f1_score(y_val, p.argmax(axis=1), average="macro"),
        }
    rf_weight = min(weight_scores, key=lambda w: weight_scores[w]["log_loss"])
    log.info("Validation scores by RF weight: %s -> using %.2f", weight_scores, rf_weight)

    model = NIDSModel(
        preprocess=preprocess,
        rf=rf,
        xgb=xgb,
        label_encoder=encoder,
        rf_weight=rf_weight,
        metadata={
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": asdict(cfg),
            "train_rows": int(len(train_df)),
            "train_rows_after_smote": int(len(y_train)),
            "xgb_best_iteration": int(xgb.best_iteration),
            "val_scores_by_rf_weight": {
                str(w): {k: round(v, 4) for k, v in s.items()} for w, s in weight_scores.items()
            },
            "rf_seconds": round(rf_seconds, 1),
            "xgb_seconds": round(xgb_seconds, 1),
            "total_seconds": round(time.time() - t0, 1),
        },
    )
    path = model.save(paths.MODELS_DIR)
    log.info("Saved model to %s", path)
    return model
