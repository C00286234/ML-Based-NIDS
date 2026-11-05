"""Shared fixtures. Tests use a tiny model trained on synthetic flows, so they need no dataset."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from nids.features import FEATURE_NAMES_84, LABEL_COLUMN
from nids.ml.model import NIDSModel

CLASSES = ["BENIGN", "DDoS", "Portscan"]


def synthetic_flows(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    """Three well-separated classes: each shifts a different block of features."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, label in enumerate(CLASSES):
        X = rng.normal(0, 1, size=(n_per_class, len(FEATURE_NAMES_84)))
        X[:, 3 + i * 10 : 13 + i * 10] += 6
        df = pd.DataFrame(X.astype("float32"), columns=FEATURE_NAMES_84)
        df[LABEL_COLUMN] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def flows() -> pd.DataFrame:
    return synthetic_flows()


@pytest.fixture(scope="session")
def tiny_model(flows) -> NIDSModel:
    enc = LabelEncoder().fit(flows[LABEL_COLUMN])
    y = enc.transform(flows[LABEL_COLUMN])
    pre = NIDSModel.make_preprocess(n_features=20)
    X = pre.fit_transform(flows[FEATURE_NAMES_84], y)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    xgb = XGBClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    return NIDSModel(preprocess=pre, rf=rf, xgb=xgb, label_encoder=enc, metadata={"test": True})
