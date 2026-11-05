"""The deployed model: preprocessing + Random Forest + XGBoost soft-voting ensemble + SHAP.

Training, evaluation and the API all use this one class, so the numbers in the
evaluation report come from exactly the model the API serves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from nids import __version__, paths
from nids.features import BENIGN, FEATURE_NAMES_84, TRAINING_FEATURES


@dataclass
class Prediction:
    label: str
    confidence: float
    is_attack: bool
    models_agree: bool
    probabilities: dict[str, float]


@dataclass
class NIDSModel:
    preprocess: Pipeline
    rf: RandomForestClassifier
    xgb: XGBClassifier
    label_encoder: LabelEncoder
    rf_weight: float = 0.5
    metadata: dict = field(default_factory=dict)
    _explainer: object = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ build

    @staticmethod
    def make_preprocess(n_features: int) -> Pipeline:
        return Pipeline(
            [
                ("select_columns", _ColumnSelector(TRAINING_FEATURES)),
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("drop_constant", VarianceThreshold(0.0)),
                ("select_k_best", SelectKBest(f_classif, k=min(n_features, len(TRAINING_FEATURES)))),
            ]
        )

    @property
    def classes(self) -> list[str]:
        return list(self.label_encoder.classes_)

    @property
    def selected_features(self) -> list[str]:
        steps = self.preprocess.named_steps
        kept = np.array(TRAINING_FEATURES)[steps["drop_constant"].get_support()]
        return kept[steps["select_k_best"].get_support()].tolist()

    # ---------------------------------------------------------------- predict

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocess.transform(_as_frame(X))

    def predict_proba_all(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ensemble, random forest, xgboost) class probabilities."""
        Xt = self.transform(X)
        # Spinning up a thread pool costs more than predicting a handful of flows.
        self.rf.n_jobs = 1 if len(Xt) < 256 else -1
        rf_p = self.rf.predict_proba(Xt)
        xgb_p = self.xgb.predict_proba(Xt)
        return self.rf_weight * rf_p + (1 - self.rf_weight) * xgb_p, rf_p, xgb_p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba_all(X)[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.label_encoder.inverse_transform(self.predict_proba(X).argmax(axis=1))

    def predict_flows(self, X: pd.DataFrame) -> list[Prediction]:
        ens, rf_p, xgb_p = self.predict_proba_all(X)
        idx = ens.argmax(axis=1)
        labels = self.label_encoder.inverse_transform(idx)
        agree = rf_p.argmax(axis=1) == xgb_p.argmax(axis=1)
        return [
            Prediction(
                label=str(labels[i]),
                confidence=float(ens[i, idx[i]]),
                is_attack=labels[i] != BENIGN,
                models_agree=bool(agree[i]),
                probabilities={c: round(float(p), 6) for c, p in zip(self.classes, ens[i], strict=True)},
            )
            for i in range(len(idx))
        ]

    # ---------------------------------------------------------------- explain

    def explain(self, X: pd.DataFrame, label: str, top_k: int = 5) -> list[dict]:
        """Top-k SHAP contributions (from the XGBoost half of the ensemble) for one flow."""
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.xgb)
        Xt = self.transform(X.iloc[[0]] if isinstance(X, pd.DataFrame) else X)
        values = np.asarray(self._explainer.shap_values(Xt))
        class_idx = self.classes.index(label)
        # shap returns (n_samples, n_features, n_classes) for multi-class XGBoost
        contrib = values[0, :, class_idx] if values.ndim == 3 else values[class_idx][0]
        order = np.argsort(-np.abs(contrib))[:top_k]
        names = self.selected_features
        return [
            {"feature": names[i], "value": float(Xt[0, i]), "contribution": float(contrib[i])} for i in order
        ]

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path = paths.MODELS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("nids_version", __version__)
        self.metadata["classes"] = self.classes
        self.metadata["selected_features"] = self.selected_features
        explainer, self._explainer = self._explainer, None
        path = directory / paths.MODEL_FILE
        joblib.dump(self, path, compress=3)
        self._explainer = explainer
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str))
        return path

    @classmethod
    def load(cls, directory: Path = paths.MODELS_DIR) -> NIDSModel:
        path = Path(directory) / paths.MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `nids train` (or `nids model download`) first.")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain an NIDSModel")
        return model


class _ColumnSelector(TransformerMixin, BaseEstimator):
    """Pick and order the training columns from an 84-feature frame."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.array(X[self.columns], dtype=np.float32)  # always a fresh, writable copy
        arr[~np.isfinite(arr)] = np.nan  # rates divided by zero; imputed next
        return arr


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.atleast_2d(X), columns=FEATURE_NAMES_84)
