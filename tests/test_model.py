import numpy as np

from nids.features import FEATURE_NAMES_84, LABEL_COLUMN, TRAINING_FEATURES
from nids.ml.model import NIDSModel
from tests.conftest import CLASSES


def test_predicts_known_classes(tiny_model, flows):
    preds = tiny_model.predict(flows[FEATURE_NAMES_84])
    assert set(preds) <= set(CLASSES)
    assert (preds == flows[LABEL_COLUMN].to_numpy()).mean() > 0.95


def test_ensemble_is_weighted_average(tiny_model, flows):
    tiny_model.rf_weight = 0.25
    ens, rf_p, xgb_p = tiny_model.predict_proba_all(flows[FEATURE_NAMES_84].head(5))
    np.testing.assert_allclose(ens, 0.25 * rf_p + 0.75 * xgb_p)
    np.testing.assert_allclose(ens.sum(axis=1), 1.0, rtol=1e-5)
    tiny_model.rf_weight = 0.5


def test_handles_inf_and_nan(tiny_model, flows):
    X = flows[FEATURE_NAMES_84].head(3).copy()
    X.iloc[0, 5] = np.inf
    X.iloc[1, 6] = np.nan
    assert len(tiny_model.predict(X)) == 3


def test_excluded_features_are_never_used(tiny_model):
    assert "Src Port" not in tiny_model.selected_features
    assert set(tiny_model.selected_features) <= set(TRAINING_FEATURES)


def test_explain_returns_top_k(tiny_model, flows):
    row = flows[FEATURE_NAMES_84].iloc[[0]]
    top = tiny_model.explain(row, tiny_model.predict(row)[0], top_k=3)
    assert len(top) == 3
    assert {"feature", "value", "contribution"} == set(top[0])
    assert abs(top[0]["contribution"]) >= abs(top[-1]["contribution"])


def test_save_load_round_trip(tiny_model, flows, tmp_path):
    tiny_model.save(tmp_path)
    loaded = NIDSModel.load(tmp_path)
    X = flows[FEATURE_NAMES_84].head(20)
    np.testing.assert_allclose(loaded.predict_proba(X), tiny_model.predict_proba(X))
    assert (tmp_path / "metadata.json").exists()
