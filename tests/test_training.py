"""Training options, the evaluation maths and the experiment report."""

import json

import numpy as np
import pandas as pd
import pytest

from nids.features import (
    ATTACK_FAMILY,
    ATTACK_SEVERITY,
    BENIGN,
    FEATURE_NAMES_84,
    LABEL_COLUMN,
    TRAINING_FEATURES,
)
from nids.ml import dataset, evaluate, experiments, plots
from nids.ml.model import NIDSModel
from nids.ml.train import TrainConfig, fit, prepare

from .conftest import synthetic_flows

QUICK = TrainConfig(n_features=20, rf_trees=10, xgb_rounds=10, xgb_max_depth=3, n_jobs=1)


@pytest.fixture(scope="module")
def frames():
    """Imbalanced train and validation frames: 120 benign, 60 DDoS, 12 port scans."""
    def cut(seed):
        df = synthetic_flows(120, seed)
        return pd.concat([df[df[LABEL_COLUMN] == BENIGN], df[df[LABEL_COLUMN] == "DDoS"].head(60),
                          df[df[LABEL_COLUMN] == "Portscan"].head(12)], ignore_index=True)
    return cut(1), cut(2)


# ----------------------------------------------------------------- training


def test_smote_grows_rare_classes_only_in_training(frames):
    train, val = frames
    data = prepare(QUICK, train, val)
    counts = np.bincount(data.y_train)
    assert counts.tolist() == [120, 120, 120]  # 10x cap would be 600 and 120; both stop at the majority size
    assert len(data.y_val) == len(val) and data.weight is None
    assert data.train_rows == len(train)


def test_class_weights_leave_rows_alone_and_are_capped(frames):
    train, val = frames
    from dataclasses import replace
    data = prepare(replace(QUICK, imbalance="weights", max_class_weight=5), train, val)
    assert len(data.y_train) == len(train)
    assert sorted(set(data.weight.round(3))) == [1.0, 2.0, 5.0]
    plain = prepare(replace(QUICK, imbalance="none"), train, val)
    assert plain.weight is None and len(plain.y_train) == len(train)
    with pytest.raises(ValueError, match="imbalance"):
        prepare(replace(QUICK, imbalance="magic"), train, val)


def test_feature_options(frames):
    train, val = frames
    from dataclasses import replace
    assert QUICK.columns == TRAINING_FEATURES
    everything = prepare(replace(QUICK, n_features=None, exclude_features=("Src Port",)), train, val)
    assert everything.X_train.shape[1] == 83
    assert "Dst Port" not in QUICK.columns and "Src Port" not in QUICK.columns
    with_port = replace(QUICK, exclude_features=experiments.STARTING_EXCLUSIONS)
    assert "Dst Port" in with_port.columns and len(with_port.columns) == 77


def test_fit_returns_a_working_model_without_touching_disk(frames, tmp_path, monkeypatch):
    from nids import paths
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    train, val = frames
    model = fit(QUICK, train, val)
    assert not list(tmp_path.iterdir())
    assert (model.predict(val[FEATURE_NAMES_84]) == val[LABEL_COLUMN].to_numpy()).mean() > 0.95
    assert model.rf_weight in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert model.metadata["config"]["imbalance"] == "smote"
    assert len(model.selected_features) == 20


def test_model_can_use_every_feature():
    df = synthetic_flows(30)
    pre = NIDSModel.make_preprocess(None, ["Dst Port", "Flow Duration", "Protocol"])
    assert pre.fit_transform(df[FEATURE_NAMES_84]).shape == (90, 3)


def test_attack_score_is_one_minus_benign_probability(tiny_model, flows):
    for p in tiny_model.predict_flows(flows[FEATURE_NAMES_84].iloc[[0, 70, 150]]):
        assert p.attack_score == pytest.approx(1 - p.probabilities[BENIGN], abs=1e-5)
        assert p.is_attack == (p.label != BENIGN)


# --------------------------------------------------------------- evaluation

Y = np.array([BENIGN] * 1000 + ["DDoS"] * 100 + ["Portscan"] * 40 + ["Heartbleed"] * 2)


def _noisy(y, wrong, seed=0):
    rng = np.random.default_rng(seed)
    out = y.copy()
    flip = rng.choice(len(y), wrong, replace=False)
    out[flip] = np.where(y[flip] == BENIGN, "DDoS", BENIGN)
    return out


def test_scores_definitions():
    pred = Y.copy()
    pred[:10] = "DDoS"  # 10 false alarms
    pred[1000:1005] = BENIGN  # 5 missed attacks
    pred[1100:1102] = "DDoS"  # 2 scans called DDoS: wrong class, but still detected
    s = evaluate._scores(Y, pred)
    assert s["false_positive_rate"] == pytest.approx(10 / 1000)
    assert s["detection_rate"] == pytest.approx(137 / 142)
    assert s["accuracy"] == pytest.approx(1 - 17 / 1142)
    assert s["macro_f1_reliable"] > s["macro_f1"] * 0.9  # Heartbleed (2 flows) is outside the "reliable" average


def test_confidence_intervals_contain_the_point_estimate():
    pred = _noisy(Y, 40)
    point = evaluate._scores(Y, pred)
    ci = evaluate._confidence_intervals(Y, pred, [BENIGN, "DDoS", "Heartbleed", "Portscan"])
    for name, (low, high) in ci.items():
        assert low <= point[name] <= high, name
        assert high - low < 0.2


def test_operating_points_use_validation_for_the_threshold():
    rng = np.random.default_rng(0)
    benign_val = np.ones(10_000, dtype=bool)
    score_val = rng.uniform(0, 0.2, 10_000)
    y = np.array([BENIGN] * 10_000 + ["DDoS"] * 100)
    score_test = np.concatenate([rng.uniform(0, 0.2, 10_000), np.linspace(0.1, 1.0, 100)])
    points = evaluate._operating_points(score_val, benign_val, score_test, y, np.where(score_test > 0.5, "DDoS", BENIGN))
    assert points[0]["threshold"] is None and points[0]["false_positive_rate"] == 0
    budget_1pct = points[1]
    assert budget_1pct["threshold"] == pytest.approx(0.198, abs=0.002)
    assert budget_1pct["false_positive_rate"] == pytest.approx(0.01, abs=0.004)
    assert budget_1pct["detection_rate"] > points[0]["detection_rate"]
    assert budget_1pct["recall_by_class"] == {"DDoS": budget_1pct["detection_rate"]}
    assert points[3]["threshold"] > points[1]["threshold"]  # a tighter budget needs a higher threshold


def test_calibration_of_a_perfectly_honest_model():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.5, 1.0, 50_000)
    cal = evaluate._calibration(confidence, rng.uniform(size=50_000) < confidence)
    assert cal["expected_calibration_error"] < 0.01
    assert sum(b["flows"] for b in cal["bins"]) == 50_000
    overconfident = evaluate._calibration(np.full(1000, 0.95), np.arange(1000) < 600)
    assert overconfident["expected_calibration_error"] == pytest.approx(0.35)


def test_largest_confusions_are_off_diagonal_and_sorted():
    cm = np.array([[90, 10, 0], [3, 50, 0], [0, 7, 20]])
    rows = evaluate._confusions(cm, ["A", "B", "C"])
    assert [(r["true"], r["predicted"], r["flows"]) for r in rows] == [("A", "B", 10), ("C", "B", 7), ("B", "A", 3)]
    assert rows[0]["share_of_true_class"] == pytest.approx(0.1)


def test_every_class_has_a_family():
    assert set(ATTACK_FAMILY) == set(ATTACK_SEVERITY)
    assert ATTACK_FAMILY["Portscan"] == ATTACK_FAMILY["Infiltration - Portscan"]


def test_full_evaluation_writes_report_and_figures(frames, tmp_path, monkeypatch):
    train, val = frames
    test = synthetic_flows(60, seed=3)
    monkeypatch.setattr(evaluate, "load_split", lambda name: {"val": val, "test": test}[name])
    monkeypatch.setattr(evaluate, "BOOTSTRAP_ROUNDS", 20)
    results = evaluate.evaluate(fit(QUICK, train, val), out_dir=tmp_path)

    assert results["test_rows"] == 180
    assert len(results["operating_points"]) == 4
    assert set(results["confidence_intervals_95"]) == {"accuracy", "macro_f1_reliable", "detection_rate", "false_positive_rate"}
    assert json.loads((tmp_path / "metrics.json").read_text())["by_family"]["accuracy"] >= results["models"][next(iter(results["models"]))]["accuracy"]
    report = (tmp_path / "REPORT.md").read_text()
    for heading in ("Per-class results", "Where it goes wrong", "Choosing an alert threshold", "Is the confidence score honest?"):
        assert heading in report
    pytest.importorskip("matplotlib")
    for figure in ("confusion_matrix.png", "detection_tradeoff.png", "calibration.png"):
        assert (tmp_path / figure).stat().st_size > 5000


def test_figures_are_skipped_without_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "_pyplot", lambda: None)
    plots.calibration([], tmp_path / "c.png")
    assert not (tmp_path / "c.png").exists()


# -------------------------------------------------------------- experiments


def test_experiments_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    frames = {}
    for seed, name in enumerate(dataset.SPLITS):
        df = synthetic_flows(80, seed)
        df["Timestamp"] = pd.Timestamp("2017-07-03") + pd.to_timedelta(range(len(df)), unit="s")
        df["Day"] = "monday"
        frames[name] = df
    monkeypatch.setattr(dataset, "load_split", lambda name: frames[name])
    monkeypatch.setattr(experiments, "OUT_JSON", tmp_path / "experiments.json")
    monkeypatch.setattr(experiments, "OUT_MD", tmp_path / "EXPERIMENTS.md")
    monkeypatch.setattr(experiments, "TrainConfig", lambda: QUICK)
    monkeypatch.setattr(experiments, "UNSEEN_CONFIG", QUICK)
    from dataclasses import replace
    monkeypatch.setattr(experiments, "STARTING_CONFIG", replace(QUICK, exclude_features=experiments.STARTING_EXCLUSIONS))
    monkeypatch.setattr(experiments, "ABLATIONS", {"Starting point": {}, "No class balancing": {"imbalance": "none"}})
    monkeypatch.setattr(experiments.IsolationForest, "__init__", _small_forest(experiments.IsolationForest.__init__))
    monkeypatch.setattr(pd.DataFrame, "sample", _capped_sample(pd.DataFrame.sample))

    results = experiments.run()
    assert set(results) == {"splits", "ablation", "port_shift", "unseen"}
    assert set(results["splits"]) == {"time", "random"}
    assert results["ablation"]["Starting point"]["features_used"] == 20
    shift = results["port_shift"]
    assert shift["flows_moved"] == 240 and set(shift["flagged_by_class"]) == {"DDoS", "Portscan"}
    assert set(results["unseen"]["classes"]) == {"DDoS", "Portscan"}
    for row in results["unseen"]["classes"].values():
        assert row["either_at_0.01_fpr"] >= max(row["flagged_at_0.01_fpr"], row["anomaly_at_0.01_fpr"])
    text = (tmp_path / "EXPERIMENTS.md").read_text()
    for heading in ("Random split vs time-ordered split", "Ablation", "What the destination port was doing",
                    "Attacks the model has never seen"):
        assert heading in text
    assert experiments.run(only="ablation")["splits"] == results["splits"]  # earlier sections are kept


def _small_forest(init):
    def patched(self, **kwargs):
        init(self, **(kwargs | {"n_estimators": 10, "max_samples": 64, "n_jobs": 1}))
    return patched


def _capped_sample(sample):
    def patched(self, n=None, **kwargs):
        return sample(self, n=None if n is None else min(n, len(self)), **kwargs)
    return patched
