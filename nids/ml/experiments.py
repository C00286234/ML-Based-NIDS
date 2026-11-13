"""Experiments that justify (or question) the design: run with `nids experiments`.

1. splits   - the same model under a random split and a time-ordered split.
2. ablation - does SMOTE help? feature selection? the ensemble? Scored on the
              validation split, because this is where the design is chosen; the
              test split is kept for the final model only.
3. ports    - what happens to a model that knows the destination port when
              services run on other ports.
4. unseen   - hold each attack class out of training in turn. How much of an
              attack the model has never seen does it still flag?

Results go to reports/experiments.json and reports/EXPERIMENTS.md.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import log_loss

from nids import paths
from nids.features import BENIGN, BULK_FEATURES, FEATURE_NAMES_84, LABEL_COLUMN
from nids.ml import dataset
from nids.ml.evaluate import _scores
from nids.ml.model import NIDSModel
from nids.ml.train import TrainConfig, _safe, fit, fit_xgb, prepare

log = logging.getLogger(__name__)

OUT_JSON = paths.REPORTS_DIR / "experiments.json"
OUT_MD = paths.REPORTS_DIR / "EXPERIMENTS.md"

# Where the design started: everything except Src Port and the bulk features, Dst Port included.
# The ablations are changes to this, so the table stays the same whatever the defaults become.
STARTING_EXCLUSIONS = ("Src Port", *BULK_FEATURES)
STARTING_CONFIG = TrainConfig(exclude_features=STARTING_EXCLUSIONS)
FINAL_VARIANT = "Without Dst Port"  # the row that became the default TrainConfig
ABLATIONS: dict[str, dict] = {
    "Starting point: SMOTE, top 50 of 77 features": {},
    "No class balancing": {"imbalance": "none"},
    "Class weights instead of SMOTE": {"imbalance": "weights"},
    "Top 30 features": {"n_features": 30},
    "All 77 features": {"n_features": None},
    "All 83 features (bulk features added)": {"n_features": None, "exclude_features": ("Src Port",)},
    FINAL_VARIANT: {"exclude_features": (*STARTING_EXCLUSIONS, "Dst Port")},
}
# Smaller, XGBoost-only model for the 13 hold-one-out runs.
UNSEEN_CONFIG = TrainConfig(imbalance="none", benign_fraction=0.25, xgb_rounds=200, xgb_learning_rate=0.2,
                            xgb_max_depth=8)
WELL_KNOWN_PORTS = 1024
FPR_TARGETS = (0.001, 0.01)


def _all_scores(model: NIDSModel, df: pd.DataFrame) -> dict:
    """Scores for the ensemble and for each model alone on a labelled frame."""
    y = df[LABEL_COLUMN].to_numpy()
    y_idx = model.label_encoder.transform(y)
    labels = np.arange(len(model.classes))
    out = {}
    for name, p in zip(("ensemble", "random_forest", "xgboost"), model.predict_proba_all(df[FEATURE_NAMES_84]), strict=True):
        s = _scores(y, model.label_encoder.inverse_transform(p.argmax(1)))
        s["log_loss"] = float(log_loss(y_idx, _safe(p), labels=labels))
        out[name] = s
    out["rf_weight"] = model.rf_weight
    return out


def split_comparison(cfg: TrainConfig) -> dict:
    """Train the same configuration on a random split and on the time-ordered split; test each on its own test set."""
    parts = {name: dataset.load_split(name) for name in dataset.SPLITS}
    results = {}
    log.info("Split comparison: time-ordered")
    model = fit(cfg, parts["train"], parts["val"])
    results["time"] = {"validation": _all_scores(model, parts["val"]), "test": _all_scores(model, parts["test"])}
    log.info("Split comparison: random")
    everything = pd.concat(parts.values(), ignore_index=True)
    del parts
    random = dataset.split(everything, how="random", seed=cfg.seed)
    del everything
    model = fit(cfg, random["train"], random["val"])
    results["random"] = {"validation": _all_scores(model, random["val"]), "test": _all_scores(model, random["test"])}
    return results


def ablation(cfg: TrainConfig) -> dict:
    """Validation scores for cfg and for each variant of it in ABLATIONS."""
    train_df, val_df = dataset.load_split("train"), dataset.load_split("val")
    results = {}
    for name, changes in ABLATIONS.items():
        log.info("Ablation: %s", name)
        t = time.time()
        model = fit(replace(cfg, **changes), train_df, val_df)
        results[name] = _all_scores(model, val_df) | {
            "features_used": len(model.selected_features), "seconds": round(time.time() - t)}
    return results


def port_shift(model: NIDSModel | None = None, seed: int = 0) -> dict:
    """Score validation flows as they are, then with every well-known destination port replaced
    by a random high port. Nothing else about the flows changes: this is what the same traffic
    would look like against a service on a non-standard port (SSH on 2222, HTTP on 8080)."""
    val_df = dataset.load_split("val")
    if model is None:
        log.info("Port shift: training the starting configuration")
        model = fit(STARTING_CONFIG, dataset.load_split("train"), val_df)
    y = val_df[LABEL_COLUMN].to_numpy()
    moved = val_df[FEATURE_NAMES_84].copy()
    well_known = moved["Dst Port"] < WELL_KNOWN_PORTS
    moved.loc[well_known, "Dst Port"] = np.random.default_rng(seed).integers(20_000, 60_000, int(well_known.sum()))
    before, after = model.predict(val_df[FEATURE_NAMES_84]), model.predict(moved)
    classes = sorted(set(y) - {BENIGN})
    return {
        "uses_dst_port": "Dst Port" in model.selected_features,
        "flows_moved": int(well_known.sum()),
        "as_recorded": _scores(y, before),
        "ports_moved": _scores(y, after),
        "flagged_by_class": {c: {"as_recorded": float((before[y == c] != BENIGN).mean()),
                                 "ports_moved": float((after[y == c] != BENIGN).mean()),
                                 "validation_flows": int((y == c).sum())} for c in classes},
    }


def _benign_threshold(scores: np.ndarray, fpr: float) -> float:
    """The score that only `fpr` of benign flows exceed."""
    return float(np.quantile(scores, 1 - fpr))


def unseen_attacks(cfg: TrainConfig = UNSEEN_CONFIG) -> dict:
    """Hold out each attack class from training and see whether its test flows are still flagged."""
    train_df, val_df, test_df = (dataset.load_split(name) for name in dataset.SPLITS)
    benign_val = val_df[val_df[LABEL_COLUMN] == BENIGN]
    benign_test = test_df[test_df[LABEL_COLUMN] == BENIGN]

    # An anomaly detector that has only ever seen benign traffic.
    log.info("Unseen attacks: fitting the benign-only anomaly detector")
    benign_train = train_df[train_df[LABEL_COLUMN] == BENIGN].sample(300_000, random_state=cfg.seed)
    pre = NIDSModel.make_preprocess(None, cfg.columns)
    squash = lambda X: np.sign(X) * np.log1p(np.abs(X))  # noqa: E731  (features span 9 orders of magnitude)
    forest = IsolationForest(n_estimators=200, max_samples=8192, n_jobs=cfg.n_jobs, random_state=cfg.seed)
    forest.fit(squash(pre.fit_transform(benign_train[FEATURE_NAMES_84])))
    anomaly = lambda df: -forest.score_samples(squash(pre.transform(df[FEATURE_NAMES_84])))  # noqa: E731
    anomaly_cut = {t: _benign_threshold(anomaly(benign_val), t) for t in FPR_TARGETS}
    anomaly_fpr = {str(t): float((anomaly(benign_test) > cut).mean()) for t, cut in anomaly_cut.items()}

    results = {"anomaly_detector_test_fpr": anomaly_fpr, "classes": {}}
    for held_out in sorted(set(train_df[LABEL_COLUMN]) - {BENIGN}):
        log.info("Unseen attacks: holding out %s", held_out)
        data = prepare(cfg, train_df[train_df[LABEL_COLUMN] != held_out], val_df[val_df[LABEL_COLUMN] != held_out])
        xgb = fit_xgb(cfg, data)
        classes = list(data.encoder.classes_)
        benign_col = classes.index(BENIGN)
        attack_score = lambda df: 1 - xgb.predict_proba(data.preprocess.transform(df[FEATURE_NAMES_84]))[:, benign_col]  # noqa: E731, B023
        cuts = {t: _benign_threshold(attack_score(benign_val), t) for t in FPR_TARGETS}

        unseen = test_df[test_df[LABEL_COLUMN] == held_out]
        proba = xgb.predict_proba(data.preprocess.transform(unseen[FEATURE_NAMES_84]))
        called = pd.Series(np.array(classes)[proba.argmax(1)])
        score, odd = 1 - proba[:, benign_col], anomaly(unseen)
        row = {
            "test_flows": int(len(unseen)),
            "flagged_as_some_attack": float((called != BENIGN).mean()),
            "most_often_called": called.value_counts().index[0],
            "benign_false_positive_rate": float((attack_score(benign_test) > 0.5).mean()),
        }
        for t in FPR_TARGETS:
            row[f"flagged_at_{t:g}_fpr"] = float((score > cuts[t]).mean())
            row[f"anomaly_at_{t:g}_fpr"] = float((odd > anomaly_cut[t]).mean())
            row[f"either_at_{t:g}_fpr"] = float(((score > cuts[t]) | (odd > anomaly_cut[t])).mean())
        results["classes"][held_out] = row
    return results


def run(only: str | None = None) -> dict:
    cfg = TrainConfig()
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}

    def save() -> None:
        paths.REPORTS_DIR.mkdir(exist_ok=True)
        OUT_JSON.write_text(json.dumps(results, indent=2))
        OUT_MD.write_text(markdown(results))

    if only in (None, "splits"):
        results["splits"] = split_comparison(cfg)
        save()
    if only in (None, "ablation"):
        results["ablation"] = ablation(STARTING_CONFIG)
        save()
    if only in (None, "ports"):
        results["port_shift"] = port_shift()
        save()
    if only in (None, "unseen"):
        results["unseen"] = unseen_attacks()
        save()
    return results


# ------------------------------------------------------------------ report


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def markdown(r: dict) -> str:
    lines = ["# Experiments", "", "Generated by `nids experiments`. These back the design choices described in the README.", ""]
    if "splits" in r:
        lines += [
            "## Random split vs time-ordered split",
            "",
            "The same configuration, trained and tested twice. In a random split every test flow has",
            "near-identical neighbours from the same attack burst in the training set. In the time-ordered",
            "split the model trains on the earliest 70% of each class on each day and is tested on the last 15%.",
            "",
            "| Configuration | Split | Accuracy | Macro F1 (classes with 30+ test flows) | Macro F1 (all) | Detection rate | False positive rate |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, config in (("splits_starting_config", "Starting point (with Dst Port)"), ("splits", "Final (without Dst Port)")):
            for name, label in (("random", "Random"), ("time", "Time-ordered")):
                if key in r:
                    s = r[key][name]["test"]["ensemble"]
                    lines.append(f"| {config} | {label} | {_pct(s['accuracy'])} | {s['macro_f1_reliable']:.4f} | {s['macro_f1']:.4f} | "
                                 f"{_pct(s['detection_rate'])} | {_pct(s['false_positive_rate'], 3)} |")
        lines += ["", "The drop under the time-ordered split comes almost entirely from the slow denial-of-service attacks.",
                  "The last minutes of a Slowloris or Slowhttptest attack are its wind-down, when held-open connections",
                  "are finally closed. Those flows look unlike the build-up the model trained on, and many are called",
                  "benign. A random split hides this, because wind-down flows land in the training set too.", ""]
    if "ablation" in r:
        lines += [
            "## Ablation (validation split, time-ordered)",
            "",
            "One change at a time from the baseline. Scored on validation, not test, because this table",
            "is what the final configuration is chosen from.",
            "",
            "| Variant | Features | Model | Macro F1 (30+ flows) | Log-loss | Detection rate | False positive rate |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, v in r["ablation"].items():
            if name == FINAL_VARIANT:
                name = f"**{name} (chosen)**"
            for model, label in (("xgboost", "XGBoost"), ("random_forest", "Random Forest"), ("ensemble", f"Ensemble (RF weight {v['rf_weight']:.2f})")):
                s = v[model]
                first = model == "xgboost"
                lines.append(f"| {name if first else ''} | {v.get('features_used', '') if first else ''} | {label} | "
                             f"{s['macro_f1_reliable']:.4f} | {s['log_loss']:.4f} | {_pct(s['detection_rate'])} | "
                             f"{_pct(s['false_positive_rate'], 3)} |")
        lines.append("")
    if "port_shift" in r:
        ps = r["port_shift"]
        a, b = ps["as_recorded"], ps["ports_moved"]
        lines += [
            "## What the destination port was doing",
            "",
            "The starting configuration includes Dst Port, and the ablation above shows that removing it costs",
            "nothing on validation. This is why it was removed anyway. The validation flows were scored twice",
            f"by the starting configuration: as recorded, and with every destination port below {WELL_KNOWN_PORTS}",
            "replaced by a random high port. Nothing else about the flows changes; it is the same traffic aimed",
            "at a service on a non-standard port.",
            "",
            "| Validation flows | Detection rate | False positive rate | Macro F1 (30+ flows) |",
            "|---|---|---|---|",
            f"| As recorded | {_pct(a['detection_rate'])} | {_pct(a['false_positive_rate'], 3)} | {a['macro_f1_reliable']:.4f} |",
            f"| Well-known ports moved | {_pct(b['detection_rate'])} | {_pct(b['false_positive_rate'], 3)} | {b['macro_f1_reliable']:.4f} |",
            "",
            "| Attack | Validation flows | Flagged, as recorded | Flagged, ports moved |",
            "|---|---|---|---|",
        ]
        for name, c in ps["flagged_by_class"].items():
            lines.append(f"| {name} | {c['validation_flows']:,} | {_pct(c['as_recorded'], 1)} | {_pct(c['ports_moved'], 1)} |")
        lines += ["", "The model had learned \"port 80 plus this shape means DoS Hulk\" rather than the shape alone. A model",
                  "trained without Dst Port gives the same answer whatever the port, so the second row cannot happen to it.", ""]
    if "unseen" in r:
        u = r["unseen"]
        lines += [
            "## Attacks the model has never seen",
            "",
            "Each attack class is removed from training in turn (a smaller XGBoost-only model, to keep 13",
            "runs affordable), then its test flows are scored. Three ways of flagging them:",
            "",
            "- **Classifier:** the model calls the flow some other attack.",
            "- **Attack score:** 1 - P(benign), with the alert threshold set so that 1% of benign validation flows exceed it.",
            "- **Anomaly detector:** an Isolation Forest trained on benign traffic only, at the same 1% budget.",
            "",
            "| Held-out attack | Test flows | Classifier | Most often called | Attack score @ 1% FPR | Anomaly detector @ 1% FPR | Either @ 1% each |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, c in u["classes"].items():
            lines.append(f"| {name} | {c['test_flows']:,} | {_pct(c['flagged_as_some_attack'], 1)} | {c['most_often_called']} | "
                         f"{_pct(c['flagged_at_0.01_fpr'], 1)} | {_pct(c['anomaly_at_0.01_fpr'], 1)} | {_pct(c['either_at_0.01_fpr'], 1)} |")
        lines += [
            "", f"On benign test flows the anomaly detector's false positive rate at the 1% setting is "
            f"{_pct(u['anomaly_detector_test_fpr']['0.01'])}.",
            "",
            "What this shows: the model recognises a new attack only when it resembles one it already knows.",
            "A held-out scan or flood is still caught, because other scans and floods were in training. A",
            "genuinely different attack (a botnet, a brute-force login, an infiltration) passes as benign.",
            "The benign-only anomaly detector does not close the gap: these attacks are built from ordinary",
            "connections, and per-flow statistics give it nothing unusual to see. It is not part of the",
            "deployed model for that reason.", ""]
    return "\n".join(lines)
