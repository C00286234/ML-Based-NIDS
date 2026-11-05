"""`nids` command line: data, training, evaluation, serving and the demo."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from nids import __version__, paths

MODEL_URL = os.getenv(
    "NIDS_MODEL_URL",
    "https://github.com/C00286234/ML-Based-NIDS/releases/latest/download/nids_model.joblib",
)


def _data(args) -> None:
    from nids.ml import dataset

    if args.action == "download":
        print(f"Dataset in {dataset.download()}")
    elif args.action == "build":
        stats = dataset.build(attempted=args.attempted, merge_web_attacks=not args.keep_web_classes)
        print(json.dumps({k: v for k, v in stats.items() if k != "class_counts"}, indent=2))
    elif args.action == "sample":
        from nids.demo import make_sample

        print(f"Wrote {make_sample()}")


def _train(args) -> None:
    from nids.ml.train import TrainConfig, train

    cfg = TrainConfig(benign_fraction=0.05, rf_trees=50, xgb_rounds=100) if args.quick else TrainConfig()
    model = train(cfg)
    print(json.dumps(model.metadata["val_scores_by_rf_weight"], indent=2))
    if not args.no_eval:
        _evaluate(args)


def _evaluate(args) -> None:
    from nids.ml.evaluate import evaluate

    results = evaluate()
    for name, s in results["models"].items():
        print(f"{name:<40} accuracy={s['accuracy']:.4f}  macro-F1={s['macro_f1']:.4f}  "
              f"FPR={s['false_positive_rate']:.4%}")
    print(f"Report: {paths.REPORTS_DIR / 'REPORT.md'}")


def _serve(args) -> None:
    import uvicorn

    uvicorn.run("nids.api.app:app", host=args.host, port=args.port, log_level="info")


def _dashboard(args) -> None:
    from nids.dashboard.app import app

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


def _engine(args) -> None:
    from nids.engine.engine import main

    sys.argv = ["nids engine", *args.engine_args]
    main()


def _demo(args) -> None:
    from nids.demo import run

    run(api_url=args.api, rate=args.rate, attack_ratio=args.attack_ratio, duration=args.duration,
        seed=args.seed, api_key=os.getenv("NIDS_API_KEY"))


def _model(args) -> None:
    import requests

    dest = paths.MODELS_DIR / paths.MODEL_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url}")
    with requests.get(args.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"Saved {dest}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nids", description="ML-based network intrusion detection")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data", help="download the dataset and build train/val/test splits")
    d.add_argument("action", choices=["download", "build", "sample"])
    d.add_argument("--attempted", choices=["benign", "attack", "drop"], default="benign",
                   help="how to label 'Attempted' attack flows (default: benign, as the dataset authors advise)")
    d.add_argument("--keep-web-classes", action="store_true",
                   help="keep Brute Force / XSS / SQL Injection separate instead of one 'Web Attack' class")
    d.set_defaults(func=_data)

    t = sub.add_parser("train", help="train the ensemble (then evaluate on the test split)")
    t.add_argument("--quick", action="store_true", help="small, fast model for trying things out")
    t.add_argument("--no-eval", action="store_true")
    t.set_defaults(func=_train)

    sub.add_parser("evaluate", help="evaluate the saved model on the held-out test split").set_defaults(
        func=_evaluate)

    s = sub.add_parser("serve", help="run the prediction API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    db = sub.add_parser("dashboard", help="run the web dashboard")
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=5001)
    db.set_defaults(func=_dashboard)

    e = sub.add_parser("engine", help="live capture engine (needs root and the 'capture' extra)")
    e.add_argument("engine_args", nargs=argparse.REMAINDER)
    e.set_defaults(func=_engine)

    dm = sub.add_parser("demo", help="replay held-out flows against the API")
    dm.add_argument("--api", default=os.getenv("NIDS_API_URL", "http://localhost:8000"))
    dm.add_argument("--rate", type=float, default=4.0, help="flows per second")
    dm.add_argument("--attack-ratio", type=float, default=0.3)
    dm.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    dm.add_argument("--seed", type=int, default=None)
    dm.set_defaults(func=_demo)

    m = sub.add_parser("model", help="download the pre-trained model from GitHub Releases")
    m.add_argument("action", choices=["download"])
    m.add_argument("--url", default=MODEL_URL)
    m.set_defaults(func=_model)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
