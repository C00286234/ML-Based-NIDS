"""Project paths. Override the root with NIDS_HOME or the models folder with NIDS_MODELS_DIR."""

import os
from pathlib import Path

ROOT = Path(os.getenv("NIDS_HOME", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "CICIDS2017_improved"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("NIDS_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = Path(os.getenv("NIDS_CONFIG", ROOT / "config" / "nids_config.yaml"))

MODEL_FILE = "nids_model.joblib"
