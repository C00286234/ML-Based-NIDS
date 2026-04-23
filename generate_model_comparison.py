"""
Generate a per-class precision/recall comparison chart between Random Forest
and XGBoost on the held-out test set.

Reproduces the exact stratified train/val/test split from
04_model_training_improved.py (random_state=42), applies the saved scaler and
feature selector, runs both models, and plots two subplots (precision + recall).

Overwrites site/assets/figures/model_comparison.png.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

BASE = r"C:\Users\folosiye\FYP-NIDS-ML"
DATA_CSV = os.path.join(BASE, "data", "processed", "cicids2017_cleaned.csv")
MODELS = os.path.join(BASE, "models")
OUT_PNG = os.path.join(BASE, "site", "assets", "figures", "model_comparison.png")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- Load ---
print("Loading dataset...")
df = pd.read_csv(DATA_CSV)
print(f"  rows: {len(df):,}")

X = df.drop("Label", axis=1)
y_labels = df["Label"]

label_encoder = joblib.load(os.path.join(MODELS, "label_encoder.pkl"))
y = label_encoder.transform(y_labels)

# --- Reproduce the exact split ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=(TEST_SIZE + VAL_SIZE),
    random_state=RANDOM_STATE,
    stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)
print(f"  test set: {len(X_test):,}")

# --- Apply saved preprocessing ---
scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))
selector = joblib.load(os.path.join(MODELS, "feature_selector.pkl"))

X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

# --- Run both models ---
print("Running Random Forest...")
rf = joblib.load(os.path.join(MODELS, "random_forest_model.pkl"))
rf_pred = rf.predict(X_test_selected)

print("Running XGBoost...")
xgb = joblib.load(os.path.join(MODELS, "xgboost_model.pkl"))
xgb_pred = xgb.predict(X_test_selected)

# --- Per-class metrics ---
classes = label_encoder.classes_
rf_prec, rf_rec, _, _ = precision_recall_fscore_support(
    y_test, rf_pred, labels=range(len(classes)), zero_division=0
)
xgb_prec, xgb_rec, _, _ = precision_recall_fscore_support(
    y_test, xgb_pred, labels=range(len(classes)), zero_division=0
)

# --- Plot (Tokyo-Night palette to match the site) ---
BG = "#1a1b26"
FG = "#c0caf5"
GRID = "#3b4261"
RF_COLOR = "#7aa2f7"   # blue
XGB_COLOR = "#bb9af7"  # purple

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "axes.titlecolor": FG,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 10), facecolor=BG, sharex=True
)

x = np.arange(len(classes))
width = 0.38

# Precision
ax1.bar(x - width/2, rf_prec, width, label="Random Forest",
        color=RF_COLOR, edgecolor=BG, linewidth=0.5)
ax1.bar(x + width/2, xgb_prec, width, label="XGBoost",
        color=XGB_COLOR, edgecolor=BG, linewidth=0.5)
ax1.set_facecolor(BG)
ax1.set_ylabel("Precision", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 1.05)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.grid(axis="y", color=GRID, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_title(
    "Per-Class Precision & Recall — Random Forest vs XGBoost (test set)",
    fontsize=14, fontweight="bold", pad=16,
)
ax1.legend(
    loc="lower right", frameon=True, facecolor=BG, edgecolor=GRID,
    labelcolor=FG, fontsize=11,
)

# Recall
ax2.bar(x - width/2, rf_rec, width, label="Random Forest",
        color=RF_COLOR, edgecolor=BG, linewidth=0.5)
ax2.bar(x + width/2, xgb_rec, width, label="XGBoost",
        color=XGB_COLOR, edgecolor=BG, linewidth=0.5)
ax2.set_facecolor(BG)
ax2.set_ylabel("Recall", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.01, 0.2))
ax2.grid(axis="y", color=GRID, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xticks(x)
ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=10)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=160, facecolor=BG, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

# --- Also print a summary so it's easy to cite in the report ---
print("\nPer-class summary (precision / recall):")
print(f"{'Class':<28} {'RF prec':>8} {'RF rec':>8} {'XGB prec':>9} {'XGB rec':>9}")
for i, c in enumerate(classes):
    print(f"{c:<28} {rf_prec[i]:>8.3f} {rf_rec[i]:>8.3f} {xgb_prec[i]:>9.3f} {xgb_rec[i]:>9.3f}")
