"""
Regenerate the chart-type figures for the showcase site with a consistent
Tokyo-Night palette matching the site theme. Produces:

- site/assets/figures/confusion_matrix.png   (XGBoost, held-out test set)
- site/assets/figures/label_distribution.png (CIC-IDS2017 class counts)
- site/assets/figures/shap_global.png        (Top-20 SHAP features)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

BASE = r"C:\Users\folosiye\FYP-NIDS-ML"
DATA_CSV = os.path.join(BASE, "data", "processed", "cicids2017_cleaned.csv")
MODELS = os.path.join(BASE, "models")
SHAP_CSV = os.path.join(BASE, "results", "shap_feature_importance.csv")
OUT = os.path.join(BASE, "site", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- Palette ---
BG = "#1a1b26"
PANEL = "#1f2335"
FG = "#c0caf5"
DIM = "#a9b1d6"
GRID = "#3b4261"
BLUE = "#7aa2f7"
PURPLE = "#bb9af7"
GREEN = "#9ece6a"
RED = "#f7768e"
AMBER = "#e0af68"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "axes.titlecolor": FG,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
})


def dark_cmap():
    """Blue-purple colormap that reads well on a dark background."""
    return LinearSegmentedColormap.from_list(
        "nids_dark",
        [
            (0.00, "#1f2335"),  # panel
            (0.10, "#2a2e45"),
            (0.35, "#3d59a1"),  # deep blue
            (0.65, "#7aa2f7"),  # accent blue
            (1.00, "#bb9af7"),  # purple
        ],
    )


# ============================================================
# 1. Confusion matrix (XGBoost)
# ============================================================
def build_confusion_matrix():
    print("Loading dataset...")
    df = pd.read_csv(DATA_CSV)
    X = df.drop("Label", axis=1)
    le = joblib.load(os.path.join(MODELS, "label_encoder.pkl"))
    y = le.transform(df["Label"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_temp,
    )

    scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))
    selector = joblib.load(os.path.join(MODELS, "feature_selector.pkl"))
    X_test_p = selector.transform(scaler.transform(X_test))

    print("Running XGBoost for confusion matrix...")
    xgb = joblib.load(os.path.join(MODELS, "xgboost_model.pkl"))
    xgb_pred = xgb.predict(X_test_p)

    cm = confusion_matrix(y_test, xgb_pred, labels=range(len(le.classes_)))
    # Row-normalise so small classes are visible alongside BENIGN's huge count
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_facecolor(PANEL)
    im = ax.imshow(cm_norm, cmap=dark_cmap(), vmin=0, vmax=1, aspect="auto")

    # Annotate with raw counts (readable on dark bg)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val == 0:
                continue
            color = FG if cm_norm[i, j] < 0.55 else "#15161e"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(np.arange(len(le.classes_)))
    ax.set_yticks(np.arange(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=38, ha="right", fontsize=10)
    ax.set_yticklabels(le.classes_, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("XGBoost Confusion Matrix  ·  held-out test set (row-normalised)",
                 fontsize=14, fontweight="bold", pad=18)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.yaxis.set_tick_params(color=FG)
    cbar.ax.set_ylabel("Fraction of true class", color=FG, fontsize=10)
    for t in cbar.ax.get_yticklabels():
        t.set_color(FG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT, "confusion_matrix.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  -> {out}")


# ============================================================
# 2. Label distribution
# ============================================================
def build_label_distribution():
    print("Building label distribution...")
    df = pd.read_csv(DATA_CSV, usecols=["Label"])
    counts = df["Label"].value_counts().sort_values(ascending=True)

    colors = [GREEN if label == "BENIGN" else BLUE for label in counts.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(PANEL)
    bars = ax.barh(counts.index, counts.values, color=colors,
                   edgecolor=BG, linewidth=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Number of flows (log scale)", fontsize=12, fontweight="bold")
    ax.set_title("CIC-IDS2017 Class Distribution  ·  2.1M flows across 16 classes",
                 fontsize=14, fontweight="bold", pad=18)
    ax.grid(axis="x", color=GRID, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right-aligned count labels
    for bar, val in zip(bars, counts.values):
        ax.text(val * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color=DIM)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=GREEN, edgecolor=BG, label="Benign"),
        Patch(facecolor=BLUE,  edgecolor=BG, label="Attack class"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=11)

    plt.tight_layout()
    out = os.path.join(OUT, "label_distribution.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  -> {out}")


# ============================================================
# 3. SHAP global feature importance
# ============================================================
def build_shap_global():
    print("Building SHAP global chart...")
    df = pd.read_csv(SHAP_CSV).sort_values("Mean_SHAP_Value", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor(PANEL)

    # Gradient-coloured bars: lower rank -> blue, higher rank -> purple
    n = len(df)
    colors = [
        (1 - t / max(n - 1, 1)) * np.array([0.48, 0.63, 0.97]) +
        (t / max(n - 1, 1)) * np.array([0.73, 0.60, 0.97])
        for t in range(n)
    ]

    ax.barh(df["Feature"], df["Mean_SHAP_Value"],
            color=colors, edgecolor=BG, linewidth=0.6)
    ax.set_xlabel("Mean |SHAP value| across test set",
                  fontsize=12, fontweight="bold")
    ax.set_title("Global SHAP Feature Importance  ·  top 20 features driving XGBoost predictions",
                 fontsize=14, fontweight="bold", pad=18)
    ax.grid(axis="x", color=GRID, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    # Value labels
    for i, v in enumerate(df["Mean_SHAP_Value"]):
        ax.text(v * 1.01, i, f"{v:.4f}", va="center",
                fontsize=8, color=DIM)

    plt.tight_layout()
    out = os.path.join(OUT, "shap_global.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  -> {out}")


if __name__ == "__main__":
    build_confusion_matrix()
    build_label_distribution()
    build_shap_global()
    print("\nDone.")
