"""Figures for the evaluation report (matplotlib, written as PNG files).

One visual language for all of them: a single blue for the model, grey for
reference lines, hairline grids, labels placed on the marks they describe.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        log.warning("matplotlib not installed (the 'viz' extra); skipping figures")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 14,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })
    return plt


def confusion_matrix(cm: np.ndarray, classes: list[str], path: Path) -> None:
    """cm: counts, rows = true class. Drawn as the share of each true class."""
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.colors import LinearSegmentedColormap

    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    ax.imshow(share, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=1, aspect="auto")
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                value = share[i, j]
                text = f"{value:.2f}" if value >= 0.005 else "<.01"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="#ffffff" if value > 0.55 else (INK if value >= 0.005 else MUTED))
    labels = [f"{c}  (n={int(t):,})" for c, t in zip(classes, cm.sum(axis=1), strict=True)]
    ax.set_xticks(range(n), classes, rotation=40, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Where each true class ends up (share of its test flows)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def detection_tradeoff(fpr: np.ndarray, detection: np.ndarray, points: list[dict], path: Path) -> None:
    """Detection rate against false positive rate (log scale), with the named operating points marked."""
    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    keep = fpr >= 1e-5
    ax.plot(fpr[keep], 100 * detection[keep], color=BLUE, linewidth=2, solid_capstyle="round")
    marked = sorted((p for p in points if p["false_positive_rate"] > 0), key=lambda p: p["false_positive_rate"])
    lowest = min([100 * detection[keep].min()] + [100 * p["detection_rate"] for p in marked]) if keep.any() else 99.0
    span = max(100 - lowest, 0.2)
    key = []
    for i, p in enumerate(marked):
        x, y = max(p["false_positive_rate"], 1e-5), 100 * p["detection_rate"]
        ax.plot([x], [y], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        # Numbered, alternating above and below the line, with the detail in a key:
        # the points often sit too close together to label in place.
        ax.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(0, 9 if i % 2 else -17), ha="center",
                    fontsize=9, fontweight="bold", color=INK)
        key.append(f"{i + 1}  {p['short']}: {y:.2f}% caught, {100 * p['false_positive_rate']:.2g}% false alarms")
    ax.text(0.55, 0.05, "\n".join(key), transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            color=INK_2, linespacing=1.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(lowest - 0.45 * span, 100 + 0.2 * span)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.set_xlabel("Benign flows wrongly flagged (false positive rate, log scale)")
    ax.set_ylabel("Attack flows caught")
    ax.set_title("Detection against false alarms as the alert threshold moves")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration(bins: list[dict], path: Path) -> None:
    """Per confidence band: what the model claimed against how often it was right."""
    plt = _pyplot()
    if plt is None:
        return
    used = [b for b in bins if b["flows"] >= 30]  # a band with a handful of flows is noise
    x = np.arange(len(used))
    claimed = [100 * b["mean_confidence"] for b in used]
    right = [100 * b["accuracy"] for b in used]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, claimed, linestyle="none", marker="_", markersize=26, markeredgewidth=2.5, color=MUTED,
            label="Confidence the model stated")
    ax.plot(x, right, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Share that were right")
    for xi, yi in zip(x, right, strict=True):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5,
                    color=INK_2)
    ax.set_xticks(x, [f"{100 * b['from']:g}-{100 * b['to']:g}%\n{b['flows']:,} flows" for b in used])
    ax.set_xlim(-0.6, len(used) - 0.4)
    ax.set_ylim(min(min(claimed), min(right)) - 8, 108)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Confidence band")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    ax.set_title("Is the confidence score honest?")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
