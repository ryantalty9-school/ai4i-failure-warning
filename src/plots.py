"""Small matplotlib charts saved as experiment artifacts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve  # noqa: E402

BLUE, GRAY, RED = "#2f6db5", "#8a94a6", "#b83232"


def confusion(y, proba, threshold, out_path, title):
    cm = confusion_matrix(y, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=150)
    ax.imshow(cm, cmap="Blues")
    labels = [["True negative", "False alarm"], ["Missed failure", "Caught failure"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1f2933")
    ax.set_xticks([0, 1], ["Predicted OK", "Predicted failure"], fontsize=8)
    ax.set_yticks([0, 1], ["Actual OK", "Actual failure"], fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def pr_curve(y, proba, threshold, out_path, title):
    p, r, t = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=150)
    ax.plot(r, p, color=BLUE, lw=1.6)
    idx = int(np.argmin(np.abs(t - threshold))) if len(t) else 0
    ax.scatter([r[idx]], [p[idx]], color=RED, zorder=3, label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Recall (share of failures caught)", fontsize=8)
    ax.set_ylabel("Precision (share of alerts that are real)", fontsize=8)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def importance(series, out_path, title, top=12):
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=150)
    ax.barh(s.index, s.values, color=BLUE)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
