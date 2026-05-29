#!/usr/bin/env python3
"""Render best-split figures for the presentation slide."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# UC Davis colors
AGGIE_BLUE = "#022851"
AGGIE_GOLD = "#FFBF00"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Load data
data = np.load(
    REPO / "crossval_results/repnet_crosslead_deeper_multiseed_pe/2026-05-04_21-06-34/all_probs.npz",
    allow_pickle=True,
)
y_true = data["y_true"]
all_probs = data["probs"]  # (30, 431)

with open(REPO / "export/results/multisplit_dbb6f49/per_split.json") as f:
    splits = json.load(f)

# Best split by AUROC
aurocs = np.array([s["auroc"] for s in splits])
best_idx = int(np.argmax(aurocs))
best_split = splits[best_idx]
probs = all_probs[best_idx]

best_auroc = roc_auc_score(y_true, probs)
best_auprc = average_precision_score(y_true, probs)

print(f"Best split: {best_idx}, AUROC={best_auroc:.4f}, AUPRC={best_auprc:.4f}")

# --- Figure 1: ROC + PR curves side by side ---
fpr, tpr, _ = roc_curve(y_true, probs)
prec, rec, _ = precision_recall_curve(y_true, probs)
prevalence = y_true.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# ROC
ax1.plot(fpr, tpr, color=AGGIE_BLUE, linewidth=2, label=f"AUROC = {best_auroc:.3f}")
ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, alpha=0.6)
ax1.fill_between(fpr, tpr, alpha=0.08, color=AGGIE_BLUE)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve (Best Split)")
ax1.legend(loc="lower right", framealpha=0.9)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_aspect("equal")

# PR
ax2.plot(rec, prec, color="#C99700", linewidth=2, label=f"AUPRC = {best_auprc:.3f}")
ax2.axhline(prevalence, linestyle="--", color="gray", linewidth=0.8, alpha=0.6,
            label=f"Prevalence = {prevalence:.2f}")
ax2.fill_between(rec, prec, alpha=0.08, color="#C99700")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve (Best Split)")
ax2.legend(loc="upper right", framealpha=0.9)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_aspect("equal")

fig.tight_layout()
fig.savefig(OUT / "best_roc_pr.png", dpi=300, facecolor="white")
plt.close(fig)
print(f"  -> best_roc_pr.png")

# --- Figure 2: Prediction distribution (mirror histogram) ---
fpr_j, tpr_j, thr_j = roc_curve(y_true, probs)
tau_youden = float(thr_j[np.argmax(tpr_j - fpr_j)])

bins = np.linspace(0, 1, 41)
centers = 0.5 * (bins[:-1] + bins[1:])
bar_w = bins[1] - bins[0]
h_norm, _ = np.histogram(probs[y_true == 0], bins=bins)
h_pe, _ = np.histogram(probs[y_true == 1], bins=bins)

n_pe = int((y_true == 1).sum())
n_norm = int((y_true == 0).sum())

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(centers, h_pe, width=bar_w, color="#C0392B", alpha=0.85,
       label=f"PE+ (n={n_pe})")
ax.bar(centers, -h_norm, width=bar_w, color=AGGIE_BLUE, alpha=0.75,
       label=f"Normal (n={n_norm})")
ax.axvline(tau_youden, color=AGGIE_GOLD, linewidth=2, linestyle="--",
           label=f"Youden's J = {tau_youden:.2f}")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Predicted P(PE)")
ax.set_ylabel("Count")
ax.set_title("Prediction Distribution (Best Split)")
ax.legend(loc="upper right", framealpha=0.9, fontsize=7)

yticks = ax.get_yticks()
ax.set_yticklabels([str(int(abs(t))) for t in yticks])
fig.tight_layout()
fig.savefig(OUT / "best_pred_dist.png", dpi=300, facecolor="white")
plt.close(fig)
print(f"  -> best_pred_dist.png")
