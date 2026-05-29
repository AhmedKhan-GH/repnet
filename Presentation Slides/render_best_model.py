#!/usr/bin/env python3
"""Render best model (dbb6f49) figures using StratifiedGroupKFold.

Uses export/code/train.py methodology:
  _stratified=True -> StratifiedGroupKFold(n_splits=5)
  split_seed = split_i * 7 + 1000
  best split = #17
"""

import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/ahmed/PycharmProjects/repnet")
sys.path.insert(0, str(REPO / "export" / "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

import torch
import torch.nn as nn

from prepare import load_ecg_data, preprocess

OUT = Path(__file__).resolve().parent
DATA_DIR = str(REPO / "data" / "seniordesign_upload")
RESULTS_DIR = REPO / "multisplit_dbb6f49"

AGGIE_BLUE = "#022851"
AGGIE_GOLD = "#FFBF00"
AGGIE_GOLD_DARK = "#C99700"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "lines.linewidth": 1.8,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})


class PerLeadCNN(nn.Module):
    def __init__(self, n_leads=12, filters=(16, 32, 48), kernels=(31, 21, 11),
                 dropout=0.15, n_classes=2):
        super().__init__()
        layers = []
        in_ch = 1
        for f, k in zip(filters, kernels):
            layers.extend([
                nn.Conv1d(in_ch, f, k, stride=2, padding=k // 2, bias=False),
                nn.BatchNorm1d(f), nn.Mish(),
            ])
            in_ch = f
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.n_leads = n_leads
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[-1] * n_leads, n_classes)

    def forward(self, x):
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        x = x.reshape(B, L * x.shape[-1])
        return self.fc(self.head_drop(x))


# --- Best split index from per_split.json ---
with open(RESULTS_DIR / "per_split.json") as f:
    splits = json.load(f)
best_idx = int(np.argmax([s["auroc"] for s in splits]))

# --- Load data ---
print("Loading data...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_all, y_all, patient_ids, _ = load_ecg_data(DATA_DIR)
X_all = preprocess(X_all)
X_all = X_all[:, :, ::2]

# --- StratifiedGroupKFold split (same as train.py _stratified=True) ---
split_seed = best_idx * 7 + 1000
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
dev_idx, test_idx = next(iter(sgkf.split(X_all, y_all, groups=patient_ids)))
X_te, y_te = X_all[test_idx], y_all[test_idx]

# --- Load model and predict ---
model = PerLeadCNN(filters=(16, 32, 48), kernels=(31, 21, 11), dropout=0.15)
model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pt",
                                 map_location=device, weights_only=True))
model.to(device).eval()

with torch.no_grad():
    probs = torch.softmax(
        model(torch.tensor(X_te, dtype=torch.float32).to(device)),
        dim=1)[:, 1].cpu().numpy()

auroc = roc_auc_score(y_te, probs)
auprc = average_precision_score(y_te, probs)
print(f"Test: N={len(y_te)}, pos={int(y_te.sum())}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

# --- Youden threshold ---
fpr, tpr, thr = roc_curve(y_te, probs)
j_idx = np.argmax(tpr - fpr)
tau = float(thr[j_idx])
preds = (probs >= tau).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
npv = tn / (tn + fn)
prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * prec_val * sens / (prec_val + sens) if (prec_val + sens) > 0 else 0

print(f"Youden: thresh={tau:.3f} sens={sens:.3f} spec={spec:.3f} NPV={npv:.3f} F1={f1:.3f}")
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

# === Figure 1: ROC + PR ===
prec_arr, rec_arr, _ = precision_recall_curve(y_te, probs)
prevalence = y_te.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

ax1.plot(fpr, tpr, color=AGGIE_BLUE, linewidth=2, label=f"AUROC = {auroc:.3f}")
ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, alpha=0.6)
ax1.fill_between(fpr, tpr, alpha=0.08, color=AGGIE_BLUE)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve (Best Model)")
ax1.legend(loc="lower right", framealpha=0.9)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_aspect("equal")

ax2.plot(rec_arr, prec_arr, color=AGGIE_GOLD_DARK, linewidth=2,
         label=f"AUPRC = {auprc:.3f}")
ax2.axhline(prevalence, linestyle="--", color="gray", linewidth=0.8, alpha=0.6,
            label=f"Prevalence = {prevalence:.2f}")
ax2.fill_between(rec_arr, prec_arr, alpha=0.08, color=AGGIE_GOLD_DARK)
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve (Best Model)")
ax2.legend(loc="upper right", framealpha=0.9)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_aspect("equal")

fig.tight_layout()
fig.savefig(OUT / "best_roc_pr.png", dpi=300, facecolor="white")
plt.close(fig)
print("  -> best_roc_pr.png")

# === Figure 2: Prediction distribution ===
bins = np.linspace(0, 1, 41)
centers = 0.5 * (bins[:-1] + bins[1:])
bar_w = bins[1] - bins[0]
h_norm, _ = np.histogram(probs[y_te == 0], bins=bins)
h_pe, _ = np.histogram(probs[y_te == 1], bins=bins)
n_pe = int((y_te == 1).sum())
n_norm = int((y_te == 0).sum())

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(centers, h_pe, width=bar_w, color="#C0392B", alpha=0.85,
       label=f"PE+ (n={n_pe})")
ax.bar(centers, -h_norm, width=bar_w, color=AGGIE_BLUE, alpha=0.75,
       label=f"Normal (n={n_norm})")
ax.axvline(tau, color=AGGIE_GOLD, linewidth=2, linestyle="--",
           label=f"Youden's J = {tau:.2f}")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Predicted P(PE)")
ax.set_ylabel("Count")
ax.set_title(f"Prediction Distribution (Best Model, AUROC={auroc:.3f})")
ax.legend(loc="upper right", framealpha=0.9, fontsize=7)

max_abs = max(h_norm.max(), h_pe.max())
ax.set_ylim(-max_abs * 1.15, max_abs * 1.15)
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([str(int(abs(t))) for t in yticks])

fig.tight_layout()
fig.savefig(OUT / "best_pred_dist.png", dpi=300, facecolor="white")
plt.close(fig)
print("  -> best_pred_dist.png")

# === Figure 3: Confusion Matrix (counts + normalized) ===
cm = confusion_matrix(y_te, preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

labels = ["Normal", "Preeclampsia"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
ax1.set_title("Confusion Matrix (Counts)")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else AGGIE_BLUE,
                 fontsize=14, fontweight="bold")

im2 = ax2.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=100)
ax2.set_title("Confusion Matrix (Normalized by True Label)")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm_norm[i, j]:.2f}%", ha="center", va="center",
                 color="white" if cm_norm[i, j] > 50 else AGGIE_BLUE,
                 fontsize=14, fontweight="bold")

for ax in (ax1, ax2):
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.spines[:].set_visible(True)

fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=300, facecolor="white")
plt.close(fig)
print("  -> confusion_matrix.png")

# === Table data for LaTeX ===
print(f"\n=== LaTeX Table Data ===")
print(f"Best Model: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
      f"Sens={100*sens:.1f}%  Spec={100*spec:.1f}%  "
      f"NPV={100*npv:.1f}%  F1={f1:.3f}")
