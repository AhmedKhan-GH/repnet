"""
Run inference on the balanced dataset using the best model weights.
Outputs ROC curve, classification report, and per-sample predictions.
"""

import sys
import shutil
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.models.repnet_efficient_hybrid import RepNetEfficientHybrid
from src.data.dataset import load_seniordesign
from src.preprocessing.filters import NotchFilter, BaselineWanderFilter
from src.preprocessing.normalization import ZScoreNormalization

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("optuna/2026-04-17_00-37-13")
DATA_DIR   = Path("data/seniordesign_upload_balanced")
PARAMS     = dict(stage_filters=(32, 64), wide_kernel=7, narrow_kernel=5,
                  n_heads=4, dropout=0.06355381998641418)

# ── Load data ─────────────────────────────────────────────────────────────────
if not (DATA_DIR / "metadata.csv").exists():
    shutil.copy(DATA_DIR / "metadata_balanced.csv", DATA_DIR / "metadata.csv")
    print("Copied metadata_balanced.csv → metadata.csv")

print("Loading dataset …")
X, y = load_seniordesign(str(DATA_DIR))

X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
X, _ = ZScoreNormalization(per_lead=True).transform(X)

print(f"Loaded {len(X)} samples — {(y==0).sum()} normal, {(y==1).sum()} preeclampsia")

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = RepNetEfficientHybrid(**PARAMS).to(device)
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))
model.eval()
print(f"Model loaded on {device}")

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    logits   = model(X_tensor)
    probs    = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

auroc = roc_auc_score(y, probs)
fpr, tpr, thresholds = roc_curve(y, probs)

# Youden's J optimal threshold
j_idx  = np.argmax(tpr - fpr)
best_t = thresholds[j_idx]
preds  = (probs >= best_t).astype(int)

print(f"\n── ROC ──────────────────────────────────")
print(f"  AUROC              : {auroc:.4f}")
print(f"  Youden threshold   : {best_t:.4f}")
print(f"  TPR @ Youden       : {tpr[j_idx]:.4f}")
print(f"  FPR @ Youden       : {fpr[j_idx]:.4f}")

print(f"\n── Classification report (threshold={best_t:.3f}) ──")
print(classification_report(y, preds, target_names=["Normal", "Preeclampsia"]))

print("── Confusion matrix ─────────────────────")
cm = confusion_matrix(y, preds)
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"  Sensitivity (recall PE) : {tp/(tp+fn):.4f}")
print(f"  Specificity (recall No) : {tn/(tn+fp):.4f}")

# ── Save ROC data ─────────────────────────────────────────────────────────────
import json, csv

out_dir = MODEL_DIR
np.save(out_dir / "roc_fpr.npy", fpr)
np.save(out_dir / "roc_tpr.npy", tpr)
np.save(out_dir / "roc_thresholds.npy", thresholds)
np.save(out_dir / "probs.npy", probs)
np.save(out_dir / "labels.npy", y)

print(f"\nSaved ROC arrays and predictions to {out_dir}/")

# ── Plot ROC ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUROC = {auroc:.4f}")
ax.scatter(fpr[j_idx], tpr[j_idx], color="crimson", zorder=5,
           label=f"Youden J (t={best_t:.3f})\nTPR={tpr[j_idx]:.3f}, FPR={fpr[j_idx]:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — RepNetEfficientHybrid\n(balanced dataset, full inference)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(out_dir / "roc_curve.png", dpi=150)
print(f"ROC plot saved to {out_dir / 'roc_curve.png'}")
