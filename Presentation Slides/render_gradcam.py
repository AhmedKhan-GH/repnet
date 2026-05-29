#!/usr/bin/env python3
"""Render bidirectional Grad-CAM saliency heatmaps for PerLeadCNN.

Adapted from export/notebooks/final_results.ipynb (Section 4c).
Computes Grad-CAM for both PE and Normal classes, then takes the
differential (cam_PE - cam_Normal) so red = PE evidence, blue = Normal evidence.
"""

import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "export" / "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import load_ecg_data, preprocess

OUT = Path(__file__).resolve().parent
DATA_DIR = str(REPO / "data" / "seniordesign_upload")
RESULTS_DIR = REPO / "multisplit_dbb6f49"
DEVICE = torch.device("cpu")

AGGIE_BLUE = "#022851"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
SHOW_LEADS = [0, 1, 4, 6]  # I, II, aVL, V1

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
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


# --- Load data and model ---
print("Loading data...")
X_all, y_all, patient_ids, _ = load_ecg_data(DATA_DIR)
X_all = preprocess(X_all)
X_all = X_all[:, :, ::2]

with open(RESULTS_DIR / "per_split.json") as f:
    splits = json.load(f)
best_idx = int(np.argmax([s["auroc"] for s in splits]))

split_seed = best_idx * 7 + 1000
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
dev_idx, test_idx = next(iter(sgkf.split(X_all, y_all, groups=patient_ids)))
X_te, y_te = X_all[test_idx], y_all[test_idx]

model = PerLeadCNN(filters=(16, 32, 48), kernels=(31, 21, 11), dropout=0.15)
model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pt",
                                 map_location=DEVICE, weights_only=True))
model.eval()

# --- Bidirectional Grad-CAM (from export/notebooks/final_results.ipynb) ---
_acts, _grads = {}, {}
def _fwd(m, i, o): _acts['v'] = o
def _bwd(m, gi, go): _grads['v'] = go[0]
h1 = model.backbone[-1].register_forward_hook(_fwd)
h2 = model.backbone[-1].register_full_backward_hook(_bwd)

T_input = X_te.shape[-1]

def compute_gradcam(xb, target_class):
    model.zero_grad()
    out = model(xb)
    out[:, target_class].sum().backward()
    weights = _grads['v'].mean(dim=-1, keepdim=True)
    cam = torch.relu((weights * _acts['v']).sum(dim=1))
    B = xb.shape[0]
    cam = cam.reshape(B, 12, -1)
    cam = F.interpolate(cam, size=T_input, mode='linear', align_corners=False)
    return cam.detach().cpu().numpy()

print("Computing bidirectional Grad-CAM for all test samples...")
all_cams_pe, all_cams_norm, all_probs = [], [], []
for s in range(0, len(X_te), 32):
    e = min(s + 32, len(X_te))
    xb = torch.tensor(X_te[s:e], dtype=torch.float32, device=DEVICE)
    all_cams_pe.append(compute_gradcam(xb, target_class=1))
    all_cams_norm.append(compute_gradcam(xb, target_class=0))
    with torch.no_grad():
        all_probs.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())

all_cams_pe = np.concatenate(all_cams_pe)
all_cams_norm = np.concatenate(all_cams_norm)
all_probs = np.concatenate(all_probs)

for cams in [all_cams_pe, all_cams_norm]:
    for i in range(len(cams)):
        cmin, cmax = cams[i].min(), cams[i].max()
        if cmax > cmin:
            cams[i] = (cams[i] - cmin) / (cmax - cmin)

all_cams_diff = all_cams_pe - all_cams_norm

h1.remove()
h2.remove()

# --- Pick exemplar cases (not the most extreme — second-most confident) ---
tp_mask = (y_te == 1) & (all_probs > 0.5)
tn_mask = (y_te == 0) & (all_probs < 0.3)
tp_sorted = np.where(tp_mask)[0][np.argsort(-all_probs[tp_mask])]
tn_sorted = np.where(tn_mask)[0][np.argsort(all_probs[tn_mask])]
tp_idx = tp_sorted[9] if len(tp_sorted) > 9 else tp_sorted[-1]
tn_idx = tn_sorted[9] if len(tn_sorted) > 9 else tn_sorted[-1]

print(f"Exemplar TP: P(PE) = {all_probs[tp_idx]:.3f}")
print(f"Exemplar TN: P(PE) = {all_probs[tn_idx]:.3f}")

# === Figure 1: Bidirectional Grad-CAM (2 leads, TP vs TN) ===
n_leads = len(SHOW_LEADS)
time_sec = np.arange(T_input) / 250.0

fig, axes = plt.subplots(n_leads, 2, figsize=(12, 1.8 * n_leads),
                         sharex=True,
                         gridspec_kw={"hspace": 0.4, "wspace": 0.15})

for col, (idx, title) in enumerate([
    (tp_idx, f"True Positive — P(PE) = {all_probs[tp_idx]:.2f}"),
    (tn_idx, f"True Negative — P(PE) = {all_probs[tn_idx]:.2f}"),
]):
    ecg = X_te[idx]
    cam_diff = all_cams_diff[idx]
    vabs = max(abs(cam_diff.min()), abs(cam_diff.max()))

    axes[0, col].set_title(title, fontsize=11, fontweight="bold", color=AGGIE_BLUE)
    for row, lead_i in enumerate(SHOW_LEADS):
        ax = axes[row, col]
        sig = ecg[lead_i]
        margin = 0.15 * (sig.max() - sig.min() + 1e-6)

        ax.imshow(cam_diff[lead_i].reshape(1, -1), aspect='auto', cmap='RdBu_r',
                  alpha=0.45, extent=[0, 10, sig.min() - margin, sig.max() + margin],
                  origin='lower',
                  norm=SymLogNorm(linthresh=0.03, vmin=-vabs, vmax=vabs))
        ax.plot(time_sec, sig, 'k-', linewidth=0.7)
        ax.set_xlim(0, 10)
        ax.set_ylim(sig.min() - margin, sig.max() + margin)
        ax.set_ylabel(LEAD_NAMES[lead_i], fontsize=10, fontweight="bold",
                      rotation=0, labelpad=20, va="center")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if row < n_leads - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=9)

sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                           norm=SymLogNorm(linthresh=0.03, vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.012, pad=0.02)
cbar.set_label("Red → PE evidence    Blue → Normal evidence", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.savefig(OUT / "gradcam_saliency.png", dpi=300, facecolor="white")
plt.close(fig)
print("  -> gradcam_saliency.png")

# === Figure 2: Lead importance bar chart only ===
imp_pe = all_cams_diff[y_te == 1].mean(axis=(0, 2))
imp_norm = all_cams_diff[y_te == 0].mean(axis=(0, 2))

fig, ax = plt.subplots(figsize=(5, 4.5))

x = np.arange(12)
w = 0.35
ax.barh(x - w / 2, imp_pe, height=w, color="#e74c3c", alpha=0.8, label="PE patients")
ax.barh(x + w / 2, imp_norm, height=w, color="#3498db", alpha=0.8, label="Normal patients")
ax.axvline(0, color='black', linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels(LEAD_NAMES)
ax.set_xlabel("Mean Differential Saliency (PE − Normal)")
ax.set_title("Lead Importance (Bidirectional Grad-CAM)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "lead_importance.png", dpi=300, facecolor="white")
plt.close(fig)
print("  -> lead_importance.png")

# Print lead ranking
print("\nLead importance (mean differential saliency for PE patients):")
order = np.argsort(imp_pe)[::-1]
for i in order:
    print(f"  {LEAD_NAMES[i]:>4s}: PE={imp_pe[i]:+.3f}  Normal={imp_norm[i]:+.3f}")
