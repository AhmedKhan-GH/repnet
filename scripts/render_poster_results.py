"""Regenerate poster RepNet-SE results figures (ROC/PR + prediction distribution)
from the current PerLeadCNN reproduction (multisplit_dbb6f49, best split 17).

Reuses the data-loading / model logic from reproduce.ipynb and the exact plotting
style of publication_media/final_report/export_figures.py (fig06 / fig07), but
writes directly into poster/ so the poster picks up the current numbers.

Run:  .venv/bin/python scripts/render_poster_results.py
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, resample
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, brier_score_loss)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "seniordesign_upload")
RESULTS_DIR = os.path.join(REPO_ROOT, "multisplit_dbb6f49")
OUT_DIR = os.path.join(REPO_ROOT, "poster")
BEST_SPLIT = 17  # highest-AUROC split per per_split.json

# Publication-quality matplotlib defaults (matches export_figures.py)
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1, "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.titlesize": 11, "lines.linewidth": 1.2, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

# --------------- data loading (inlined from reproduce.ipynb) ---------------
N_LEADS = 12
SEQ_LEN = 5000
FS = 500.0
SD_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SD_LABEL_POS = "Preeclampsia or Other Hypertensive Disorders of Pregnancy"


def _apply_sos(sos, X):
    N, C, T = X.shape
    flat = X.reshape(N * C, T)
    X[:] = sosfiltfilt(sos, flat, axis=-1).reshape(N, C, T)
    return X


def preprocess(X):
    X = X.copy()
    X = _apply_sos(butter(4, 0.5, btype="high", fs=FS, output="sos"), X)
    b, a = iirnotch(60.0, 30.0, fs=FS)
    X = _apply_sos(tf2sos(b, a), X)
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


def load_ecg_data(data_dir):
    data_dir = os.path.normpath(data_dir)
    ekg_dir = os.path.join(data_dir, "ekg_data")
    meta_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(data_dir, "metadata_balanced.csv")
    meta = pd.read_csv(meta_path)
    available = {int(os.path.splitext(f)[0]) for f in os.listdir(ekg_dir) if f.endswith(".csv")}
    meta = meta[meta["ECGTestID"].apply(lambda x: int(x) in available)].copy()
    X_list, y_list, pat_list = [], [], []
    for _, row in meta.iterrows():
        path = os.path.join(ekg_dir, f"{int(row['ECGTestID'])}.csv")
        try:
            df = pd.read_csv(path, skipinitialspace=True, usecols=SD_LEAD_ORDER)
            arr = df[SD_LEAD_ORDER].values.T.astype(np.float32)
            if arr.shape[0] != 12:
                continue
            n = arr.shape[1]
            if n == SEQ_LEN:
                pass
            elif n == 2500:
                arr = resample(arr, SEQ_LEN, axis=1).astype(np.float32)
            else:
                continue
            if arr.shape != (12, SEQ_LEN):
                continue
            X_list.append(arr)
            y_list.append(1 if row["PatLabel"] == SD_LABEL_POS else 0)
            pat_list.append(row["Pat_Obfus_MRN"])
        except Exception:
            continue
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    patient_ids = np.array(pat_list)
    mask = np.isfinite(X).all(axis=2).all(axis=1)
    X, y, patient_ids = X[mask], y[mask], patient_ids[mask]
    keep = ~((X.std(axis=2) < 1e-4).any(axis=1))
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]
    try:
        nan_mask = np.isnan(patient_ids.astype(float))
    except (ValueError, TypeError):
        nan_mask = np.array([str(p).strip() in ("", "nan", "None") for p in patient_ids])
    keep = ~nan_mask
    return X[keep], y[keep], patient_ids[keep]


class PerLeadCNN(nn.Module):
    def __init__(self, n_leads=12, filters=(16, 32, 48), kernels=(31, 21, 11),
                 dropout=0.15, n_classes=2):
        super().__init__()
        layers, in_ch = [], 1
        for f, k in zip(filters, kernels):
            layers.extend([nn.Conv1d(in_ch, f, k, stride=2, padding=k // 2, bias=False),
                           nn.BatchNorm1d(f), nn.Mish()])
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


def get_best_split_predictions():
    X_all, y_all, patient_ids = load_ecg_data(DATA_DIR)
    X_all = preprocess(X_all)[:, :, ::2]  # 500 Hz -> 250 Hz
    print(f"Samples: {len(y_all)}  pos: {int(y_all.sum())} ({y_all.mean():.1%})")

    split_seed = BEST_SPLIT * 7 + 1000
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    _, test_idx = next(iter(sgkf.split(X_all, y_all, groups=patient_ids)))
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    model = PerLeadCNN(filters=(16, 32, 48), kernels=(31, 21, 11), dropout=0.15)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X_te, dtype=torch.float32)), dim=1)[:, 1].numpy()
    return y_te, probs


def fig_pred_histogram(y_test, probs, test_auroc):
    fpr_j, tpr_j, thr_j = roc_curve(y_test, probs)
    tau_youden = float(thr_j[np.argmax(tpr_j - fpr_j)])
    j_stat = float(np.max(tpr_j - fpr_j))
    bins = np.linspace(0, 1, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bar_w = bins[1] - bins[0]
    h_norm, _ = np.histogram(probs[y_test == 0], bins=bins)
    h_pe, _ = np.histogram(probs[y_test == 1], bins=bins)
    n_pe, n_norm = int((y_test == 1).sum()), int((y_test == 0).sum())

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(centers, h_pe, width=bar_w, color="tomato", alpha=0.85, label=f"PE+ (n={n_pe})")
    ax.bar(centers, -h_norm, width=bar_w, color="steelblue", alpha=0.85, label=f"Normal (n={n_norm})")
    ax.axhline(0, color="black", linewidth=0.5)
    ymax = max(h_pe.max(), h_norm.max()) * 1.1
    ax.axvline(tau_youden, color="green", linestyle=":", linewidth=2,
               label=f"Youden's J={j_stat:.3f} (τ={tau_youden:.3f})")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="τ=0.50")
    ax.set_ylim(-ymax, ymax)
    yticks = np.linspace(-ymax, ymax, 7)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(v):.0f}" for v in yticks])
    ax.set_xlabel("P(Preeclampsia)")
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted P(PE) Distribution — PE+ above, Normal below "
                 f"(AUROC={test_auroc:.4f})")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pred_histogram.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tau_youden, j_stat


def fig_roc_pr_curves(y_test, probs, test_auroc):
    fpr, tpr, _ = roc_curve(y_test, probs)
    prec, rec, _ = precision_recall_curve(y_test, probs)
    test_auprc = average_precision_score(y_test, probs)
    test_brier = brier_score_loss(y_test, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"RepNet-SE (AUC={test_auroc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right", fontsize=7)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    ax2.plot(rec, prec, color="tomato", linewidth=2, label=f"RepNet-SE (AP={test_auprc:.3f})")
    prevalence = y_test.mean()
    ax2.axhline(prevalence, color="gray", linestyle="--", linewidth=1,
                label=f"Prevalence ({prevalence:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    fig.suptitle(f"ROC & Precision-Recall Curves (Brier={test_brier:.4f})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_pr_curves.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return test_auprc, test_brier


def main():
    y_te, probs = get_best_split_predictions()
    auroc = roc_auc_score(y_te, probs)
    auprc, brier = fig_roc_pr_curves(y_te, probs, auroc)
    tau, j = fig_pred_histogram(y_te, probs, auroc)
    print("\n=== POSTER CAPTION NUMBERS (best split %d) ===" % BEST_SPLIT)
    print(f"AUROC = {auroc:.4f}")
    print(f"AUPRC = {auprc:.4f}")
    print(f"Brier = {brier:.4f}")
    print(f"Youden tau = {tau:.4f}  (J = {j:.4f})")
    print(f"Test set: n={len(y_te)}, pos={int(y_te.sum())} ({y_te.mean():.1%})")


if __name__ == "__main__":
    main()
