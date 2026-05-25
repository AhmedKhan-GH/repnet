#!/usr/bin/env python3
"""Export all RepNet-SE analysis figures as 300 DPI PNGs for publication.

Re-renders every figure from notebooks/repnet_se_analysis.ipynb using
matplotlib for consistent, high-resolution output.

Usage:
    python publication_media/final_report/export_figures.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — must precede any `src` imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")                      # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress convergence / future warnings from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 11,
    "lines.linewidth": 1.2,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

RESULTS_DIR = REPO_ROOT / "cv_results" / "neural_final_2026-05-20_08-29-02"
FIG_DIR = REPO_ROOT / "publication_media" / "final_report" / "figures"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def savefig(fig, name: str) -> None:
    """Save figure and close."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.relative_to(REPO_ROOT)}")


# ===================================================================
# PART A: Statistical summary figures (no model / data needed)
# ===================================================================

def load_results():
    """Load the 20-seed results JSON."""
    with open(RESULTS_DIR / "results.json") as f:
        results = json.load(f)
    aurocs = np.array([r["metrics"]["auroc"] for r in results])
    auprcs = np.array([r["metrics"]["auprc"] for r in results])
    seeds = [r["seed"] for r in results]
    return results, aurocs, auprcs, seeds


def fig01_perseed_auroc_bar(aurocs, seeds):
    """Per-seed AUROC bar chart sorted descending."""
    print("Figure 1: perseed_auroc_bar.png")
    sort_idx = np.argsort(aurocs)[::-1]
    colors = ["#2ecc71" if a >= 0.70 else "#e74c3c" for a in aurocs[sort_idx]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_labels = [str(seeds[i]) for i in sort_idx]
    bars = ax.bar(x_labels, aurocs[sort_idx], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1, label="Target 0.70")
    ax.axhline(aurocs.mean(), color="blue", linestyle=":", linewidth=1,
               label=f"Mean {aurocs.mean():.4f}")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test AUROC")
    ax.set_title("RepNet-SE: Per-Seed Test AUROC (20 Holdout Splits)")
    ax.set_ylim(0.4, 0.85)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    savefig(fig, "perseed_auroc_bar.png")


def fig02_auroc_auprc_histograms(aurocs, auprcs):
    """Side-by-side AUROC and AUPRC distribution histograms."""
    print("Figure 2: auroc_auprc_histograms.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(aurocs, bins=12, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(aurocs.mean(), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean {aurocs.mean():.4f}")
    ax1.set_title("AUROC Distribution")
    ax1.set_xlabel("AUROC")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=7)

    ax2.hist(auprcs, bins=12, color="orange", alpha=0.8, edgecolor="white")
    ax2.axvline(auprcs.mean(), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean {auprcs.mean():.4f}")
    ax2.set_title("AUPRC Distribution")
    ax2.set_xlabel("AUPRC")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=7)

    fig.suptitle("Test Metric Distributions (20 Splits)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "auroc_auprc_histograms.png")


def fig03_auroc_boxplot(aurocs):
    """Box + strip plot of AUROC with mean and target lines."""
    print("Figure 3: auroc_boxplot.png")
    n = len(aurocs)
    mean = aurocs.mean()
    std = aurocs.std(ddof=1)
    sem = std / np.sqrt(n)
    ci_lo = mean - 1.96 * sem
    ci_hi = mean + 1.96 * sem

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    bp = ax.boxplot(aurocs, widths=0.5, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                    boxprops=dict(facecolor=(70/255, 130/255, 180/255, 0.3), edgecolor="navy"),
                    medianprops=dict(color="navy", linewidth=1.5))
    # Strip (jittered points)
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=n)
    ax.scatter(np.ones(n) + jitter, aurocs, s=30, c="steelblue", alpha=0.7,
               edgecolors="white", linewidths=0.5, zorder=5)
    ax.axhline(mean, color="red", linestyle="--", linewidth=1,
               label=f"Mean {mean:.4f}")
    ax.axhline(0.70, color="green", linestyle=":", linewidth=1,
               label="Target 0.70")
    ax.set_ylabel("Test AUROC")
    ax.set_title(f"AUROC across {n} seeds\n"
                 f"mean={mean:.4f}  SD={std:.4f}  95% CI [{ci_lo:.4f},{ci_hi:.4f}]",
                 fontsize=9)
    ax.set_ylim(max(0, aurocs.min() - 0.05), min(1, aurocs.max() + 0.05))
    ax.set_xticks([1])
    ax.set_xticklabels(["AUROC"])
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, "auroc_boxplot.png")


def fig04_auroc_vs_auprc_scatter(aurocs, auprcs, seeds):
    """Per-seed AUROC vs AUPRC scatter with seed labels."""
    print("Figure 4: auroc_vs_auprc_scatter.png")
    r_corr, p_corr = sp_stats.pearsonr(aurocs, auprcs)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(aurocs, auprcs, c=aurocs, cmap="RdYlGn", s=60,
                    edgecolors="gray", linewidths=0.5, zorder=5)
    for i, s in enumerate(seeds):
        ax.annotate(str(s), (aurocs[i], auprcs[i]), fontsize=5,
                    textcoords="offset points", xytext=(0, 6), ha="center")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("AUROC", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel("Test AUROC")
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Per-Seed AUROC vs AUPRC (20 Holdout Splits)")
    ax.annotate(f"Pearson r={r_corr:.3f}, p={p_corr:.4f}",
                xy=(0.02, 0.98), xycoords="axes fraction", fontsize=7,
                va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    fig.tight_layout()
    savefig(fig, "auroc_vs_auprc_scatter.png")


def fig05_repnet_vs_lgbm_boxplot(aurocs):
    """RepNet-SE vs LightGBM side-by-side box + strip."""
    print("Figure 5: repnet_vs_lgbm_boxplot.png")
    lgbm_dirs = sorted((REPO_ROOT / "cv_results").glob("final_*/results.json"))
    if not lgbm_dirs:
        print("  [SKIP] No LightGBM results found — generating placeholder")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No LightGBM results available for comparison",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Neural vs LightGBM: Test AUROC Distribution")
        savefig(fig, "repnet_vs_lgbm_boxplot.png")
        return

    with open(lgbm_dirs[-1]) as f:
        lgbm_data = json.load(f)
    lgbm_aurocs = np.array([r["metrics"]["auroc"] for r in lgbm_data])

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [aurocs, lgbm_aurocs]
    labels = ["RepNet-SE (Neural)", "5-Bag LightGBM"]
    colors_box = ["steelblue", "orange"]

    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True,
                    labels=labels,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for i, d in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, s=25, c=colors_box[i],
                   alpha=0.7, edgecolors="white", linewidths=0.5, zorder=5)

    ax.axhline(0.70, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("Test AUROC")
    ax.set_title("Neural vs LightGBM: Test AUROC Distribution")
    fig.tight_layout()
    savefig(fig, "repnet_vs_lgbm_boxplot.png")


# ===================================================================
# PART B: Load model and data
# ===================================================================

def load_model_and_data(results, aurocs, seeds):
    """Load best-seed model, raw data, and compute predictions on test set."""
    from src.models.repnet_se import RepNetSE
    from src.train_explorer_v2 import load_combined, preprocess_waveforms

    best_seed = int(seeds[np.argmax(aurocs)])
    print(f"\nLoading best model (seed={best_seed}, AUROC={aurocs.max():.4f})...")

    ckpt = torch.load(
        RESULTS_DIR / "weights" / f"best_seed_{best_seed}.pt",
        map_location=DEVICE, weights_only=False,
    )
    model = RepNetSE(**ckpt["net_cfg"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} params, config={ckpt['net_cfg']}")

    print("Loading and preprocessing data...")
    X_wave, X_feat, y, patient_ids, feat_cols = load_combined(
        str(REPO_ROOT / "data" / "seniordesign_upload")
    )
    X_wave = preprocess_waveforms(X_wave)
    print(f"  Data: {X_wave.shape}, pos={int(y.sum())} ({100 * y.mean():.1f}%)")

    # Recreate the exact same split as the notebook
    ss = np.random.SeedSequence(best_seed)
    split_seed = int(ss.generate_state(1)[0])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=patient_ids))

    X_test = X_wave[test_idx]
    y_test = y[test_idx]
    print(f"  Test set: N={len(y_test)}, pos={int(y_test.sum())} ({100 * y_test.mean():.1f}%)")

    # Get predictions
    print("Computing predictions...")
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    test_auroc = roc_auc_score(y_test, probs)
    print(f"  Test AUROC (single model): {test_auroc:.4f}")

    return model, X_test, y_test, probs, x_t, test_auroc, best_seed


# ===================================================================
# PART C: Best-seed model figures
# ===================================================================

def fig06_pred_histogram(y_test, probs, test_auroc):
    """Mirror histogram of predicted P(PE) by true class — raw counts + Youden's J."""
    print("Figure 6: pred_histogram.png")

    fpr_j, tpr_j, thr_j = roc_curve(y_test, probs)
    tau_youden = float(thr_j[np.argmax(tpr_j - fpr_j)])
    j_stat = float(np.max(tpr_j - fpr_j))

    bins = np.linspace(0, 1, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bar_w = bins[1] - bins[0]
    h_norm, _ = np.histogram(probs[y_test == 0], bins=bins)
    h_pe, _ = np.histogram(probs[y_test == 1], bins=bins)

    n_pe = int((y_test == 1).sum())
    n_norm = int((y_test == 0).sum())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(centers, h_pe, width=bar_w, color="tomato", alpha=0.85,
           label=f"PE+ (n={n_pe})")
    ax.bar(centers, -h_norm, width=bar_w, color="steelblue", alpha=0.85,
           label=f"Normal (n={n_norm})")
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
    ax.set_title(f"Predicted P(PE) Distribution (AUROC={test_auroc:.4f})\n"
                 f"PE+ above axis (n={n_pe}), Normal below (n={n_norm}) "
                 f"— raw counts, proportional to prevalence")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, "pred_histogram.png")


def fig07_roc_pr_curves(y_test, probs, test_auroc):
    """ROC + Precision-Recall curves side by side."""
    print("Figure 7: roc_pr_curves.png")
    fpr, tpr, _ = roc_curve(y_test, probs)
    prec, rec, _ = precision_recall_curve(y_test, probs)
    test_auprc = average_precision_score(y_test, probs)
    test_brier = brier_score_loss(y_test, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.plot(fpr, tpr, color="steelblue", linewidth=2,
             label=f"RepNet-SE (AUC={test_auroc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right", fontsize=7)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    ax2.plot(rec, prec, color="tomato", linewidth=2,
             label=f"RepNet-SE (AP={test_auprc:.3f})")
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
    savefig(fig, "roc_pr_curves.png")
    return fpr, tpr


def fig08_threshold_sweep(y_test, probs, fpr, tpr):
    """Youden's J, sensitivity, and specificity vs threshold."""
    print("Figure 8: threshold_sweep.png")
    thresholds = np.arange(0.01, 1.00, 0.01)
    sensitivities, specificities, youdens = [], [], []

    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        sensitivities.append(sens)
        specificities.append(spec)
        youdens.append(sens + spec - 1)

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    youdens = np.array(youdens)

    # Find Youden optimal from ROC curve
    _, _, thresholds_roc = roc_curve(y_test, probs)
    fpr_roc, tpr_roc, _ = roc_curve(y_test, probs)
    tau_youden = float(thresholds_roc[np.argmax(tpr_roc - fpr_roc)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, sensitivities, color="tomato", linewidth=1.5, label="Sensitivity")
    ax.plot(thresholds, specificities, color="steelblue", linewidth=1.5, label="Specificity")
    ax.plot(thresholds, youdens, color="green", linewidth=1.5, linestyle="--", label="Youden's J")
    ax.axvline(tau_youden, color="black", linestyle=":", linewidth=1,
               label=f"Youden optimal ({tau_youden:.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Threshold Sweep: Sensitivity / Specificity / Youden's J")
    ax.legend(loc="center right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "threshold_sweep.png")


def fig09_calibration_curve(y_test, probs, test_auroc):
    """Calibration curve with prediction histogram."""
    print("Figure 9: calibration_curve.png")
    test_brier = brier_score_loss(y_test, probs)
    fraction_pos, mean_predicted = calibration_curve(
        y_test, probs, n_bins=10, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5),
                                    gridspec_kw={"height_ratios": [0.7, 0.3],
                                                 "hspace": 0.08})

    ax1.plot(mean_predicted, fraction_pos, "o-", color="steelblue", linewidth=2,
             markersize=6, label="RepNet-SE")
    ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfectly calibrated")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Calibration Curve (Brier={test_brier:.4f})")
    ax1.legend(fontsize=7)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticklabels([])

    ax2.hist(probs[y_test == 0], bins=20, color="steelblue", alpha=0.6,
             label="Normal", edgecolor="white")
    ax2.hist(probs[y_test == 1], bins=20, color="tomato", alpha=0.6,
             label="PE+", edgecolor="white")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=7)

    fig.suptitle(f"Calibration Analysis (Brier={test_brier:.4f})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "calibration_curve.png")


# ===================================================================
# PART D: Attention / interpretability figures
# ===================================================================

def fig10_lead_attention_bars(model, x_t, y_test):
    """Per-lead attention weights by class (bar chart)."""
    print("Figure 10: lead_attention_bars.png")
    with torch.no_grad():
        _ = model(x_t)
        lead_weights = model.lead_pool.last_weights.cpu().numpy()  # (N, 12)

    w_neg = lead_weights[y_test == 0].mean(axis=0)
    w_pos = lead_weights[y_test == 1].mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_pos = np.arange(12)
    w = 0.35
    ax.bar(x_pos - w / 2, w_neg, w, label="Normal (PE-)", color="steelblue", alpha=0.8)
    ax.bar(x_pos + w / 2, w_pos, w, label="Preeclampsia (PE+)", color="red", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(LEAD_NAMES)
    ax.set_xlabel("ECG Lead")
    ax.set_ylabel("Attention Weight (softmax)")
    ax.set_title("Lead Attention Weights by Class")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "lead_attention_bars.png")
    return lead_weights


def fig11_lead_attention_heatmap(lead_weights, probs):
    """Lead attention heatmap across samples sorted by P(PE)."""
    print("Figure 11: lead_attention_heatmap.png")
    sort_by_prob = np.argsort(probs)

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(lead_weights[sort_by_prob].T, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels(LEAD_NAMES, fontsize=7)
    ax.set_xlabel("Samples (sorted by predicted PE probability)")
    ax.set_ylabel("ECG Lead")
    ax.set_title("Lead Attention Weights (samples sorted by P(PE), low -> high)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Weight", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "lead_attention_heatmap.png")


def _capture_crosslead_attention(model, x_batch):
    """Run forward pass and capture cross-lead attention weights from all stages."""
    attn_weights_store = {}

    def make_attn_hook(name):
        def hook(module, input, output):
            _, weights = output
            if weights is not None:
                attn_weights_store[name] = weights.detach().cpu().numpy()
        return hook

    hooks = []
    hooks.append(model.stage1_attn.attn.register_forward_hook(make_attn_hook("stage1")))
    for i, stage in enumerate(model.later_stages):
        hooks.append(stage["attn"].attn.register_forward_hook(make_attn_hook(f"stage{i + 2}")))

    with torch.no_grad():
        _ = model(x_batch)

    for h in hooks:
        h.remove()
    return attn_weights_store


def fig12_crosslead_attention_stages(model, x_t):
    """Average cross-lead attention matrices per stage."""
    print("Figure 12: crosslead_attention_stages.png")
    n_use = min(50, x_t.shape[0])
    attn_store = _capture_crosslead_attention(model, x_t[:n_use])

    n_stages = len(attn_store)
    fig, axes = plt.subplots(1, n_stages, figsize=(3.5 * n_stages, 3.5))
    if n_stages == 1:
        axes = [axes]

    for i, (name, weights) in enumerate(attn_store.items()):
        avg_w = weights.mean(axis=0)
        if avg_w.shape[0] == 12 and avg_w.shape[1] == 12:
            im = axes[i].imshow(avg_w, cmap="Blues", aspect="equal")
            axes[i].set_xticks(range(12))
            axes[i].set_xticklabels(LEAD_NAMES, fontsize=5, rotation=45)
            axes[i].set_yticks(range(12))
            axes[i].set_yticklabels(LEAD_NAMES, fontsize=5)
            axes[i].set_title(f"Stage {i + 1} Attention", fontsize=8)
            if i == n_stages - 1:
                cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=5)

    fig.suptitle(f"Cross-Lead Attention (averaged over {n_use} samples)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "crosslead_attention_stages.png")


def fig13_crosslead_attention_diff(model, x_t, y_test):
    """Cross-lead attention difference PE+ vs PE-."""
    print("Figure 13: crosslead_attention_diff.png")
    n_use = min(50, x_t.shape[0])
    y_sub = y_test[:n_use]

    attn_by_class = {}
    for label, label_name in [(0, "PE-"), (1, "PE+")]:
        mask = y_sub == label
        if mask.sum() == 0:
            continue
        x_sub = x_t[:n_use][mask]
        store = _capture_crosslead_attention(model, x_sub)
        attn_by_class[label_name] = {k: v.mean(axis=0) for k, v in store.items()}

    if "PE+" not in attn_by_class or "PE-" not in attn_by_class:
        print("  [SKIP] Not enough class representatives")
        return

    # Find the deepest stage with 12x12 attention
    stage_key = None
    for k in reversed(list(attn_by_class["PE+"].keys())):
        if attn_by_class["PE+"][k].shape == (12, 12):
            stage_key = k
            break
    if stage_key is None:
        print("  [SKIP] No 12x12 attention found")
        return

    diff = attn_by_class["PE+"][stage_key] - attn_by_class["PE-"][stage_key]
    stage_num = stage_key.replace("stage", "")

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(diff).max()
    im = ax.imshow(diff, cmap="RdBu_r", aspect="equal",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(LEAD_NAMES, fontsize=7, rotation=45)
    ax.set_yticks(range(12))
    ax.set_yticklabels(LEAD_NAMES, fontsize=7)
    ax.set_title(f"Stage {stage_num} Cross-Lead Attention Difference (PE+ minus PE-)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "crosslead_attention_diff.png")


# ===================================================================
# PART E: Saliency maps
# ===================================================================

def compute_saliency(model, x, target_class=1):
    """Compute gradient-based saliency map."""
    x_in = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    logits = model(x_in)
    score = logits[:, target_class].sum()
    score.backward()
    saliency = x_in.grad.abs().cpu().numpy()
    return saliency


def fig14_saliency_heatmap(model, X_test, y_test):
    """Saliency heatmap (leads x time) for PE- and PE+."""
    print("Figure 14: saliency_heatmap.png")
    n_samples = min(100, len(X_test))
    x_sal = torch.tensor(X_test[:n_samples], dtype=torch.float32).to(DEVICE)
    saliency = compute_saliency(model, x_sal)

    sal_neg = saliency[y_test[:n_samples] == 0].mean(axis=0)  # (12, 2500)
    sal_pos = saliency[y_test[:n_samples] == 1].mean(axis=0)  # (12, 2500)
    time_axis = np.arange(2500) / 250.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={"hspace": 0.25})
    vmax = max(sal_neg.max(), sal_pos.max())

    im1 = ax1.imshow(sal_neg, aspect="auto", cmap="hot",
                     extent=[0, 10, -0.5, 11.5], interpolation="bilinear",
                     vmin=0, vmax=vmax)
    ax1.set_yticks(range(12))
    ax1.set_yticklabels(LEAD_NAMES, fontsize=7)
    ax1.set_title("Normal (PE-)", fontsize=9, fontweight="bold")

    im2 = ax2.imshow(sal_pos, aspect="auto", cmap="hot",
                     extent=[0, 10, -0.5, 11.5], interpolation="bilinear",
                     vmin=0, vmax=vmax)
    ax2.set_yticks(range(12))
    ax2.set_yticklabels(LEAD_NAMES, fontsize=7)
    ax2.set_title("Preeclampsia (PE+)", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], location="right",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Saliency", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Average Gradient Saliency Maps by Class",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "saliency_heatmap.png")
    return saliency, sal_neg, sal_pos


def fig15_saliency_per_lead_bars(sal_neg, sal_pos):
    """Average saliency per lead by class."""
    print("Figure 15: saliency_per_lead_bars.png")
    lead_sal_neg = sal_neg.mean(axis=1)
    lead_sal_pos = sal_pos.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_pos = np.arange(12)
    w = 0.35
    ax.bar(x_pos - w / 2, lead_sal_neg, w, label="Normal (PE-)",
           color="steelblue", alpha=0.8)
    ax.bar(x_pos + w / 2, lead_sal_pos, w, label="Preeclampsia (PE+)",
           color="red", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(LEAD_NAMES)
    ax.set_xlabel("ECG Lead")
    ax.set_ylabel("Mean |Gradient|")
    ax.set_title("Average Saliency per Lead by Class")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "saliency_per_lead_bars.png")


def fig16_saliency_overlay_ecg(X_test, y_test, probs, saliency):
    """Single high-confidence PE+ sample with saliency overlay on all 12 leads."""
    print("Figure 16: saliency_overlay_ecg.png")
    n_samples = saliency.shape[0]
    pe_mask = y_test[:n_samples] == 1
    pe_probs = probs[:n_samples][pe_mask]
    pe_indices = np.where(pe_mask)[0]

    if len(pe_indices) == 0:
        print("  [SKIP] No PE+ samples in saliency batch")
        return

    best_pe_idx = pe_indices[np.argmax(pe_probs)]
    ecg = X_test[best_pe_idx]
    sal = saliency[best_pe_idx]
    time_axis = np.arange(2500) / 250.0

    eps = 0.02 * sal.max()
    sal_log = np.log1p(sal / eps) / np.log1p(sal.max() / eps)

    fig, axes = plt.subplots(12, 1, figsize=(12, 14),
                             gridspec_kw={"hspace": 0.08,
                                          "top": 0.94, "bottom": 0.04,
                                          "left": 0.06, "right": 0.90})

    for i, name in enumerate(LEAD_NAMES):
        ax = axes[i]
        ax.plot(time_axis, ecg[i], "k-", linewidth=0.5)
        sc = ax.scatter(time_axis, ecg[i], c=sal_log[i], cmap="hot",
                        s=1, vmin=0, vmax=1, zorder=5)
        ax.set_xlim(0, time_axis[-1])
        ylim = max(2.5, np.abs(ecg[i]).max() * 1.2)
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=8, fontweight="bold", rotation=0,
                      labelpad=14, va="center")
        if i < 11:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"ECG with Log-Saliency Overlay (PE+ sample, P(PE)={probs[best_pe_idx]:.3f})",
                 fontsize=10, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), location="right",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Log-Saliency", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    savefig(fig, "saliency_overlay_ecg.png")


# ===================================================================
# PART F: SE channel analysis
# ===================================================================

def fig17_se_weights_by_class(model, x_t, y_test):
    """SE channel weights PE+ vs PE- at each stage."""
    print("Figure 17: se_weights_by_class.png")

    def _capture_se_weights(model, x_batch):
        se_store = {}
        def make_se_hook(name):
            def hook(module, input, output):
                x = input[0]
                w = x.mean(dim=-1)
                w = module.fc(w)
                se_store[name] = w.detach().cpu().numpy()
            return hook

        hooks = []
        hooks.append(model.stage1_conv.se.register_forward_hook(make_se_hook("stage1")))
        for i, stage in enumerate(model.later_stages):
            hooks.append(stage["conv"].se.register_forward_hook(make_se_hook(f"stage{i + 2}")))
        with torch.no_grad():
            _ = model(x_batch)
        for h in hooks:
            h.remove()
        return se_store

    se_by_class = {}
    for label, label_name in [(0, "PE-"), (1, "PE+")]:
        mask = y_test == label
        x_sub = x_t[mask][:100]
        store = _capture_se_weights(model, x_sub)
        se_by_class[label_name] = {k: v.mean(axis=0) for k, v in store.items()}

    if "PE+" not in se_by_class or "PE-" not in se_by_class:
        print("  [SKIP] Missing class data")
        return

    n_se = len(se_by_class["PE+"])
    fig, axes = plt.subplots(1, n_se, figsize=(3.5 * n_se, 3.5))
    if n_se == 1:
        axes = [axes]

    for i, name in enumerate(se_by_class["PE+"].keys()):
        diff = se_by_class["PE+"][name] - se_by_class["PE-"][name]
        colors = ["tomato" if d > 0 else "steelblue" for d in diff]
        axes[i].bar(range(len(diff)), diff, color=colors, alpha=0.8, edgecolor="white", linewidth=0.3)
        axes[i].axhline(0, color="black", linewidth=0.5)
        stage_num = name.replace("stage", "")
        axes[i].set_title(f"Stage {stage_num} SE Diff\n(PE+ minus PE-)", fontsize=8)
        axes[i].set_xlabel("Channel", fontsize=7)
        axes[i].tick_params(labelsize=5)

    fig.suptitle("SE Channel Weight Difference by Class (PE+ minus PE-)\n"
                 "Red = more active for PE+, Blue = more active for Normal",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "se_weights_by_class.png")


# ===================================================================
# PART G: Integrated Gradients
# ===================================================================

def integrated_gradients(model, x_np, target_class=1, n_steps=50, smooth_sigma=5.0):
    """Integrated Gradients: accumulate gradients along interpolation path."""
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0.5 / n_steps, 1.0 - 0.5 / n_steps, n_steps, device=DEVICE)
    interp = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * (x - baseline).unsqueeze(0)
    interp.requires_grad_(True)
    model.zero_grad()
    logits = model(interp)
    grads = torch.autograd.grad(logits[:, target_class].sum(), interp)[0]
    avg_grad = grads.mean(dim=0)
    attr = ((x - baseline) * avg_grad).cpu().numpy()
    if smooth_sigma > 0:
        attr = gaussian_filter1d(attr, sigma=smooth_sigma, axis=-1)
    return attr


def gradcam_per_lead(model, x_np, target_class=1, smooth_sigma=2.0):
    """Grad-CAM at the last convolutional stage, per lead."""
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    activations = {}

    def fwd_hook(_m, _i, o):
        activations["feat"] = o
        o.retain_grad()

    h = model.later_stages[-1]["conv"].register_forward_hook(fwd_hook)
    model.zero_grad()
    logits = model(x)
    logits[0, target_class].backward()
    h.remove()

    feat = activations["feat"]
    grad = feat.grad
    weights = grad.mean(dim=-1, keepdim=True)
    cam = F.relu((weights * feat).sum(dim=2))
    cam = F.interpolate(cam.view(1, 12, -1), size=x_np.shape[-1],
                        mode="linear", align_corners=False)
    cam = cam.squeeze(0).detach().cpu().numpy()
    if smooth_sigma > 0:
        cam = gaussian_filter1d(cam, sigma=smooth_sigma, axis=-1)
    return cam


def combined_saliency(model, x_np, target_class=1):
    """Average of normalized IG and Grad-CAM for robust attribution."""
    ig = integrated_gradients(model, x_np, target_class=target_class)
    gc = gradcam_per_lead(model, x_np, target_class=target_class)
    ig_n = ig / (np.abs(ig).max() or 1.0)
    gc_n = gc / (np.abs(gc).max() or 1.0)
    return 0.5 * (ig_n + gc_n), ig, gc


def log_scale_attr(a, eps=0.02):
    """Compress dynamic range so subtle regions remain visible."""
    sign = np.sign(a)
    mag = np.abs(a)
    return sign * np.log1p(mag / eps) / np.log1p(1.0 / eps)


def plot_saliency_ecg(ecg, attr, prob, title, savename, lead_names=LEAD_NAMES, fs=250.0):
    """12-lead ECG with log-scaled saliency heatmap overlay."""
    t = np.arange(ecg.shape[1]) / fs
    attr_log = log_scale_attr(attr / (np.abs(attr).max() or 1.0))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, axes = plt.subplots(12, 1, figsize=(14, 14),
                             gridspec_kw={"hspace": 0.08,
                                          "top": 0.94, "bottom": 0.04,
                                          "left": 0.05, "right": 0.92})
    for i, name in enumerate(lead_names):
        ax = axes[i]
        ax.imshow(attr_log[i][np.newaxis, :], aspect="auto", cmap="RdBu_r",
                  norm=norm, extent=[0, t[-1], -2.5, 2.5], alpha=0.65,
                  interpolation="bilinear")
        ax.plot(t, ecg[i], "k-", linewidth=0.5)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=8, fontweight="bold", rotation=0,
                      labelpad=14, va="center")
        if i < 11:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), location="right",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Attribution (Red->PE+, Blue->Normal)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    savefig(fig, savename)


def select_cases(y_test, probs):
    """Select 8 representative cases: confident + borderline TP/TN/FP/FN."""
    pred = (probs >= 0.5).astype(int)

    tp_mask = (y_test == 1) & (pred == 1)
    fp_mask = (y_test == 0) & (pred == 1)
    fn_mask = (y_test == 1) & (pred == 0)
    tn_mask = (y_test == 0) & (pred == 0)

    cases = {}
    if tp_mask.any():
        idx = np.where(tp_mask)[0][np.argmax(probs[tp_mask])]
        cases["confident_TP"] = idx
    if tn_mask.any():
        idx = np.where(tn_mask)[0][np.argmin(probs[tn_mask])]
        cases["confident_TN"] = idx
    if fp_mask.any():
        idx = np.where(fp_mask)[0][np.argmax(probs[fp_mask])]
        cases["confident_FP"] = idx
    if fn_mask.any():
        idx = np.where(fn_mask)[0][np.argmin(probs[fn_mask])]
        cases["confident_FN"] = idx
    if tp_mask.any():
        idx = np.where(tp_mask)[0][np.argmin(np.abs(probs[tp_mask] - 0.5))]
        cases["borderline_TP"] = idx
    if tn_mask.any():
        idx = np.where(tn_mask)[0][np.argmin(np.abs(probs[tn_mask] - 0.5))]
        cases["borderline_TN"] = idx
    if fp_mask.any():
        idx = np.where(fp_mask)[0][np.argmin(np.abs(probs[fp_mask] - 0.5))]
        cases["borderline_FP"] = idx
    if fn_mask.any():
        idx = np.where(fn_mask)[0][np.argmin(np.abs(probs[fn_mask] - 0.5))]
        cases["borderline_FN"] = idx

    return cases, pred


def fig18_ig_class_averaged_heatmap(ig_pos, ig_neg):
    """Class-averaged IG heatmaps (PE- and PE+ side by side)."""
    print("Figure 18: ig_class_averaged_heatmap.png")
    t = np.arange(2500) / 250.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={"hspace": 0.25, "top": 0.93, "bottom": 0.08})

    vmax = max(np.abs(ig_pos).max(), np.abs(ig_neg).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im0 = axes[0].imshow(ig_neg, aspect="auto", cmap="RdBu_r", norm=norm,
                         extent=[0, 10, -0.5, 11.5], interpolation="bilinear")
    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels(LEAD_NAMES, fontsize=7)
    axes[0].set_title("Normal (PE-) -- Mean IG Attribution", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("Time (s)", fontsize=8)

    im1 = axes[1].imshow(ig_pos, aspect="auto", cmap="RdBu_r", norm=norm,
                         extent=[0, 10, -0.5, 11.5], interpolation="bilinear")
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(LEAD_NAMES, fontsize=7)
    axes[1].set_title("Preeclampsia (PE+) -- Mean IG Attribution", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Time (s)", fontsize=8)

    cbar = fig.colorbar(im1, ax=axes.tolist(), location="right",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Mean IG Attribution", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Class-Averaged Integrated Gradients (all test samples)",
                 fontsize=11, fontweight="bold")
    savefig(fig, "ig_class_averaged_heatmap.png")


def fig19_ig_per_lead_bars(ig_pos, ig_neg):
    """Per-lead IG magnitude comparison PE+ vs PE-."""
    print("Figure 19: ig_per_lead_bars.png")
    lead_ig_pos = np.abs(ig_pos).mean(axis=1)
    lead_ig_neg = np.abs(ig_neg).mean(axis=1)
    diff = lead_ig_pos - lead_ig_neg

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"wspace": 0.35, "left": 0.06, "right": 0.96})

    x_pos = np.arange(12)
    w = 0.35
    axes[0].bar(x_pos - w / 2, lead_ig_neg, w, label="Normal (PE-)",
                color="steelblue", alpha=0.8)
    axes[0].bar(x_pos + w / 2, lead_ig_pos, w, label="Preeclampsia (PE+)",
                color="firebrick", alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(LEAD_NAMES, fontsize=7)
    axes[0].set_ylabel("Mean |IG Attribution|", fontsize=8)
    axes[0].set_title("Per-Lead IG Magnitude by Class", fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=7)

    order = np.argsort(diff)
    colors = ["firebrick" if d > 0 else "steelblue" for d in diff[order]]
    axes[1].barh(range(12), diff[order], color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels([LEAD_NAMES[i] for i in order], fontsize=7)
    axes[1].set_xlabel("Delta |IG| (PE+ minus PE-)", fontsize=8)
    axes[1].set_title("Lead Importance Difference", fontsize=9, fontweight="bold")
    axes[1].axvline(0, color="black", linewidth=0.5)

    fig.suptitle("Which leads does the model focus on more for PE+ vs Normal?",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "ig_per_lead_bars.png")


def fig20_ig_temporal_profile(ig_pos, ig_neg):
    """Temporal saliency profile aggregated across leads."""
    print("Figure 20: ig_temporal_profile.png")
    t = np.arange(2500) / 250.0
    temporal_pos = np.abs(ig_pos).mean(axis=0)
    temporal_neg = np.abs(ig_neg).mean(axis=0)

    temporal_pos_s = gaussian_filter1d(temporal_pos, sigma=15)
    temporal_neg_s = gaussian_filter1d(temporal_neg, sigma=15)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(t, temporal_pos_s, alpha=0.3, color="firebrick", label="PE+ (mean |IG|)")
    ax.fill_between(t, temporal_neg_s, alpha=0.3, color="steelblue", label="PE- (mean |IG|)")
    ax.plot(t, temporal_pos_s, color="firebrick", linewidth=1.2)
    ax.plot(t, temporal_neg_s, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Mean |IG| (across leads)", fontsize=9)
    ax.set_title("Temporal Saliency Profile -- Where does the model look in the ECG?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "ig_temporal_profile.png")


def compute_all_ig(model, X_test, y_test):
    """Compute class-averaged IG across all test samples."""
    print("  Computing IG for all test samples...")
    ig_all = []
    for i in range(len(X_test)):
        ig_i = integrated_gradients(model, X_test[i], target_class=1,
                                     n_steps=32, smooth_sigma=5.0)
        ig_all.append(ig_i)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(X_test)}")
    ig_all = np.array(ig_all)
    ig_pos = ig_all[y_test == 1].mean(axis=0)
    ig_neg = ig_all[y_test == 0].mean(axis=0)
    print(f"  Done. PE+: {(y_test == 1).sum()}, PE-: {(y_test == 0).sum()}")
    return ig_pos, ig_neg


# ===================================================================
# PART H: 8-case saliency gallery
# ===================================================================

def fig21_saliency_gallery(model, X_test, y_test, probs, cases, pred):
    """IG+GradCAM for 8 representative cases."""
    print("Figures 21: 8-case saliency gallery")
    model.eval()
    for case_name, idx in cases.items():
        savename = f"saliency_{case_name}.png"
        print(f"  {savename} (idx={idx})...")
        attr, _, _ = combined_saliency(model, X_test[idx])
        label = "PE+" if y_test[idx] == 1 else "PE-"
        correct = "correct" if pred[idx] == int(y_test[idx]) else "WRONG"
        title = (f'{case_name.replace("_", " ")} -- P(PE)={probs[idx]:.3f} | '
                 f"True={label} | {correct} | IG + Grad-CAM")
        plot_saliency_ecg(X_test[idx], attr, probs[idx], title, savename)


# ===================================================================
# PART I: Activation propagation
# ===================================================================

def fig22_activation_flow(model, X_test, y_test, probs, cases):
    """Layer-by-layer activation maps through all stages for a single lead."""
    print("Figure 22: activation_flow.png")
    TRACK_LEAD = 1  # Lead II
    if "confident_TP" not in cases:
        print("  [SKIP] No confident TP case")
        return
    TRACK_IDX = cases["confident_TP"]

    flow_acts = {}
    def cap(name):
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                output = output[0]
            flow_acts[name] = output.detach().cpu()
        return hook

    handles = [
        model.stage1_conv.register_forward_hook(cap("s1_conv")),
        model.stage1_attn.register_forward_hook(cap("s1_attn")),
    ]
    for i, stage in enumerate(model.later_stages):
        handles.append(stage["conv"].register_forward_hook(cap(f"s{i + 2}_conv")))
        handles.append(stage["attn"].register_forward_hook(cap(f"s{i + 2}_attn")))

    with torch.no_grad():
        x_in = torch.tensor(X_test[TRACK_IDX:TRACK_IDX + 1],
                             dtype=torch.float32).to(DEVICE)
        _ = model(x_in)

    for h in handles:
        h.remove()

    ecg_lead = X_test[TRACK_IDX, TRACK_LEAD, :]
    t_input = np.arange(2500) / 250.0

    # Build stage keys dynamically based on what was captured
    stage_keys = []
    for key in sorted(flow_acts.keys()):
        stage_num = key[1]  # e.g., '1', '2', etc.
        layer_type = key.split("_")[1]  # 'conv' or 'attn'
        if layer_type == "conv":
            if stage_num == "1":
                label = f"Stage 1: MultiScale DSConv"
            else:
                label = f"Stage {stage_num}: SEPerLead"
        else:
            label = f"Stage {stage_num}: CrossLeadAttn output"
        shape = tuple(flow_acts[key].shape[2:])
        stage_keys.append((f"{label} -- shape {shape}", key))

    n_rows = 1 + len(stage_keys)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows + 1),
                             gridspec_kw={"hspace": 0.3})

    # Input ECG
    axes[0].plot(t_input, ecg_lead, "k-", linewidth=1.0)
    axes[0].set_xlim(0, 10)
    axes[0].set_ylabel("mV", fontsize=7)
    axes[0].set_title(f"Input ECG  Lead {LEAD_NAMES[TRACK_LEAD]}  (2500 samples = 10 s)",
                      fontsize=8)
    axes[0].tick_params(labelsize=6)

    for r, (label, key) in enumerate(stage_keys):
        ax = axes[r + 1]
        a = flow_acts[key][0, TRACK_LEAD].numpy()
        C, T_stage = a.shape
        t_stage = np.linspace(0, 10.0, T_stage)
        cmax = float(np.abs(a).max()) or 1.0

        im = ax.imshow(a, aspect="auto", cmap="RdBu_r",
                       extent=[0, 10, -0.5, C - 0.5],
                       vmin=-cmax, vmax=cmax, interpolation="nearest")
        ax.set_ylabel("ch", fontsize=7)
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=5)

    axes[-1].set_xlabel("time (s)", fontsize=7)

    fig.suptitle(
        f"Layer-by-layer activations | idx {TRACK_IDX} (confident TP) | "
        f"lead {LEAD_NAMES[TRACK_LEAD]} | P(PE)={probs[TRACK_IDX]:.3f}\n"
        f"Each row = activation map after that layer (channel on y, time on x). "
        f"Diverging colormap (red = positive, blue = negative).",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, "activation_flow.png")


# ===================================================================
# PART J: Embedding analysis
# ===================================================================

def extract_embeddings(model, X_test):
    """Extract post-lead-pool embeddings for all test samples."""
    print("  Extracting embeddings...")
    embeddings_store = []

    def embed_hook(_m, _i, output):
        embeddings_store.append(output.detach().cpu())

    h = model.lead_pool.register_forward_hook(embed_hook)
    with torch.no_grad():
        batch_size = 64
        for start in range(0, len(X_test), batch_size):
            xb = torch.tensor(X_test[start:start + batch_size],
                               dtype=torch.float32).to(DEVICE)
            _ = model(xb)
    h.remove()
    all_embeddings = torch.cat(embeddings_store, dim=0).numpy()
    print(f"  Embeddings shape: {all_embeddings.shape}")

    W = model.fc.weight.detach().cpu().numpy()  # (2, 64)
    b = model.fc.bias.detach().cpu().numpy()     # (2,)
    vote_weights = W[1] - W[0]                  # (64,)

    return all_embeddings, vote_weights, b


def fig23_vote_weights_structural(vote_weights, b):
    """FC vote weight bar chart (structural)."""
    print("Figure 23: vote_weights_structural.png")
    sort_idx = np.argsort(vote_weights)
    colors_w = ["tomato" if v > 0 else "steelblue" for v in vote_weights[sort_idx]]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(np.arange(len(vote_weights)), vote_weights[sort_idx],
           color=colors_w, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Channel (sorted)")
    ax.set_ylabel("Vote weight")
    ax.set_title(f"Structural vote weights W[PE,i] - W[Normal,i] (sorted)\n"
                 f"Red = wired toward PE, Blue = toward Normal. "
                 f"Bias margin = {b[1] - b[0]:+.3f}")
    fig.tight_layout()
    savefig(fig, "vote_weights_structural.png")


def fig24_votes_tp_vs_tn(all_embeddings, vote_weights, b, probs, cases):
    """Per-channel votes for confident TP vs TN."""
    print("Figure 24: votes_tp_vs_tn.png")
    if "confident_TP" not in cases or "confident_TN" not in cases:
        print("  [SKIP] Missing TP or TN case")
        return

    tp_idx = cases["confident_TP"]
    tn_idx = cases["confident_TN"]
    votes_tp = all_embeddings[tp_idx] * vote_weights
    votes_tn = all_embeddings[tn_idx] * vote_weights
    sort_abs = np.argsort(-np.abs(votes_tp))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2),
                                    gridspec_kw={"hspace": 0.3})

    for ax, votes, label, idx in [
        (ax1, votes_tp, "TP", tp_idx),
        (ax2, votes_tn, "TN", tn_idx),
    ]:
        colors = ["tomato" if v > 0 else "steelblue" for v in votes[sort_abs]]
        ax.bar(np.arange(len(votes)), votes[sort_abs],
               color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Vote", fontsize=7)
        margin = votes.sum() + b[1] - b[0]
        ax.set_title(f"Confident {label} idx={idx} P(PE)={probs[idx]:.3f} "
                     f"(margin={margin:+.3f})", fontsize=8)
        ax.tick_params(labelsize=5)

    ax2.set_xlabel("Channel (sorted by |TP vote|)", fontsize=7)
    fig.suptitle("Per-channel votes: embed[i] * (W[PE,i] - W[Normal,i])\n"
                 "Red = pushes toward PE, Blue = toward Normal",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "votes_tp_vs_tn.png")


def fig25_votes_population_mean(all_embeddings, vote_weights, y_test, b):
    """Population-level mean votes by class."""
    print("Figure 25: votes_population_mean.png")
    all_votes = all_embeddings * vote_weights[None, :]
    mean_votes_pe = all_votes[y_test == 1].mean(axis=0)
    mean_votes_norm = all_votes[y_test == 0].mean(axis=0)
    diff_order = np.argsort(-(mean_votes_pe - mean_votes_norm))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2),
                                    gridspec_kw={"hspace": 0.3})

    for ax, mv, label, n in [
        (ax1, mean_votes_pe, "PE", int(y_test.sum())),
        (ax2, mean_votes_norm, "Normal", int((y_test == 0).sum())),
    ]:
        colors = ["tomato" if v > 0 else "steelblue" for v in mv[diff_order]]
        ax.bar(np.arange(len(mv)), mv[diff_order],
               color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Mean vote", fontsize=7)
        ax.set_title(f"Mean vote -- {label} class (N={n})", fontsize=8)
        ax.tick_params(labelsize=5)

    ax2.set_xlabel("Channel (sorted by PE-Normal diff)", fontsize=7)
    fig.suptitle("Population mean votes per class\n"
                 "Sorted by differential PE-Normal signal",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "votes_population_mean.png")


def fig26_pca_class_and_boundary(all_embeddings, vote_weights, b, y_test, probs):
    """PCA colored by class with decision boundary contour."""
    print("Figure 26: pca_class_and_boundary.png")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(all_embeddings)

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb_scaled)

    x_min, x_max = emb_pca[:, 0].min() - 1, emb_pca[:, 0].max() + 1
    y_min, y_max = emb_pca[:, 1].min() - 1, emb_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = pca.inverse_transform(grid_pca)
    grid_orig = scaler.inverse_transform(grid_scaled)
    decision = (grid_orig @ vote_weights + (b[1] - b[0])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(xx, yy, decision, levels=50, cmap="RdBu_r", alpha=0.3)
    ax.contour(xx, yy, decision, levels=[0], colors="black", linewidths=2.5)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("PE margin", fontsize=8)

    for label, color, name in [(0, "steelblue", "Normal"), (1, "tomato", "PE+")]:
        mask = y_test == label
        ax.scatter(emb_pca[mask, 0], emb_pca[mask, 1],
                   s=20, c=color, alpha=0.7, label=name,
                   edgecolors="white", linewidths=0.5)

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% var)")
    ax.set_title(f"PCA of Learned Embeddings (64-d -> 2-d)\n"
                 f"Explained variance: {100 * evr.sum():.1f}% | "
                 f"Black line = decision boundary (PE margin = 0)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "pca_class_and_boundary.png")
    return emb_pca, emb_scaled, pca, scaler


def fig27_pca_scree(emb_scaled):
    """PCA scree plot + cumulative variance."""
    print("Figure 27: pca_scree.png")
    pca_full = PCA().fit(emb_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = len(pca_full.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(range(1, n_comp + 1), pca_full.explained_variance_ratio_,
            color="steelblue", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("Scree Plot")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.plot(range(1, n_comp + 1), cumvar, "o-", color="steelblue", markersize=4)
    ax2.axhline(0.90, color="red", linestyle="--", linewidth=1, label="90% variance")
    ax2.axhline(0.95, color="orange", linestyle=":", linewidth=1, label="95% variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("Cumulative Variance")
    ax2.legend(fontsize=7)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f"PCA Scree Analysis: {n_90} PCs for 90%, {n_95} PCs for 95% variance",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "pca_scree.png")


def compute_tsne(emb_scaled, perplexity=30):
    """Run t-SNE."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="pca", max_iter=1000)
    return tsne.fit_transform(emb_scaled)


def fig28_tsne_with_boundary(emb_tsne, emb_scaled, all_embeddings,
                              vote_weights, b, y_test, probs):
    """t-SNE with decision margin contour + class labels."""
    print("Figure 28: tsne_with_boundary.png")
    margins = all_embeddings @ vote_weights + (b[1] - b[0])

    x_min, x_max = emb_tsne[:, 0].min() - 2, emb_tsne[:, 0].max() + 2
    y_min, y_max = emb_tsne[:, 1].min() - 2, emb_tsne[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    margin_grid = griddata(emb_tsne, margins, grid_points, method="nearest")
    margin_grid = margin_grid.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(xx, yy, margin_grid, levels=50, cmap="RdBu_r", alpha=0.25)
    ax.contour(xx, yy, margin_grid, levels=[0], colors="black", linewidths=2.5)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("PE margin", fontsize=8)

    for label, color, name in [(0, "steelblue", "Normal"), (1, "tomato", "PE+")]:
        mask = y_test == label
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                   s=25, c=color, alpha=0.75, label=name,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of Learned Embeddings (perplexity=30)\n"
                 "Black contour = decision boundary (PE margin=0) | "
                 "Background = nearest-neighbor interpolated margin")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "tsne_with_boundary.png")


def fig29_tsne_perplexity_comparison(emb_scaled, emb_tsne_30, y_test):
    """t-SNE at perplexity 10/30/50."""
    print("Figure 29: tsne_perplexity_comparison.png")
    perplexities = [10, 30, 50]
    embeddings_list = []
    for perp in perplexities:
        if perp == 30:
            embeddings_list.append(emb_tsne_30)
        else:
            embeddings_list.append(compute_tsne(emb_scaled, perplexity=perp))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for col, (perp, emb_p) in enumerate(zip(perplexities, embeddings_list)):
        ax = axes[col]
        for label, color, name in [(0, "steelblue", "Normal"), (1, "tomato", "PE+")]:
            mask = y_test == label
            ax.scatter(emb_p[mask, 0], emb_p[mask, 1],
                       s=15, c=color, alpha=0.6, label=name if col == 0 else None,
                       edgecolors="white", linewidths=0.3)
        ax.set_title(f"Perplexity={perp}", fontsize=9)
        ax.set_xlabel("t-SNE 1", fontsize=7)
        ax.set_ylabel("t-SNE 2", fontsize=7)
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle("t-SNE Robustness Check: Multiple Perplexities",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "tsne_perplexity_comparison.png")


def fig30_tsne_label_vs_prob(emb_tsne, y_test, probs):
    """t-SNE colored by true label vs predicted probability."""
    print("Figure 30: tsne_label_vs_prob.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: true label
    for label, color, name in [(0, "steelblue", "Normal"), (1, "tomato", "PE+")]:
        mask = y_test == label
        ax1.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                    s=20, c=color, alpha=0.7, label=name,
                    edgecolors="white", linewidths=0.5)
    ax1.set_title("Colored by True Label", fontsize=9)
    ax1.set_xlabel("t-SNE 1", fontsize=7)
    ax1.set_ylabel("t-SNE 2", fontsize=7)
    ax1.legend(fontsize=7)

    # Right: predicted probability
    sc = ax2.scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                     c=probs, cmap="RdBu_r", s=20, alpha=0.8,
                     edgecolors="white", linewidths=0.5, vmin=0, vmax=1)
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("P(PE)", fontsize=8)
    ax2.set_title("Colored by Predicted P(PE)", fontsize=9)
    ax2.set_xlabel("t-SNE 1", fontsize=7)
    ax2.set_ylabel("t-SNE 2", fontsize=7)

    fig.suptitle("t-SNE: True Labels vs Predicted Probabilities\n"
                 "Right panel shows continuous P(PE) gradient -- smooth transitions indicate "
                 "the model has learned a meaningful manifold",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "tsne_label_vs_prob.png")


def fig31_tsne_outcome(emb_tsne, y_test, probs):
    """t-SNE colored by TP/TN/FP/FN."""
    print("Figure 31: tsne_outcome.png")
    pred = (probs >= 0.5).astype(int)
    outcome = np.full(len(y_test), "", dtype=object)
    outcome[(y_test == 0) & (pred == 0)] = "TN"
    outcome[(y_test == 0) & (pred == 1)] = "FP"
    outcome[(y_test == 1) & (pred == 0)] = "FN"
    outcome[(y_test == 1) & (pred == 1)] = "TP"

    colors_map = {"TP": "#2ecc71", "TN": "steelblue", "FP": "#e67e22", "FN": "#e74c3c"}
    size_map = {"TP": 30, "TN": 15, "FP": 45, "FN": 55}
    marker_map = {"TP": "o", "TN": "o", "FP": "X", "FN": "X"}
    alpha_map = {"TP": 0.85, "TN": 0.5, "FP": 0.85, "FN": 0.85}

    fig, ax = plt.subplots(figsize=(9, 7))

    for cat in ["TN", "TP", "FP", "FN"]:
        mask = outcome == cat
        count = mask.sum()
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                   s=size_map[cat], c=colors_map[cat], alpha=alpha_map[cat],
                   marker=marker_map[cat], label=f"{cat} (N={count})",
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE: Misclassification Map (tau=0.50)\n"
                 "FP (orange X) and FN (red X) highlighted. "
                 "Clustering of errors suggests systematic failure modes.")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "tsne_outcome.png")


# ===================================================================
# Main execution
# ===================================================================

def main():
    print("=" * 70)
    print("  RepNet-SE Figure Export Script")
    print(f"  Output: {FIG_DIR.relative_to(REPO_ROOT)}/")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Create output directory
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PART A: Statistical summary figures (no model needed)
    # ------------------------------------------------------------------
    print("\n--- Part A: Statistical summary figures ---")
    results, aurocs, auprcs, seeds = load_results()

    fig01_perseed_auroc_bar(aurocs, seeds)
    fig02_auroc_auprc_histograms(aurocs, auprcs)
    fig03_auroc_boxplot(aurocs)
    fig04_auroc_vs_auprc_scatter(aurocs, auprcs, seeds)
    fig05_repnet_vs_lgbm_boxplot(aurocs)

    # ------------------------------------------------------------------
    # PART B: Load model and data
    # ------------------------------------------------------------------
    print("\n--- Part B: Load model and data ---")
    model, X_test, y_test, probs, x_t, test_auroc, best_seed = \
        load_model_and_data(results, aurocs, seeds)

    # ------------------------------------------------------------------
    # PART C: Best-seed model figures
    # ------------------------------------------------------------------
    print("\n--- Part C: Best-seed model figures ---")
    fig06_pred_histogram(y_test, probs, test_auroc)
    fpr, tpr = fig07_roc_pr_curves(y_test, probs, test_auroc)
    fig08_threshold_sweep(y_test, probs, fpr, tpr)
    fig09_calibration_curve(y_test, probs, test_auroc)

    # ------------------------------------------------------------------
    # PART D: Attention / interpretability
    # ------------------------------------------------------------------
    print("\n--- Part D: Attention & interpretability ---")
    lead_weights = fig10_lead_attention_bars(model, x_t, y_test)
    fig11_lead_attention_heatmap(lead_weights, probs)
    fig12_crosslead_attention_stages(model, x_t)
    fig13_crosslead_attention_diff(model, x_t, y_test)

    # ------------------------------------------------------------------
    # PART E: Saliency maps (gradient-based)
    # ------------------------------------------------------------------
    print("\n--- Part E: Gradient saliency ---")
    saliency, sal_neg, sal_pos = fig14_saliency_heatmap(model, X_test, y_test)
    fig15_saliency_per_lead_bars(sal_neg, sal_pos)
    fig16_saliency_overlay_ecg(X_test, y_test, probs, saliency)

    # ------------------------------------------------------------------
    # PART F: SE channel analysis
    # ------------------------------------------------------------------
    print("\n--- Part F: SE channel analysis ---")
    fig17_se_weights_by_class(model, x_t, y_test)

    # ------------------------------------------------------------------
    # PART G: Integrated Gradients (class-averaged)
    # ------------------------------------------------------------------
    print("\n--- Part G: Integrated Gradients ---")
    ig_pos, ig_neg = compute_all_ig(model, X_test, y_test)
    fig18_ig_class_averaged_heatmap(ig_pos, ig_neg)
    fig19_ig_per_lead_bars(ig_pos, ig_neg)
    fig20_ig_temporal_profile(ig_pos, ig_neg)

    # ------------------------------------------------------------------
    # PART H: 8-case saliency gallery
    # ------------------------------------------------------------------
    print("\n--- Part H: 8-case saliency gallery ---")
    cases, pred = select_cases(y_test, probs)
    print("Selected cases:")
    for label, idx in cases.items():
        print(f"  {label:<18}  idx={idx:>4}  true={int(y_test[idx])}  "
              f"P(PE)={probs[idx]:.3f}")
    fig21_saliency_gallery(model, X_test, y_test, probs, cases, pred)

    # ------------------------------------------------------------------
    # PART I: Activation propagation
    # ------------------------------------------------------------------
    print("\n--- Part I: Activation propagation ---")
    fig22_activation_flow(model, X_test, y_test, probs, cases)

    # ------------------------------------------------------------------
    # PART J: Embedding analysis
    # ------------------------------------------------------------------
    print("\n--- Part J: Embedding analysis ---")
    all_embeddings, vote_weights, b = extract_embeddings(model, X_test)
    fig23_vote_weights_structural(vote_weights, b)
    fig24_votes_tp_vs_tn(all_embeddings, vote_weights, b, probs, cases)
    fig25_votes_population_mean(all_embeddings, vote_weights, y_test, b)

    emb_pca, emb_scaled, pca, scaler = fig26_pca_class_and_boundary(
        all_embeddings, vote_weights, b, y_test, probs)
    fig27_pca_scree(emb_scaled)

    print("\n  Computing t-SNE (perplexity=30)...")
    emb_tsne = compute_tsne(emb_scaled, perplexity=30)
    print("  t-SNE done.")

    fig28_tsne_with_boundary(emb_tsne, emb_scaled, all_embeddings,
                             vote_weights, b, y_test, probs)
    fig29_tsne_perplexity_comparison(emb_scaled, emb_tsne, y_test)
    fig30_tsne_label_vs_prob(emb_tsne, y_test, probs)
    fig31_tsne_outcome(emb_tsne, y_test, probs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_files = len(list(FIG_DIR.glob("*.png")))
    print(f"\n{'=' * 70}")
    print(f"  Done! {n_files} figures saved to {FIG_DIR.relative_to(REPO_ROOT)}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
