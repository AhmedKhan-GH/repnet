#!/usr/bin/env python3
"""Render poster-friendly saliency maps: lead II from confident TP and TN."""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = REPO_ROOT / "cv_results" / "neural_final_2026-05-20_08-29-02"
FIG_DIR = REPO_ROOT / "publication_media" / "final_report" / "figures"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
POSTER_LEADS = [1]  # lead II


def integrated_gradients(model, x_np, target_class=1, n_steps=50, smooth_sigma=5.0):
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
    ig = integrated_gradients(model, x_np, target_class=target_class)
    gc = gradcam_per_lead(model, x_np, target_class=target_class)
    ig_n = ig / (np.abs(ig).max() or 1.0)
    gc_n = gc / (np.abs(gc).max() or 1.0)
    return 0.5 * (ig_n + gc_n)


def log_scale_attr(a, eps=0.02):
    sign = np.sign(a)
    mag = np.abs(a)
    return sign * np.log1p(mag / eps) / np.log1p(1.0 / eps)


def main():
    import json
    from src.models.repnet_se import RepNetSE
    from src.train_explorer_v2 import load_combined, preprocess_waveforms

    # Load results
    with open(RESULTS_DIR / "results.json") as f:
        results = json.load(f)
    seeds = np.array([r["seed"] for r in results])
    aurocs = np.array([r["metrics"]["auroc"] for r in results])
    best_seed = int(seeds[np.argmax(aurocs)])
    print(f"Best seed: {best_seed}, AUROC: {aurocs.max():.4f}")

    # Load model
    ckpt = torch.load(
        RESULTS_DIR / "weights" / f"best_seed_{best_seed}.pt",
        map_location=DEVICE, weights_only=False,
    )
    model = RepNetSE(**ckpt["net_cfg"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data
    X_wave, X_feat, y, patient_ids, feat_cols = load_combined(
        str(REPO_ROOT / "data" / "seniordesign_upload")
    )
    X_wave = preprocess_waveforms(X_wave)

    ss = np.random.SeedSequence(best_seed)
    split_seed = int(ss.generate_state(1)[0])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=patient_ids))

    X_test = X_wave[test_idx]
    y_test = y[test_idx]

    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    pred = (probs >= 0.5).astype(int)

    # Find confident TP and TN
    tp_mask = (y_test == 1) & (pred == 1)
    tn_mask = (y_test == 0) & (pred == 0)
    tp_idx = np.where(tp_mask)[0][np.argmax(probs[tp_mask])]
    tn_idx = np.where(tn_mask)[0][np.argmin(probs[tn_mask])]

    print(f"TP: idx={tp_idx}, P(PE)={probs[tp_idx]:.3f}")
    print(f"TN: idx={tn_idx}, P(PE)={probs[tn_idx]:.3f}")

    # Compute saliency
    print("Computing saliency for TP...")
    attr_tp = combined_saliency(model, X_test[tp_idx])
    print("Computing saliency for TN...")
    attr_tn = combined_saliency(model, X_test[tn_idx])

    # Plot lead II from each, stacked vertically
    lead_idx = 1  # Lead II
    t = np.arange(X_test.shape[2]) / 250.0
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 3.5),
                                    gridspec_kw={"hspace": 0.35,
                                                 "top": 0.88, "bottom": 0.12,
                                                 "left": 0.05, "right": 0.90})

    for ax, ecg, attr, label, prob in [
        (ax1, X_test[tp_idx], attr_tp, "True Positive (PE+)", probs[tp_idx]),
        (ax2, X_test[tn_idx], attr_tn, "True Negative (Normal)", probs[tn_idx]),
    ]:
        attr_log = log_scale_attr(attr[lead_idx] / (np.abs(attr).max() or 1.0))
        ax.imshow(attr_log[np.newaxis, :], aspect="auto", cmap="RdBu_r",
                  norm=norm, extent=[0, t[-1], -2.5, 2.5], alpha=0.65,
                  interpolation="bilinear")
        ax.plot(t, ecg[lead_idx], "k-", linewidth=0.7)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks([])
        ax.set_ylabel("II", fontsize=10, fontweight="bold", rotation=0,
                      labelpad=14, va="center")
        ax.set_title(f"{label}  —  P(PE) = {prob:.3f}",
                     fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=7)

    ax2.set_xlabel("Time (s)", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], location="right",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Attribution (Red → PE+, Blue → Normal)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Lead II Saliency — IG + Grad-CAM", fontsize=12, fontweight="bold")

    out = FIG_DIR / "saliency_poster_lead_II.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
