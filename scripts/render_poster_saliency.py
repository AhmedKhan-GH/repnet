"""Render poster Lead-II saliency maps (TP + TN) for the current PerLeadCNN.

Picks a confident true-positive and true-negative from the best split (17) test
set, computes combined Integrated-Gradients + Grad-CAM attribution, and renders
the stacked Lead-II figure used in the poster interpretability block.

Reuses the verified data/model code from render_poster_results.py.

Run:  .venv/bin/python scripts/render_poster_saliency.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from render_poster_results import (
    load_ecg_data, preprocess, PerLeadCNN, DATA_DIR, RESULTS_DIR, OUT_DIR, BEST_SPLIT,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
LEAD_IDX = 1  # Lead II
FS = 250.0    # after downsampling


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
    """Grad-CAM on PerLeadCNN's last backbone conv (per-lead activations)."""
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,12,T)
    acts = {}

    def fwd_hook(_m, _i, o):
        acts["feat"] = o
        o.retain_grad()

    last_conv = [m for m in model.backbone if isinstance(m, nn.Conv1d)][-1]
    h = last_conv.register_forward_hook(fwd_hook)
    model.zero_grad()
    logits = model(x)
    logits[0, target_class].backward()
    h.remove()

    feat = acts["feat"]               # (B*L, C, T') = (12, 48, T')
    grad = feat.grad                  # (12, 48, T')
    weights = grad.mean(dim=-1, keepdim=True)        # (12, 48, 1)
    cam = F.relu((weights * feat).sum(dim=1))        # (12, T')
    cam = F.interpolate(cam.unsqueeze(0), size=x_np.shape[-1],
                        mode="linear", align_corners=False)  # (1, 12, T)
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
    # ---- data + model for best split ----
    X_all, y_all, patient_ids = load_ecg_data(DATA_DIR)
    X_all = preprocess(X_all)[:, :, ::2]
    split_seed = BEST_SPLIT * 7 + 1000
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    _, test_idx = next(iter(sgkf.split(X_all, y_all, groups=patient_ids)))
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    model = PerLeadCNN(filters=(16, 32, 48), kernels=(31, 21, 11), dropout=0.15).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X_te, dtype=torch.float32, device=DEVICE)),
                              dim=1)[:, 1].cpu().numpy()
    pred = (probs >= 0.5).astype(int)

    # ---- pick confident TP ----
    tp_mask = (y_te == 1) & (pred == 1)
    tn_mask = (y_te == 0) & (pred == 0)
    tp_idx = np.where(tp_mask)[0][np.argmax(probs[tp_mask])]
    attr_tp = combined_saliency(model, X_te[tp_idx], target_class=1)
    print(f"TP: idx={tp_idx}, P(PE)={probs[tp_idx]:.3f}")

    # ---- pick a TN with CONSISTENT blue bands across the whole trace.
    #      Score = blueness of the weakest time-window (maximin), so every
    #      stretch of the trace must carry some blue. TN_RANK picks among the
    #      best (0 = most consistent). ----
    N_TN_SEARCH = 40
    N_WINDOWS = 10
    TN_RANK = 1

    def blue_consistency(attr):
        lead = attr[LEAD_IDX] / (np.abs(attr).max() or 1.0)
        blue = np.clip(-log_scale_attr(lead), 0, None)  # blue magnitude as displayed
        return min(w.mean() for w in np.array_split(blue, N_WINDOWS))

    tn_candidates = np.where(tn_mask)[0]
    tn_candidates = tn_candidates[np.argsort(probs[tn_candidates])[:N_TN_SEARCH]]
    scored = sorted(
        ((blue_consistency(combined_saliency(model, X_te[c], target_class=1)), c)
         for c in tn_candidates),
        key=lambda s: s[0], reverse=True,  # higher = more consistent blue
    )
    best_score, tn_idx = scored[TN_RANK]
    attr_tn = combined_saliency(model, X_te[tn_idx], target_class=1)
    print(f"TN (consistent-blue rank {TN_RANK} of {N_TN_SEARCH}): idx={tn_idx}, "
          f"P(PE)={probs[tn_idx]:.3f}, weakest-window blue={best_score:.4f}")

    # ---- plot (mirrors old export_poster_saliency.py) ----
    t = np.arange(X_te.shape[2]) / FS
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 3.5),
                                   gridspec_kw={"hspace": 0.35, "top": 0.88,
                                                "bottom": 0.12, "left": 0.05, "right": 0.90})
    for ax, ecg, attr, label, prob in [
        (ax1, X_te[tp_idx], attr_tp, "True Positive (PE+)", probs[tp_idx]),
        (ax2, X_te[tn_idx], attr_tn, "True Negative (Normal)", probs[tn_idx]),
    ]:
        attr_log = log_scale_attr(attr[LEAD_IDX] / (np.abs(attr).max() or 1.0))
        ax.imshow(attr_log[np.newaxis, :], aspect="auto", cmap="RdBu_r", norm=norm,
                  extent=[0, t[-1], -2.5, 2.5], alpha=0.65, interpolation="bilinear")
        ax.plot(t, ecg[LEAD_IDX], "k-", linewidth=0.7)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks([])
        ax.set_ylabel("II", fontsize=10, fontweight="bold", rotation=0, labelpad=14, va="center")
        ax.set_title(f"{label}  —  P(PE) = {prob:.3f}", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=7)
    ax2.set_xlabel("Time (s)", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], location="right", fraction=0.015, pad=0.02)
    cbar.set_label("Attribution (Red → PE+, Blue → Normal)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Lead II Saliency — IG + Grad-CAM (PerLeadCNN)", fontsize=12, fontweight="bold")

    out = os.path.join(OUT_DIR, "saliency_poster_lead_II.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
