"""Generate the analysis figures for the released PerLeadCNN model.

Produces (into `figures/`):
  - split_distribution.png : AUROC/AUPRC across the 30 splits (from per_split.json)
  - roc_pr_curves.png      : ROC + precision-recall on the best split
  - prediction_distribution.png : predicted P(PE) for PE+ vs Normal (best split)
  - calibration_curve.png  : reliability diagram (best split)
  - confusion_matrix.png   : confusion matrix at Youden's J (best split)
  - lead_importance.png    : per-lead importance from the fusion-layer weights
  - saliency_leadII.png    : Integrated-Gradients attribution on Lead II (best split)

The first figure needs only the bundled JSON; the rest need the dataset
(PHI, see DATA.md) and the bundled best checkpoint.

Run (from the package root):
    python -m src.analyze
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .data import DEFAULT_DATA_DIR, SD_LEAD_ORDER, load_dataset, test_split
from .model import PerLeadCNN

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PKG_ROOT, "results", "multisplit_dbb6f49")
FIG_DIR = os.path.join(PKG_ROOT, "figures")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def _load_best_model():
    model = PerLeadCNN(filters=(16, 32, 48), kernels=(31, 21, 11), dropout=0.15)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model


# --------------------------------------------------------------------------
# Figures that need only the recorded JSON
# --------------------------------------------------------------------------
def fig_split_distribution():
    with open(os.path.join(RESULTS_DIR, "per_split.json")) as f:
        per_split = json.load(f)
    aurocs = [s["auroc"] for s in per_split]
    auprcs = [s["auprc"] for s in per_split]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, vals, name, color in ((a1, aurocs, "AUROC", "steelblue"),
                                  (a2, auprcs, "AUPRC", "tomato")):
        ax.hist(vals, bins=12, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(vals), color="black", linestyle="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_xlabel(name)
        ax.set_ylabel("# splits")
        ax.set_title(f"{name} over {len(vals)} splits")
        ax.legend()
    fig.suptitle("PerLeadCNN — 30-split patient-grouped distribution",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "split_distribution.png")


def fig_lead_importance():
    """Per-lead importance = L2 norm of the fusion-layer weights for each lead."""
    model = _load_best_model()
    W = model.fc.weight.detach().numpy()           # (2, 576)
    n_classes, total = W.shape
    feat_per_lead = total // len(SD_LEAD_ORDER)     # 48
    Wl = W.reshape(n_classes, len(SD_LEAD_ORDER), feat_per_lead)
    importance = np.linalg.norm(Wl, axis=(0, 2))    # (12,)
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([SD_LEAD_ORDER[i] for i in order], importance[order],
           color="seagreen", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Fusion-weight L2 norm")
    ax.set_xlabel("ECG lead")
    ax.set_title("Per-lead importance (PerLeadCNN fusion layer)")
    fig.tight_layout()
    _save(fig, "lead_importance.png")


# --------------------------------------------------------------------------
# Figures that need the data + best checkpoint
# --------------------------------------------------------------------------
def _best_split_predictions(X, y, groups):
    with open(os.path.join(RESULTS_DIR, "per_split.json")) as f:
        per_split = json.load(f)
    best_i = int(np.argmax([s["auroc"] for s in per_split]))
    test_idx = test_split(best_i, y, groups)
    X_te, y_te = X[test_idx], y[test_idx]
    model = _load_best_model()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X_te, dtype=torch.float32)),
                              dim=1)[:, 1].numpy()
    return best_i, X_te, y_te, probs


def fig_roc_pr(y_te, probs):
    auroc = roc_auc_score(y_te, probs)
    auprc = average_precision_score(y_te, probs)
    brier = brier_score_loss(y_te, probs)
    fpr, tpr, _ = roc_curve(y_te, probs)
    prec, rec, _ = precision_recall_curve(y_te, probs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auroc:.3f}")
    a1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    a1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
    a1.legend(loc="lower right")
    a2.plot(rec, prec, color="tomato", lw=2, label=f"AP={auprc:.3f}")
    a2.axhline(y_te.mean(), ls="--", color="gray", lw=1,
               label=f"prevalence={y_te.mean():.2f}")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    a2.legend(loc="upper right")
    fig.suptitle(f"Best split — ROC & PR (Brier={brier:.3f})", fontweight="bold")
    fig.tight_layout()
    _save(fig, "roc_pr_curves.png")


def fig_prediction_distribution(y_te, probs):
    fpr, tpr, thr = roc_curve(y_te, probs)
    tau = float(thr[np.argmax(tpr - fpr)])
    bins = np.linspace(0, 1, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    h_pe, _ = np.histogram(probs[y_te == 1], bins=bins)
    h_no, _ = np.histogram(probs[y_te == 0], bins=bins)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(centers, h_pe, width=bins[1] - bins[0], color="tomato", alpha=0.85,
           label=f"PE+ (n={int((y_te==1).sum())})")
    ax.bar(centers, -h_no, width=bins[1] - bins[0], color="steelblue", alpha=0.85,
           label=f"Normal (n={int((y_te==0).sum())})")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(tau, color="green", ls=":", lw=2, label=f"Youden tau={tau:.3f}")
    ax.set(xlabel="P(Preeclampsia)", ylabel="Count",
           title="Predicted P(PE): PE+ above, Normal below")
    ax.legend()
    fig.tight_layout()
    _save(fig, "prediction_distribution.png")


def fig_calibration(y_te, probs):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(probs, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(probs[m].mean())
            ys.append(y_te[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "o-", color="purple", label="PerLeadCNN")
    ax.set(xlabel="Mean predicted P(PE)", ylabel="Observed frequency",
           title="Calibration (best split)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "calibration_curve.png")


def fig_confusion(y_te, probs):
    fpr, tpr, thr = roc_curve(y_te, probs)
    tau = thr[np.argmax(tpr - fpr)]
    cm = confusion_matrix(y_te, (probs >= tau).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "PE+"],
           yticklabels=["Normal", "PE+"], xlabel="Predicted", ylabel="True",
           title=f"Confusion @ Youden (tau={tau:.2f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def fig_saliency_leadII(X_te, y_te, n_steps=20):
    """Integrated-Gradients attribution toward the PE class, averaged on Lead II."""
    model = _load_best_model()
    X = torch.tensor(X_te, dtype=torch.float32)
    baseline = torch.zeros_like(X)
    total = torch.zeros_like(X)
    for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
        pt = (baseline + alpha * (X - baseline)).clone().requires_grad_(True)
        logit_pe = model(pt)[:, 1].sum()
        grad, = torch.autograd.grad(logit_pe, pt)
        total += grad
    attr = ((X - baseline) * total / n_steps).detach().numpy()  # (n,12,T)

    lead = SD_LEAD_ORDER.index("II")
    t = np.arange(X.shape[2])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for cls, color, name in ((1, "tomato", "PE+ (true positives)"),
                             (0, "steelblue", "Normal (true negatives)")):
        m = y_te == cls
        if m.sum():
            ax.plot(t, attr[m, lead, :].mean(axis=0), color=color, lw=1.3, label=name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Time (samples @ 250 Hz)", ylabel="IG attribution (Lead II)",
           title="Integrated-Gradients attribution toward PE — Lead II (best split)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "saliency_leadII.png")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating figures...")

    made, failed = [], []

    def run(fn, *args):
        try:
            fn(*args)
            made.append(fn.__name__)
        except Exception as e:  # keep going; report at the end
            failed.append((fn.__name__, repr(e)))
            print(f"  [skip] {fn.__name__}: {e}")

    run(fig_split_distribution)   # JSON only

    if os.path.isdir(os.path.join(DEFAULT_DATA_DIR, "ekg_data")):
        run(fig_lead_importance)
        X, y, groups = load_dataset()
        _, X_te, y_te, probs = _best_split_predictions(X, y, groups)
        run(fig_roc_pr, y_te, probs)
        run(fig_prediction_distribution, y_te, probs)
        run(fig_calibration, y_te, probs)
        run(fig_confusion, y_te, probs)
        run(fig_saliency_leadII, X_te, y_te)
    else:
        print(f"  [info] dataset not found at {DEFAULT_DATA_DIR}; "
              "only JSON-based figures generated.")

    print(f"\nDone. {len(made)} figure(s) written to figures/.")
    if failed:
        print(f"{len(failed)} failed: {[f[0] for f in failed]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
