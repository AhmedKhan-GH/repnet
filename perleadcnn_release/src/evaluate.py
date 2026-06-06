"""Reproduce the released metrics from the bundled checkpoints.

Re-evaluates `best_model.pt` and `median_model.pt` on their exact
patient-grouped test splits and compares against the recorded
`per_split.json` / `summary.json`.

Run (from the package root):
    python -m src.evaluate
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .data import load_dataset, test_split
from .metrics import compute_metrics
from .model import PerLeadCNN

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PKG_ROOT, "results", "multisplit_dbb6f49")

_COMPARE_KEYS = [
    "auroc", "auprc",
    "youden_sens", "youden_spec", "youden_prec", "youden_npv", "youden_f1",
    "sens80_sens", "sens80_spec", "sens80_prec", "sens80_npv", "sens80_f1",
]


@torch.no_grad()
def evaluate_checkpoint_on_split(split_i, model_path, X, y, groups,
                                 filters=(16, 32, 48), kernels=(31, 21, 11),
                                 dropout=0.15) -> dict:
    """Load a checkpoint, evaluate on split `split_i`'s test set, return metrics."""
    test_idx = test_split(split_i, y, groups)
    X_te, y_te = X[test_idx], y[test_idx]

    model = PerLeadCNN(filters=filters, kernels=kernels, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    logits = model(torch.tensor(X_te, dtype=torch.float32))
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return compute_metrics(y_te, probs)


def _load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def main():
    per_split = _load_json("per_split.json")
    summary = _load_json("summary.json")
    aurocs = [s["auroc"] for s in per_split]
    best_i = int(np.argmax(aurocs))
    median_i = int(np.argsort(aurocs)[len(aurocs) // 2])

    print("=" * 64)
    print("PerLeadCNN — reproducing released metrics from checkpoints")
    print("=" * 64)
    print(f"Recorded (30-split): AUROC {summary['auroc_mean']:.4f} "
          f"+/- {summary['auroc_std']:.4f} | "
          f"AUPRC {summary['auprc_mean']:.4f} +/- {summary['auprc_std']:.4f}")
    print(f"Params: {summary['num_params']:,}\n")

    print("Loading dataset (this reads every recording; may take a minute)...")
    X, y, groups = load_dataset()
    print(f"  {len(y)} recordings, {int(y.sum())} positive ({y.mean():.1%}), "
          f"{len(np.unique(groups))} patients, shape {X.shape}\n")

    all_ok = True
    for label, ckpt, split_i in (("BEST", "best_model.pt", best_i),
                                 ("MEDIAN", "median_model.pt", median_i)):
        repro = evaluate_checkpoint_on_split(
            split_i, os.path.join(RESULTS_DIR, ckpt), X, y, groups)
        rec = per_split[split_i]
        print(f"--- {label} model | split {split_i} (seed {split_i*7+1000}) ---")
        print(f"{'metric':<16}{'reproduced':>12}{'recorded':>12}{'match':>8}")
        for k in _COMPARE_KEYS:
            ok = abs(repro[k] - rec[k]) < 1e-4
            all_ok &= ok
            print(f"{k:<16}{repro[k]:>12.4f}{rec[k]:>12.4f}{'OK' if ok else 'FAIL':>8}")
        print()

    print("=" * 64)
    print("ALL CHECKPOINTS REPRODUCE RECORDED METRICS" if all_ok
          else "MISMATCH DETECTED — see rows marked FAIL")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
