"""Reproduce the 5-bag ensemble for seed 45 and compute full threshold metrics.

Retrains all 5 bags with the exact same seeds used in train_neural_final.py,
averages their predictions, and outputs a classification report at tau=0.50
and tau=Youden with sensitivity, specificity, precision, F1, accuracy, etc.

Usage:
    python -m src.reproduce_ensemble_seed45
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.models.repnet_se import RepNetSE
from src.train_neural_final import predict, train_one
from src.train_explorer_v2 import load_combined, preprocess_waveforms

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "cv_results" / "neural_final_2026-05-20_08-29-02"
OUT_PATH = REPO_ROOT / "cv_results" / "neural_final_2026-05-20_08-29-02" / "ensemble_seed45_metrics.json"

MASTER_SEED = 45
N_BAGS = 5

NET_CFG = dict(
    stage_filters=(16, 32, 48, 64),
    stage_kernels=(7, 5, 5, 3),
    ms_kernels=(5, 9, 15),
    dropout=0.15,
    n_heads=4,
    se_reduction=4,
    attn_pool_hidden=32,
)
TRAIN_CFG = dict(
    lr=2e-3,
    weight_decay=5e-3,
    batch_size=64,
    epochs=80,
    patience=20,
    label_smoothing=0.05,
    mixup_alpha=0.2,
    grad_clip=1.0,
    loss="label_smooth",
)


def metrics_at_threshold(y_true, probs, tau):
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    acc = (tp + tn) / len(y_true)
    return dict(
        threshold=float(tau),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(ppv),
        npv=float(npv),
        f1=float(f1),
        accuracy=float(acc),
    )


def main():
    print(f"Loading data...")
    X_wave, _, y, patient_ids, _ = load_combined(
        str(REPO_ROOT / "data" / "seniordesign_upload")
    )
    X_wave = preprocess_waveforms(X_wave)

    ss = np.random.SeedSequence(MASTER_SEED)
    split_seed = int(ss.generate_state(1)[0])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=patient_ids))
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed + 1)
    tr_idx, val_idx = next(sgkf2.split(
        np.zeros(len(y[dev_idx])), y[dev_idx], groups=patient_ids[dev_idx]))

    Xw_tr = X_wave[dev_idx][tr_idx]
    Xw_val = X_wave[dev_idx][val_idx]
    Xw_test = X_wave[test_idx]
    y_tr = y[dev_idx][tr_idx]
    y_val = y[dev_idx][val_idx]
    y_test = y[test_idx]

    print(f"Test set: N={len(y_test)}, PE+={int(y_test.sum())}, PE-={int((y_test == 0).sum())}")
    print(f"Training {N_BAGS} bags for master seed {MASTER_SEED}...\n")

    bag_probs = []
    for bag_i in range(N_BAGS):
        bag_seed = MASTER_SEED + bag_i * 1000
        print(f"  Bag {bag_i + 1}/{N_BAGS} (seed={bag_seed})...")
        model, val_auroc, n_params = train_one(
            Xw_tr, y_tr, Xw_val, y_val, bag_seed, NET_CFG, TRAIN_CFG)
        probs = predict(model, Xw_test)
        bag_probs.append(probs)
        print(f"    val_auroc={val_auroc:.4f}, test_auroc={roc_auc_score(y_test, probs):.4f}")
        del model
        torch.cuda.empty_cache()

    avg_probs = np.mean(bag_probs, axis=0)

    auroc = roc_auc_score(y_test, avg_probs)
    auprc = average_precision_score(y_test, avg_probs)
    brier = brier_score_loss(y_test, avg_probs)

    fpr, tpr, thr = roc_curve(y_test, avg_probs)
    tau_youden = float(thr[np.argmax(tpr - fpr)])
    j_stat = float(np.max(tpr - fpr))

    m50 = metrics_at_threshold(y_test, avg_probs, 0.50)
    my = metrics_at_threshold(y_test, avg_probs, tau_youden)

    output = {
        "master_seed": MASTER_SEED,
        "n_bags": N_BAGS,
        "bag_seeds": [MASTER_SEED + i * 1000 for i in range(N_BAGS)],
        "n_test": int(len(y_test)),
        "n_pos_test": int(y_test.sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "youden_j": j_stat,
        "tau_0.50": m50,
        "tau_youden": my,
        "per_bag_test_aurocs": [float(roc_auc_score(y_test, p)) for p in bag_probs],
    }

    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 65}")
    print(f"  5-Bag Ensemble Results (seed {MASTER_SEED})")
    print(f"{'=' * 65}")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"  Brier:  {brier:.4f}")
    print(f"  Youden: J={j_stat:.3f}, tau={tau_youden:.3f}")
    print()
    print(f"  tau=0.50:    Sens={m50['sensitivity']:.3f}  Spec={m50['specificity']:.3f}  "
          f"Prec={m50['precision']:.3f}  F1={m50['f1']:.3f}  Acc={m50['accuracy']:.3f}")
    print(f"  tau=Youden:  Sens={my['sensitivity']:.3f}  Spec={my['specificity']:.3f}  "
          f"Prec={my['precision']:.3f}  F1={my['f1']:.3f}  Acc={my['accuracy']:.3f}")
    print(f"\n  Saved to: {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
