"""Evaluation metrics — identical definitions to those used for the release.

`compute_metrics` returns a flat dict whose keys match the fields stored in
`results/multisplit_dbb6f49/per_split.json`, so reproduced numbers can be
compared field-by-field.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)

SENS_TARGET = 0.80


def _operating_point(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return dict(sens=sens, spec=spec, prec=prec, npv=npv, acc=acc, f1=f1,
                threshold=float(thr), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def compute_metrics(y_true, probs) -> dict:
    """Threshold-free + two operating points (Youden's J, sensitivity >= 80%)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    out = {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
    }

    fpr, tpr, thresholds = roc_curve(y_true, probs)

    # Youden's J: maximize sensitivity + specificity - 1 = tpr - fpr.
    thr_youden = thresholds[int(np.argmax(tpr - fpr))]

    # Sensitivity-favoring screening point: highest specificity with sens >= 0.80.
    valid = tpr >= SENS_TARGET
    if valid.any():
        cand = np.where(valid)[0]
        thr_sens80 = thresholds[cand[int(np.argmax(1 - fpr[cand]))]]
    else:
        thr_sens80 = thr_youden

    for name, thr in (("youden", thr_youden), ("sens80", thr_sens80)):
        for k, v in _operating_point(y_true, probs, thr).items():
            out[f"{name}_{k}"] = v
    return out
