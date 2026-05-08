"""Train RepNet CrossLead (2-stage) on the pre-balanced dataset.

Conditions:
  - Dataset:       data/seniordesign_upload_balanced  (~50/50)
  - Quality filter: drops flat leads + missing patient IDs
  - Preprocess:    BWF 0.5 Hz HP + Notch 60 Hz + per-lead Z-score
  - Augmentation:  NONE — training set is shuffled only
  - Balancing:     NONE — data is already balanced
  - Loss:          cross_entropy
  - Splits:        patient-grouped 80/20 holdout + 3-fold patient-grouped CV
  - Params:        lr=8.76e-4, dropout=0.0636 (Optuna best from original study)

This is the apples-to-apples baseline for the leaky original study:
same balanced data + same params, but with patient-grouped splits.

Usage:
    python -m src.train_repnet_crosslead_balanced
    python -m src.train_repnet_crosslead_balanced --lr 1e-3 --dropout 0.1
"""

import argparse
import logging

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_crosslead import RepNetCrossLeadModel
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42

PARAMS = dict(
    stage_filters = (32, 64),
    wide_kernel   = 7,
    narrow_kernel = 5,
    n_heads       = 4,
    batch_size    = 64,
    epochs        = 50,
    loss_fn       = "cross_entropy",
    lr            = 8.76e-4,
    dropout       = 0.0636,
)


def preprocess(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def quality_filter(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
    flat_std_thresh: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    flat_mask = (X.std(axis=2) < flat_std_thresh).any(axis=1)
    n_flat = int(flat_mask.sum())
    keep = ~flat_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]

    try:
        nan_mask = np.isnan(patient_ids.astype(float))
    except (ValueError, TypeError):
        nan_mask = np.array([str(p).strip() in ("", "nan", "None") for p in patient_ids])
    n_missing = int(nan_mask.sum())
    keep = ~nan_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]

    return X, y, patient_ids, {"flat_lead_dropped": n_flat, "missing_id_dropped": n_missing}


def main():
    parser = argparse.ArgumentParser(
        description="Train RepNet CrossLead 2-stage on balanced dataset (patient-grouped, no aug)"
    )
    parser.add_argument("--data-dir", type=str, default="data/seniordesign_upload_balanced")
    parser.add_argument("--lr",       type=float, default=PARAMS["lr"])
    parser.add_argument("--dropout",  type=float, default=PARAMS["dropout"])
    parser.add_argument("--n-folds",  type=int,   default=3)
    parser.add_argument("--seed",     type=int,   default=SEED)
    args = parser.parse_args()

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info(
        "After QC: N=%d  pos=%d  neg=%d  (dropped %d flat-lead, %d missing-ID)",
        len(y), int(y.sum()), int((y == 0).sum()),
        qc["flat_lead_dropped"], qc["missing_id_dropped"],
    )

    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=0.20, seed=args.seed,
    )
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)

    logger.info(
        "Dev: %d (pos=%.1f%%, %d-fold patient-grouped CV)  Test: %d (pos=%.1f%%)",
        len(y_dev), 100 * y_dev.mean(), args.n_folds,
        len(y_test), 100 * y_test.mean(),
    )

    params = {**PARAMS, "lr": args.lr, "dropout": args.dropout}

    # --- Cross-validation ---
    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr,  y_tr  = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]

        rng = np.random.default_rng(args.seed + fold_idx)
        shuffle = rng.permutation(len(y_tr))
        X_tr, y_tr = X_tr[shuffle], y_tr[shuffle]

        model = RepNetCrossLeadModel(**params)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)
        logger.info("Fold %d/%d | AUROC=%.4f", fold_idx + 1, args.n_folds, auroc)

    logger.info(
        "CV AUROC: %.4f (±%.4f)",
        float(np.mean(fold_aurocs)), float(np.std(fold_aurocs)),
    )

    # --- Final retrain on full dev (patient-grouped 90/10 early-stop split) ---
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    model = RepNetCrossLeadModel(**params)
    model.fit(X_tr, y_tr, X_es, y_es)

    # --- Test evaluation ---
    proba = model.predict_proba(X_test)
    test_auroc = roc_auc_score(y_test, proba)
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    youden_thresh = float(thresholds[np.argmax(tpr - fpr)])

    print("=" * 60)
    print("RepNet CrossLead 2-stage | balanced | patient-grouped | no aug")
    print("=" * 60)
    print(f"CV  AUROC : {np.mean(fold_aurocs):.4f} (±{np.std(fold_aurocs):.4f})")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"\nThreshold = 0.50:")
    print(classification_report(y_test, (proba >= 0.50).astype(int),
                                target_names=["No PE", "PE"]))
    print(f"Threshold = {youden_thresh:.3f} (Youden's J):")
    print(classification_report(y_test, (proba >= youden_thresh).astype(int),
                                target_names=["No PE", "PE"]))


if __name__ == "__main__":
    main()
