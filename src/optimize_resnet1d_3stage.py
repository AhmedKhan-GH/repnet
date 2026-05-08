"""Optuna lr + dropout search for the 3-stage ResNet1D ablation.

Same data pipeline as `train_resnet1d_3stage.py` (which itself mirrors the best
CrossLead trainer): patient-grouped 80/20 holdout + 5-fold CV, augment-balance
with sigma=0.060 / max_shift=276, undersample to 1:1, plain cross-entropy.

Fixed: stage_filters=(48,96,192), kernels=(7,5,3), batch_size=64, epochs=50,
weight_decay=1.67e-4. The architecture and weight-decay are matched to the
CrossLead Deeper best so the only thing the search resolves is "what lr/dropout
does this smaller, no-attention variant prefer?"

Search:  lr ∈ [1e-4, 5e-3] log,  dropout ∈ [0.05, 0.4].

Outputs (saved to optuna_resnet1d_3stage/YYYY-MM-DD_HH-MM-SS/):
  - study.db
  - best_params.json
  - best_model.pt
  - summary.txt
  - {optimization_history,parallel_coordinate,slice_plot,contour_plot,param_importances}.html

Usage:
    python -m src.optimize_resnet1d_3stage --n-trials 20
    python -m src.optimize_resnet1d_3stage --n-trials 30 --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.resnet1d_3stage import ResNet1D3StageModel
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED          = 42
AUG_SIGMA     = 0.060      # match train_resnet1d_3stage.py
AUG_MAX_SHIFT = 276

FIXED_PARAMS = dict(
    stage_filters = (48, 96, 192),
    kernels       = (7, 5, 3),
    batch_size    = 64,
    epochs        = 50,
    weight_decay  = 1.67e-4,
    loss_fn       = "cross_entropy",
)


def preprocess(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def quality_filter(X, y, patient_ids, flat_std_thresh=1e-4):
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


def augment_balance_train(X, y, seed=SEED):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X_pos = X[y == 1]
    X_neg = X[y == 0]
    X_pos_g, _ = GaussianNoise(sigma=AUG_SIGMA).transform(X_pos.copy())
    X_pos_t, _ = RandomTimeShift(max_shift=AUG_MAX_SHIFT).transform(X_pos.copy())

    X_aug = np.concatenate([X_neg, X_pos, X_pos_g, X_pos_t], axis=0)
    y_aug = np.concatenate([
        np.zeros(len(X_neg), dtype=y.dtype),
        np.ones(len(X_pos),  dtype=y.dtype),
        np.ones(len(X_pos),  dtype=y.dtype),
        np.ones(len(X_pos),  dtype=y.dtype),
    ], axis=0)

    X_bal, y_bal = MajorityUndersampling(ratio=1.0, seed=seed).transform(X_aug, y_aug)
    np.random.set_state(rng_state)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


def objective(trial, X_dev, y_dev, folds):
    lr      = trial.suggest_float("lr",      1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    params  = {**FIXED_PARAMS, "lr": lr, "dropout": dropout}

    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr,  y_tr  = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]
        X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=SEED + fold_idx)

        model = ResNet1D3StageModel(**params)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)

        logger.info("Trial %d | fold %d/%d | AUROC=%.4f",
                    trial.number, fold_idx + 1, len(folds), auroc)

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc  = float(np.std(fold_aurocs))
    logger.info("Trial %d | lr=%.5f dropout=%.3f | AUROC=%.4f (±%.4f)",
                trial.number, lr, dropout, mean_auroc, std_auroc)
    return mean_auroc


def main():
    parser = argparse.ArgumentParser(
        description="Optuna lr+dropout search for ResNet1D 3-stage."
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-folds",  type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data/seniordesign_upload")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    run_dir = Path("optuna_resnet1d_3stage") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "results.log", encoding="utf-8"),
        ],
    )
    logger.info("Output directory: %s", run_dir)
    logger.info("Fixed params: %s", FIXED_PARAMS)
    logger.info("Augmentation: sigma=%.4f  max_shift=%d", AUG_SIGMA, AUG_MAX_SHIFT)

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d  (dropped %d flat-lead, %d missing-ID)",
                len(y), int(y.sum()), int((y == 0).sum()),
                qc["flat_lead_dropped"], qc["missing_id_dropped"])

    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=0.20, seed=args.seed,
    )
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)
    logger.info("Dev: %d (pos=%.1f%%, %d-fold patient-grouped CV)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(), args.n_folds,
                len(y_test), 100 * y_test.mean())

    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="resnet1d_3stage_lr_dropout",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    # Seed trial 0 with the best CrossLead config (lr=2.465e-3, dropout=0.0546)
    # so the prior best is the immediate baseline.
    if len(study.trials) == 0:
        study.enqueue_trial({"lr": 2.465e-3, "dropout": 0.0546})

    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds),
        n_trials=args.n_trials,
    )

    best_full = {
        **FIXED_PARAMS,
        "lr":      study.best_params["lr"],
        "dropout": study.best_params["dropout"],
    }
    with open(run_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_trial":     study.best_trial.number,
            "best_cv_auroc":  study.best_value,
            "best_params":    study.best_params,
            "fixed_params":   {k: list(v) if isinstance(v, tuple) else v
                               for k, v in FIXED_PARAMS.items()},
            "full_params":    {k: list(v) if isinstance(v, tuple) else v
                               for k, v in best_full.items()},
            "aug_sigma":      AUG_SIGMA,
            "aug_max_shift":  AUG_MAX_SHIFT,
        }, f, indent=2, default=str)

    # Retrain best on full dev → save weights + test eval.
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]
    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=args.seed)

    model = ResNet1D3StageModel(**best_full)
    model.fit(X_tr, y_tr, X_es, y_es)
    torch.save(model.model.state_dict(), run_dir / "best_model.pt")
    logger.info("Saved best model weights to %s", run_dir / "best_model.pt")

    proba = model.predict_proba(X_test)
    test_auroc = roc_auc_score(y_test, proba)
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    youden_thresh = float(thresholds[np.argmax(tpr - fpr)])

    lines = [
        "=" * 60,
        "OPTIMIZATION COMPLETE — ResNet1D 3-stage: lr + dropout",
        "=" * 60,
        f"Best trial: #{study.best_trial.number}",
        f"Best CV AUROC: {study.best_value:.4f}",
        f"Best lr:      {study.best_params['lr']:.6f}",
        f"Best dropout: {study.best_params['dropout']:.4f}",
        "",
        "=" * 60,
        "HOLDOUT TEST",
        "=" * 60,
        f"  AUROC: {test_auroc:.4f}",
        "\n  Threshold = 0.50:",
        classification_report(y_test, (proba >= 0.5).astype(int),
                              target_names=["No PE", "PE"]),
        f"  Threshold = {youden_thresh:.3f} (Youden's J):",
        classification_report(y_test, (proba >= youden_thresh).astype(int),
                              target_names=["No PE", "PE"]),
        "",
        "All trials:",
    ]
    for trial in study.trials:
        if trial.value is None:
            continue
        lines.append(
            f"  #{trial.number:3d} | lr={trial.params['lr']:.6f} "
            f"dropout={trial.params['dropout']:.4f} | AUROC={trial.value:.4f}"
        )

    summary = "\n".join(lines)
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)

    try:
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
        plot_optimization_history(study).write_html(str(run_dir / "optimization_history.html"))
        plot_parallel_coordinate(study).write_html(str(run_dir / "parallel_coordinate.html"))
        plot_slice(study).write_html(str(run_dir / "slice_plot.html"))
        plot_contour(study).write_html(str(run_dir / "contour_plot.html"))
        plot_param_importances(study).write_html(str(run_dir / "param_importances.html"))
        logger.info("Saved Optuna visualization plots to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save Optuna plots: %s", e)


if __name__ == "__main__":
    main()
