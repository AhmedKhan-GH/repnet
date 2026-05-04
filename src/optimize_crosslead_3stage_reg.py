"""Tier-1 anti-overfitting Optuna study for RepNet CrossLead Deeper (3-stage).

Motivation
----------
Previous studies fixed lr+dropout (best: lr=8.76e-4, dropout=0.0636) but the
3-stage model still overfits — train AUROC ≈ 0.857 vs test AUROC ≈ 0.684
(generalization gap +0.17). lr/dropout are settled; the next lever is
*regularization strength*.

Search space (Tier 1)
---------------------
  weight_decay  ∈ log[1e-5, 1e-2]   (currently fixed at 1e-4)
  aug_sigma     ∈ [0.01, 0.10]      (GaussianNoise σ, currently 0.02)
  aug_max_shift ∈ [50, 400]         (RandomTimeShift, currently 200)

Fixed
-----
  Architecture:    stage_filters=(32, 64, 128), kernels=(7, 5, 3), n_heads=4
  Optimizer:       lr=8.76e-4, dropout=0.0636
  Loss:            cross_entropy  (training set is 1:1 after augment+undersample)
  Pipeline:        unbalanced data + pos-only aug + MajorityUndersampling 1:1
  Splits:          patient-grouped 80/20 holdout + 5-fold patient-grouped CV
  Batch / epochs:  64 / 50

Outputs (optuna_results/optuna_crosslead_3stage_reg/YYYY-MM-DD_HH-MM-SS/):
  study.db, results.log, best_params.json, best_model.pt, summary.txt
  {contour,optimization_history,parallel_coordinate,param_importances,slice}_plot.html

Usage:
    python -m src.optimize_crosslead_3stage_reg --n-trials 30
    python -m src.optimize_crosslead_3stage_reg --n-trials 50 --n-folds 5
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

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
from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeperModel
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED = 42

# --- Fixed (settled by previous studies / architecture analysis) ---
# lr / dropout from the recent patient-grouped 3-stage study at
# optuna_results/optuna_crosslead_3stage/2026-04-26_23-12-14 (CV 0.7025, test 0.7540).
FIXED_PARAMS = dict(
    stage_filters = (32, 64, 128),
    kernels       = (7, 5, 3),
    n_heads       = 4,
    batch_size    = 64,
    epochs        = 50,
    loss_fn       = "cross_entropy",
    lr            = 2.465e-3,        # patient-grouped 3-stage best
    dropout       = 0.0546,          # patient-grouped 3-stage best
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


def augment_balance_train(
    X: np.ndarray, y: np.ndarray,
    aug_sigma: float, aug_max_shift: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """3× positives (orig + GaussianNoise + RandomTimeShift) → 1:1 undersample.

    aug_sigma and aug_max_shift parametrize the augmentation strength
    (the search variables for this study).
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X_pos = X[y == 1]
    X_neg = X[y == 0]

    X_pos_g, _ = GaussianNoise(sigma=aug_sigma).transform(X_pos.copy())
    X_pos_t, _ = RandomTimeShift(max_shift=aug_max_shift).transform(X_pos.copy())

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


def objective(
    trial: optuna.Trial,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    weight_decay  = trial.suggest_float("weight_decay",  1e-5, 1e-2, log=True)
    aug_sigma     = trial.suggest_float("aug_sigma",     0.01, 0.10)
    aug_max_shift = trial.suggest_int(  "aug_max_shift", 50,   400)

    model_params = {**FIXED_PARAMS, "weight_decay": weight_decay}

    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr,  y_tr  = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]

        X_tr, y_tr = augment_balance_train(
            X_tr, y_tr,
            aug_sigma=aug_sigma, aug_max_shift=aug_max_shift,
            seed=SEED + fold_idx,
        )

        model = RepNetCrossLeadDeeperModel(**model_params)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)

        logger.info("Trial %d | fold %d/%d | AUROC=%.4f",
                    trial.number, fold_idx + 1, len(folds), auroc)

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc  = float(np.std(fold_aurocs))
    logger.info(
        "Trial %d | wd=%.2e sigma=%.3f shift=%d | AUROC=%.4f (+/-%.4f)",
        trial.number, weight_decay, aug_sigma, aug_max_shift,
        mean_auroc, std_auroc,
    )
    return mean_auroc


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Tier-1 reg search: weight_decay + aug_sigma + aug_max_shift"
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-folds",  type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data/seniordesign_upload")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    run_dir = Path("optuna_results/optuna_crosslead_3stage_reg") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "results.log"),
        ],
    )
    logger.info("Output directory: %s", run_dir)

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

    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="crosslead_3stage_tier1_reg",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds),
        n_trials=args.n_trials,
    )

    best = study.best_params
    best_full = {**FIXED_PARAMS, "weight_decay": best["weight_decay"]}

    with open(run_dir / "best_params.json", "w") as f:
        json.dump({
            "best_trial":     study.best_trial.number,
            "best_cv_auroc":  study.best_value,
            "best_params":    best,
            "fixed_params":   FIXED_PARAMS,
            "model_params":   best_full,
        }, f, indent=2, default=str)

    # --- Final retrain on full dev with patient-grouped 90/10 early-stop ---
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]

    X_tr, y_tr = augment_balance_train(
        X_tr, y_tr,
        aug_sigma=best["aug_sigma"],
        aug_max_shift=best["aug_max_shift"],
        seed=args.seed,
    )

    model = RepNetCrossLeadDeeperModel(**best_full)
    model.fit(X_tr, y_tr, X_es, y_es)

    torch.save(model.model.state_dict(), run_dir / "best_model.pt")
    logger.info("Saved best model weights to %s", run_dir / "best_model.pt")

    proba = model.predict_proba(X_test)
    test_auroc = roc_auc_score(y_test, proba)
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    youden_thresh = float(thresholds[np.argmax(tpr - fpr)])

    lines = [
        "=" * 60,
        "OPTIMIZATION COMPLETE — RepNet CrossLead Deeper: Tier-1 reg",
        "=" * 60,
        f"Best trial: #{study.best_trial.number}",
        f"Best CV AUROC: {study.best_value:.4f}",
        f"Best weight_decay  : {best['weight_decay']:.6f}",
        f"Best aug_sigma     : {best['aug_sigma']:.4f}",
        f"Best aug_max_shift : {best['aug_max_shift']}",
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
            f"  #{trial.number:3d} | wd={trial.params['weight_decay']:.2e} "
            f"sigma={trial.params['aug_sigma']:.3f} shift={trial.params['aug_max_shift']:3d} "
            f"| AUROC={trial.value:.4f}"
        )

    summary = "\n".join(lines)
    with open(run_dir / "summary.txt", "w") as f:
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
