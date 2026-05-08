"""Optuna search over lr + dropout + n_tokens for RepNetCrossLeadLargeAttn (3 layers).

Companion to `optimize_crosslead_large_attn.py` (arch search). Here, n_layers is
FIXED at 3 -- depth was already validated as the right point on the cost/quality
curve. We sweep the optimization-side knobs the original crosslead study tuned
(lr, dropout) plus the one architectural granularity dial that's specific to
this model (n_tokens, the temporal patches per attention layer).

Setup matches the user's preferred eval protocol:
  - Full unbalanced dataset (data/seniordesign_upload, ~85/15)
  - Patient-grouped 80/20 holdout + 5-fold CV (no leakage)
  - Weighted CE loss
  - Basic 3x augmentation, no over/undersampling

Search space:
  lr       : log-uniform [1e-4, 5e-3]   (matches original crosslead study)
  dropout  : uniform     [0.05, 0.40]   (matches original crosslead study;
                                          attn_dropout tied to same value)
  n_tokens : {4, 8, 16}

Outputs (saved to optuna_crosslead_large_attn_lr/<timestamp>/):
  - study.db, best_params.json, summary.txt, best_model.pt
  - {contour, parallel, history, importance, slice}_plot.html

Usage:
    python -m src.optimize_crosslead_large_attn_lr --n-trials 24 --n-folds 5 --epochs 50
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
from sklearn.model_selection import train_test_split

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_crosslead_large_attn import RepNetCrossLeadLargeAttnModel
from src.preprocessing.augmentation import (
    AmplitudeScaling,
    GaussianNoise,
    RandomTimeShift,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

# Architecture stays at depth-3, only optimization knobs + n_tokens vary.
FIXED_PARAMS = dict(
    n_layers=3,
    base_filters=32,
    wide_kernel=7,
    narrow_kernel=5,
    n_attn_heads=4,
    attn_mlp_ratio=4,
    batch_size=64,
    loss_fn="weighted",
)


def preprocess(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def augment_train(X: np.ndarray, y: np.ndarray,
                  seed: int = SEED, n_copies: int = 2) -> tuple[np.ndarray, np.ndarray]:
    parts_X, parts_y = [X], [y]
    for i in range(n_copies):
        rng_state = np.random.get_state()
        np.random.seed(seed + i)
        X_aug = X.copy()
        X_aug, _ = GaussianNoise(sigma=0.02).transform(X_aug, None)
        X_aug, _ = AmplitudeScaling(scale_range=0.1).transform(X_aug, None)
        X_aug, _ = RandomTimeShift(max_shift=100).transform(X_aug, None)
        parts_X.append(X_aug)
        parts_y.append(y.copy())
        np.random.set_state(rng_state)
    X_out = np.concatenate(parts_X, axis=0)
    y_out = np.concatenate(parts_y, axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_out))
    return X_out[idx], y_out[idx]


def objective(trial: optuna.Trial, X_dev, y_dev, folds, epochs: int) -> float:
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.40)
    n_tokens = trial.suggest_categorical("n_tokens", [4, 8, 16])

    params = {
        **FIXED_PARAMS,
        "lr": lr,
        "dropout": dropout,
        "attn_dropout": dropout,   # tie to same value (matches original study)
        "n_tokens": n_tokens,
    }

    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)

        model = RepNetCrossLeadLargeAttnModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)

        logger.info(
            "Trial %d | lr=%.5f dropout=%.3f n_tokens=%d | fold %d/%d | AUROC=%.4f",
            trial.number, lr, dropout, n_tokens, fold_idx + 1, len(folds), auroc,
        )

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    logger.info(
        "Trial %d | lr=%.5f dropout=%.3f n_tokens=%d | CV AUROC=%.4f (+/- %.4f)",
        trial.number, lr, dropout, n_tokens, mean_auroc, std_auroc,
    )
    return mean_auroc


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def main():
    parser = argparse.ArgumentParser(
        description="Optuna search: lr + dropout + n_tokens for RepNetCrossLeadLargeAttn (n_layers=3)",
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    run_dir = Path("optuna_crosslead_large_attn_lr") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    logger.info("CUDA available: %s", torch.cuda.is_available())

    X, y, groups = load_seniordesign(args.data_dir, return_patient_ids=True)
    valid = ~np.isnan(groups.astype(float))
    n_drop = int((~valid).sum())
    if n_drop:
        logger.info("Dropping %d rows with missing Pat_Obfus_MRN", n_drop)
        X, y, groups = X[valid], y[valid], groups[valid]
    logger.info(
        "Loaded %d samples (pos=%d, neg=%d, pos_rate=%.1f%%)",
        len(y), int((y == 1).sum()), int((y == 0).sum()), 100 * y.mean(),
    )
    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, groups, test_size=0.20, seed=args.seed,
    )
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)
    logger.info(
        "Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)  patient-grouped",
        len(y_dev), 100 * y_dev.mean(), len(y_test), 100 * y_test.mean(),
    )

    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="crosslead_large_attn_lr_dropout_tokens",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds, epochs=args.epochs),
        n_trials=args.n_trials,
    )

    best_full = {
        **FIXED_PARAMS,
        "lr": study.best_params["lr"],
        "dropout": study.best_params["dropout"],
        "attn_dropout": study.best_params["dropout"],
        "n_tokens": study.best_params["n_tokens"],
    }
    with open(run_dir / "best_params.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_cv_auroc": study.best_value,
            "best_params": study.best_params,
            "fixed_params": FIXED_PARAMS,
            "full_params": best_full,
        }, f, indent=2, default=str)

    # Retrain best on full dev (90/10 stratified early-stop split)
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=args.seed,
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=args.seed)
    model = RepNetCrossLeadLargeAttnModel(**best_full, epochs=args.epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    torch.save(model.model.state_dict(), run_dir / "best_model.pt")

    proba = model.predict_proba(X_test)
    test_auroc = float(roc_auc_score(y_test, proba))
    thresh_j = youden_threshold(y_test, proba)

    lines = [
        "=" * 60,
        "OPTIMIZATION COMPLETE - RepNetCrossLeadLargeAttn (n_layers=3)",
        "                       lr + dropout + n_tokens",
        "=" * 60,
        f"Best trial:       #{study.best_trial.number}",
        f"Best CV AUROC:    {study.best_value:.4f}",
        f"Best lr:          {study.best_params['lr']:.6f}",
        f"Best dropout:     {study.best_params['dropout']:.4f}",
        f"Best n_tokens:    {study.best_params['n_tokens']}",
        "",
        "=" * 60,
        "HOLDOUT TEST",
        "=" * 60,
        f"  AUROC: {test_auroc:.4f}",
        "",
        f"  Threshold = 0.50:",
        classification_report(y_test, (proba >= 0.5).astype(int),
                              target_names=["No PE", "PE"], zero_division=0),
        f"  Threshold = {thresh_j:.3f} (Youden's J):",
        classification_report(y_test, (proba >= thresh_j).astype(int),
                              target_names=["No PE", "PE"], zero_division=0),
        "",
        "All trials:",
    ]
    for trial in study.trials:
        if trial.value is None:
            continue
        lines.append(
            f"  #{trial.number:3d} | lr={trial.params['lr']:.6f} "
            f"dropout={trial.params['dropout']:.4f} "
            f"n_tokens={trial.params['n_tokens']:2d} | AUROC={trial.value:.4f}"
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
        plot_contour(study, params=["lr", "dropout"]).write_html(
            str(run_dir / "contour_plot.html"))
        plot_optimization_history(study).write_html(
            str(run_dir / "optimization_history.html"))
        plot_parallel_coordinate(study, params=["lr", "dropout", "n_tokens"]).write_html(
            str(run_dir / "parallel_coordinate.html"))
        plot_param_importances(study).write_html(
            str(run_dir / "param_importances.html"))
        plot_slice(study, params=["lr", "dropout", "n_tokens"]).write_html(
            str(run_dir / "slice_plot.html"))
        logger.info("Saved plots to %s", run_dir)
    except Exception as e:
        logger.warning("Could not generate plots: %s", e)

    print(f"\nResults saved to: {run_dir}/")


if __name__ == "__main__":
    main()
