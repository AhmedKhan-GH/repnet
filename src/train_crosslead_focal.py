"""Train RepNet CrossLead on the UNBALANCED dataset with focal loss.

Uses the Optuna-tuned hyperparameters from the crosslead study
(`optuna/2026-04-17_00-37-13/best_params.json`):
    lr      = 0.0008756917546352803
    dropout = 0.06355381998641418

Differences vs. that study:
    - Loss switched from weighted CE → focal (gamma=2.0, alpha=0.25)
    - Trained on `data/seniordesign_upload` (~85/15 imbalance, N=2191)
      instead of the balanced subset
    - Patient-grouped 5-fold CV + 80/20 holdout (no patient leakage)

Outputs (saved to crosslead_focal/<timestamp>/):
    - config.json     — run configuration
    - cv_results.json — per-fold AUROCs + holdout test metrics
    - summary.txt     — human-readable comparison table
    - results.log     — full training log

Usage:
    python -m src.train_crosslead_focal
    python -m src.train_crosslead_focal --n-folds 5 --epochs 50
    python -m src.train_crosslead_focal --focal-gamma 2.0 --focal-alpha 0.25
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    majority_undersample,
    split_holdout_grouped,
)
from src.models.repnet_crosslead import RepNetCrossLeadModel
from src.preprocessing.augmentation import (
    AmplitudeScaling,
    GaussianNoise,
    RandomTimeShift,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

# Crosslead Optuna study results (optuna/2026-04-17_00-37-13/best_params.json),
# now with focal loss instead of weighted CE.
CROSSLEAD_FOCAL_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    n_heads=4,
    dropout=0.06355381998641418,
    lr=0.0008756917546352803,
    batch_size=64,
    loss_fn="focal",
    focal_gamma=2.0,
    focal_alpha=0.25,
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


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def run_cv(params: dict, X_dev, y_dev, folds, epochs: int,
           undersample: bool = True) -> list[float]:
    aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        # Train: undersample majority to match minority count, then augment.
        # Val: untouched (natural prevalence, no augmentation).
        if undersample:
            X_tr, y_tr = majority_undersample(X_tr, y_tr, ratio=1.0, seed=SEED + fold_idx)
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)

        logger.info(
            "  Fold %d/%d - train=%d (pos=%d neg=%d) val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr),
            int((y_tr == 1).sum()), int((y_tr == 0).sum()),
            len(y_val), int((y_val == 1).sum()), int((y_val == 0).sum()),
        )
        model = RepNetCrossLeadModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = model.score(X_val, y_val)
        aurocs.append(auc)
        print(f"  -> Fold {fold_idx+1} AUROC: {auc:.4f}")
    return aurocs


def train_final(params: dict, X_dev, y_dev, epochs: int,
                undersample: bool = True):
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=SEED,
    )

    if undersample:
        X_tr, y_tr = majority_undersample(X_tr, y_tr, ratio=1.0, seed=SEED)
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED)

    logger.info(
        "Final training: %d train (augmented%s) + %d early-stop (natural)",
        len(y_tr), ", undersampled" if undersample else "", len(y_es),
    )
    model = RepNetCrossLeadModel(**params, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    return model


def evaluate(model, X_test, y_test) -> dict:
    probs = model.predict_proba(X_test)
    test_auc = float(roc_auc_score(y_test, probs))
    thresh_j = youden_threshold(y_test, probs)
    return dict(
        test_auc=test_auc,
        threshold_youden=thresh_j,
        report_05=classification_report(
            y_test, (probs >= 0.5).astype(int),
            target_names=["No PE", "PE"], output_dict=True, zero_division=0,
        ),
        report_youden=classification_report(
            y_test, (probs >= thresh_j).astype(int),
            target_names=["No PE", "PE"], output_dict=True, zero_division=0,
        ),
        probs=probs.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train RepNet CrossLead with focal loss on the unbalanced dataset",
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--no-undersample", action="store_true",
                        help="Disable majority undersampling on train fold")
    args = parser.parse_args()

    params = dict(CROSSLEAD_FOCAL_PARAMS)
    params["focal_gamma"] = args.focal_gamma
    params["focal_alpha"] = args.focal_alpha

    run_dir = Path("crosslead_focal") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    if torch.cuda.is_available():
        logger.info("Device: %s", torch.cuda.get_device_name(0))

    X, y, groups = load_seniordesign(args.data_dir, return_patient_ids=True)

    # Drop rows with missing patient IDs (5 rows in seniordesign_upload) so
    # patient-grouped splits don't crash on NaN.
    valid = ~np.isnan(groups.astype(float))
    n_drop = int((~valid).sum())
    if n_drop:
        logger.info("Dropping %d rows with missing Pat_Obfus_MRN", n_drop)
        X, y, groups = X[valid], y[valid], groups[valid]

    logger.info(
        "Loaded %d samples (pos=%d, neg=%d, pos_rate=%.1f%%) from %d patients",
        len(y), int((y == 1).sum()), int((y == 0).sum()),
        100 * y.mean(), len(set(groups)),
    )
    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, groups, test_size=0.20, seed=args.seed,
    )
    logger.info(
        "Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
        len(y_dev), 100 * y_dev.mean(),
        len(y_test), 100 * y_test.mean(),
    )
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)

    undersample = not args.no_undersample

    config = {
        "data_dir": args.data_dir,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_total": int(len(y)),
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "pos_rate_dev": float(y_dev.mean()),
        "pos_rate_test": float(y_test.mean()),
        "undersample_majority": undersample,
        "params": params,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'#'*72}\n  RepNet CrossLead - focal loss, unbalanced data\n{'#'*72}")
    cv_aurocs = run_cv(params, X_dev, y_dev, folds, epochs=args.epochs,
                       undersample=undersample)

    print("\n  Retraining on full dev ...")
    final_model = train_final(params, X_dev, y_dev, epochs=args.epochs,
                              undersample=undersample)
    eval_metrics = evaluate(final_model, X_test, y_test)

    results = {
        "cv_aurocs": cv_aurocs,
        "cv_mean": float(np.mean(cv_aurocs)),
        "cv_std": float(np.std(cv_aurocs)),
        **eval_metrics,
    }
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "RepNet CrossLead - focal loss, unbalanced data (PATIENT-GROUPED)",
        f"  Data:           {args.data_dir}",
        f"  Total/Dev/Test: {len(y)} / {len(y_dev)} / {len(y_test)}",
        f"  Dev pos rate:   {100 * y_dev.mean():.1f}%",
        f"  Folds:          {args.n_folds}",
        f"  Epochs/fold:    {args.epochs}",
        f"  lr:             {params['lr']:.6f}  (from crosslead Optuna study)",
        f"  dropout:        {params['dropout']:.4f}  (from crosslead Optuna study)",
        f"  loss_fn:        focal (gamma={params['focal_gamma']}, alpha={params['focal_alpha']})",
        f"  undersample:    {undersample} (majority -> 1:1 with minority)",
        "",
        f"  CV AUROC:  {results['cv_mean']:.4f} +/- {results['cv_std']:.4f}",
        f"  Per-fold:  {[f'{v:.4f}' for v in cv_aurocs]}",
        f"  Test AUROC: {results['test_auc']:.4f}  (Youden's J threshold: {results['threshold_youden']:.3f})",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
