"""Head-to-head benchmark: 2-stage Baseline vs. depth-3 + 8-token cross-lead attention.

Patient-grouped splits (no leakage by Pat_Obfus_MRN). 80/20 holdout + 5-fold CV.
Each model trained on identical folds with identical fixed hyperparameters so the
only varying axis is architecture.

Outputs (saved to benchmark_baseline_attn/<timestamp>/):
  - config.json       — run configuration
  - cv_results.json   — per-model per-fold AUROCs and test AUROC
  - summary.txt       — human-readable comparison table

Usage:
    python -m src.benchmark_baseline_attn
    python -m src.benchmark_baseline_attn --n-folds 5 --epochs 50
    python -m src.benchmark_baseline_attn --ungrouped   # disable patient grouping (for leakage-bias estimate)
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
    kfold_cv_indices,
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout,
    split_holdout_grouped,
)
from src.models.repnet_baseline import RepNetBaselineModel
from src.models.repnet_baseline_large_attn import RepNetBaselineLargeAttnModel
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

BASELINE_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.1,
    lr=1e-3,
    batch_size=64,
    loss_fn="weighted",
)

ATTN_PARAMS = dict(
    stage_filters=(32, 64, 128),
    wide_kernel=7,
    narrow_kernel=5,
    narrow_kernel_2=3,
    dropout=0.1,
    n_tokens=8,
    n_attn_blocks=1,
    n_attn_heads=4,
    attn_mlp_ratio=4,
    attn_dropout=0.1,
    lr=1e-3,
    batch_size=64,
    loss_fn="weighted",
)

# Inherits lr/dropout from the crosslead Optuna study
# (optuna/2026-04-17_00-37-13/best_params.json).
CROSSLEAD_LARGE_ATTN_PARAMS = dict(
    n_layers=3,
    base_filters=32,
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.0636,
    n_tokens=8,
    n_attn_heads=4,
    attn_mlp_ratio=4,
    attn_dropout=0.0636,
    lr=0.000876,
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


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def run_cv(model_cls, params: dict, X_dev, y_dev, folds, epochs: int) -> list[float]:
    aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)
        logger.info(
            "  Fold %d/%d — train=%d (pos=%d neg=%d) val=%d",
            fold_idx + 1, len(folds), len(y_tr),
            int((y_tr == 1).sum()), int((y_tr == 0).sum()), len(y_val),
        )
        model = model_cls(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = model.score(X_val, y_val)
        aurocs.append(auc)
        print(f"  → Fold {fold_idx+1} AUROC: {auc:.4f}")
    return aurocs


def train_final(model_cls, params: dict, X_dev, y_dev, epochs: int):
    """Retrain on full dev with internal 90/10 early-stop split (label-stratified within dev)."""
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=SEED,
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED)
    logger.info(
        "Final training: %d train (augmented) + %d early-stop (clean)",
        len(y_tr), len(y_es),
    )
    model = model_cls(**params, epochs=epochs)
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
    parser = argparse.ArgumentParser(description="Benchmark Baseline vs Baseline+Attn")
    parser.add_argument("--data-dir", default="data/seniordesign_upload_balanced")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ungrouped", action="store_true",
                        help="Disable patient grouping (for leakage-bias estimate)")
    args = parser.parse_args()

    run_dir = Path("benchmark_baseline_attn") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    # Load + preprocess
    X, y, groups = load_seniordesign(args.data_dir, return_patient_ids=True)
    X = preprocess(X)

    # Holdout
    if args.ungrouped:
        logger.warning("Running with UNGROUPED splits — patient leakage possible.")
        X_dev, X_test, y_dev, y_test = split_holdout(X, y, test_size=0.20, seed=args.seed)
        folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=args.seed)
    else:
        X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
            X, y, groups, test_size=0.20, seed=args.seed,
        )
        folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)

    # Persist run config
    config = {
        "data_dir": args.data_dir,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "ungrouped": args.ungrouped,
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "baseline_params": BASELINE_PARAMS,
        "attn_params": ATTN_PARAMS,
        "crosslead_large_attn_params": CROSSLEAD_LARGE_ATTN_PARAMS,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    models_to_run = [
        ("repnet_baseline", RepNetBaselineModel, BASELINE_PARAMS),
        ("repnet_baseline_large_attn", RepNetBaselineLargeAttnModel, ATTN_PARAMS),
        ("repnet_crosslead_large_attn", RepNetCrossLeadLargeAttnModel,
         CROSSLEAD_LARGE_ATTN_PARAMS),
    ]

    results: dict = {}
    for name, model_cls, params in models_to_run:
        print(f"\n{'#'*72}\n  {name}\n{'#'*72}")
        cv_aurocs = run_cv(model_cls, params, X_dev, y_dev, folds, epochs=args.epochs)
        print(f"\n  Retraining {name} on full dev …")
        final_model = train_final(model_cls, params, X_dev, y_dev, epochs=args.epochs)
        eval_metrics = evaluate(final_model, X_test, y_test)
        results[name] = {
            "cv_aurocs": cv_aurocs,
            "cv_mean": float(np.mean(cv_aurocs)),
            "cv_std": float(np.std(cv_aurocs)),
            **eval_metrics,
        }

    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    lines = []
    lines.append(f"Benchmark — {'UNGROUPED' if args.ungrouped else 'PATIENT-GROUPED'} splits")
    lines.append(f"  Data:        {args.data_dir}")
    lines.append(f"  Dev/Test:    {len(y_dev)} / {len(y_test)}")
    lines.append(f"  Folds:       {args.n_folds}")
    lines.append(f"  Epochs/fold: {args.epochs}")
    lines.append("")
    lines.append(f"{'Model':<32} {'CV AUROC (mean ± std)':<24} {'Test AUROC':<12}")
    lines.append("-" * 72)
    for name, r in results.items():
        lines.append(
            f"{name:<32} {r['cv_mean']:.4f} ± {r['cv_std']:.4f}        {r['test_auc']:.4f}"
        )
    lines.append("")
    for name, r in results.items():
        lines.append(f"{name} per-fold AUROCs: {[f'{v:.4f}' for v in r['cv_aurocs']]}")
        lines.append(f"  Youden's J threshold: {r['threshold_youden']:.3f}")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
