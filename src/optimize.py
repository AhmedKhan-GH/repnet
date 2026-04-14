"""Optuna hyperparameter optimization loop for ECG classification.

Evaluation strategy:
  - 20% stratified holdout for final test
  - 5-fold stratified cross-validation on the remaining 80%
  - Each Optuna trial reports mean AUROC across the 5 folds
  - Best trial is retrained on full 80% dev set and evaluated on the 20% holdout

Usage:
    python -m src.optimize --n-trials 20
    python -m src.optimize --data nightingale --n-trials 20
    python -m src.optimize --data synthetic --n-trials 5
"""

import argparse
import logging

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.data.dataset import generate_synthetic_ecg, kfold_cv_indices, load_nightingale, split_holdout
from src.models import MODEL_REGISTRY
from src.preprocessing import (
    BaselineWanderFilter,
    MajorityUndersampling,
    PreprocessingPipeline,
    SMOTE,
    ZScoreNormalization,
)

logger = logging.getLogger(__name__)


def build_default_pipeline() -> PreprocessingPipeline:
    """Construct the default preprocessing pipeline.

    Add new PreprocessingStep subclasses here to make them available
    to the Optuna search.
    """
    return PreprocessingPipeline([
        MajorityUndersampling(),
        SMOTE(),
        BaselineWanderFilter(),
        ZScoreNormalization(),
    ])


def objective(
    trial: optuna.Trial,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    # 1. Choose model type
    model_name = trial.suggest_categorical("model", list(MODEL_REGISTRY.keys()))

    # 2. Configure preprocessing — each step's on/off + params are tuned
    pipeline = build_default_pipeline()
    pipeline.suggest_and_configure(trial)

    # 3. Apply signal transforms (filter, normalization) ONCE on all dev data.
    #    Resampling (undersample, SMOTE) is per-fold since it depends on the train split.
    X_dev_p, _ = pipeline.transform(X_dev)

    # 4. Get model hyperparameters (suggested once, shared across folds)
    model_cls = MODEL_REGISTRY[model_name]
    params = model_cls.suggest_params(trial)

    # 5. K-fold cross-validation
    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X_dev_p[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev_p[val_idx], y_dev[val_idx]

        # Resampling on train only (undersampling/SMOTE)
        X_train, y_train = pipeline.resample(X_train, y_train)

        model = model_cls(**params)
        model.fit(X_train, y_train, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)

        logger.info(
            "Trial %d | fold %d/%d | AUROC=%.4f",
            trial.number, fold_idx + 1, len(folds), auroc,
        )

    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    logger.info(
        "Trial %d | model=%s | pipeline=%s | mean AUROC=%.4f (+/- %.4f)",
        trial.number, model_name, pipeline, mean_auroc, std_auroc,
    )
    return mean_auroc


def rebuild_from_params(best_params: dict):
    """Reconstruct pipeline and model kwargs from a trial's params dict."""
    pipeline = build_default_pipeline()
    for step in pipeline.steps:
        prefix = f"prep_{step.name}_enabled"
        step.enabled = best_params.get(prefix, True)
        # Restore step-specific params
        for key, val in best_params.items():
            if key.startswith(f"prep_{step.name}_") and key != prefix:
                attr = key[len(f"prep_{step.name}_"):]
                setattr(step, attr, val)

    model_cls = MODEL_REGISTRY[best_params["model"]]
    model_params = {k: v for k, v in best_params.items()
                    if not k.startswith("prep_") and k != "model"}
    # Reconstruct tuple params
    if "resnet_stage1_filters" in model_params:
        model_params["stage_filters"] = (
            model_params.pop("resnet_stage1_filters"),
            model_params.pop("resnet_stage2_filters"),
            model_params.pop("resnet_stage3_filters"),
        )
    # Strip model-name prefixes from param keys
    clean_params = {}
    for k, v in model_params.items():
        clean_key = k.split("_", 1)[1] if "_" in k else k
        clean_params[clean_key] = v

    return pipeline, model_cls, clean_params


def main():
    parser = argparse.ArgumentParser(description="Optuna ECG hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=3,
                        help="Number of CV folds (default 3, use 5 for final runs)")
    parser.add_argument("--data", type=str, default="synthetic",
                        help="'synthetic', 'nightingale', or path to .npz file")
    parser.add_argument("--data-dir", type=str, default="data/Nightingale Dataset",
                        help="Directory for Nightingale dataset (used with --data nightingale)")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of synthetic samples (only used with --data synthetic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="ecg_optimization")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///study.db)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load data
    if args.data == "synthetic":
        logger.info("Generating synthetic ECG data (n=%d)", args.n_samples)
        X, y = generate_synthetic_ecg(n_samples=args.n_samples, seed=args.seed)
    elif args.data == "nightingale":
        X, y = load_nightingale(args.data_dir)
    else:
        logger.info("Loading data from %s", args.data)
        data = np.load(args.data)
        X, y = data["X"], data["y"]

    # 20% held-out test set
    X_dev, X_test, y_dev, y_test = split_holdout(X, y, test_size=0.20, seed=args.seed)
    folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=args.seed)

    logger.info(
        "Dev: %d samples (%d-fold CV), Test holdout: %d samples (pos rate: %.1f%%)",
        len(y_dev), args.n_folds, len(y_test), 100 * y.mean(),
    )

    # Run optimization
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        storage=args.storage,
    )
    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds),
        n_trials=args.n_trials,
    )

    # Report results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mean CV AUROC: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain best config on full dev set, evaluate on held-out test
    pipeline, model_cls, clean_params = rebuild_from_params(study.best_params)

    model = model_cls(**clean_params)
    X_dev_p, _ = pipeline.transform(X_dev)
    X_test_p, _ = pipeline.transform(X_test)

    # Resample dev for training
    X_dev_rs, y_dev_rs = pipeline.resample(X_dev_p.copy(), y_dev.copy())

    # Use a small random slice as early-stopping signal
    n_es = max(1, int(0.1 * len(y_dev_rs)))
    rng = np.random.default_rng(args.seed)
    es_idx = rng.choice(len(y_dev_rs), n_es, replace=False)
    train_mask = np.ones(len(y_dev_rs), dtype=bool)
    train_mask[es_idx] = False

    model.fit(X_dev_rs[train_mask], y_dev_rs[train_mask], X_dev_rs[es_idx], y_dev_rs[es_idx])
    test_auroc = model.score(X_test_p, y_test)
    print(f"\nTest AUROC (20% held-out): {test_auroc:.4f}")


if __name__ == "__main__":
    main()
