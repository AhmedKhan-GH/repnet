"""Optuna hyperparameter optimization for RepNet Baseline.

Stage 1: Learning rate + dropout search.
All other params fixed at EDA-informed defaults.

Evaluation:
  - 80/20 stratified holdout
  - 5-fold stratified CV on dev set
  - Concatenative augmentation on train folds only
  - Each trial reports mean AUROC across folds

Outputs (saved to optuna_baseline/YYYY-MM-DD_HH-MM-SS/):
  - study.db          — Optuna SQLite study (resume-able)
  - results.log       — full log of all trials
  - best_model.pt     — best model weights
  - best_params.json  — best hyperparameters
  - summary.txt       — final report

Usage:
    python -m src.optimize_baseline --n-trials 20
    python -m src.optimize_baseline --n-trials 30 --n-folds 5
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

from src.data.dataset import kfold_cv_indices, load_seniordesign, split_holdout
from src.models.repnet_baseline import RepNetBaselineModel
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.augmentation import GaussianNoise, AmplitudeScaling, RandomTimeShift

logger = logging.getLogger(__name__)

SEED = 42


def preprocess(X: np.ndarray) -> np.ndarray:
    """Fixed preprocessing: high-pass 0.5 Hz + notch 60 Hz + z-score per lead."""
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def augment_train(X: np.ndarray, y: np.ndarray,
                  seed: int = SEED, n_copies: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Concatenative augmentation: originals + n_copies augmented versions."""
    parts_X = [X]
    parts_y = [y]
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


# Fixed architecture params (EDA-informed, same as Hybrid for fair comparison)
FIXED_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    batch_size=64,
    epochs=50,
    loss_fn="weighted",
)


def objective(
    trial: optuna.Trial,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    # Stage 1: only search lr + dropout (same search space as Hybrid)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)

    params = {**FIXED_PARAMS, "lr": lr, "dropout": dropout}

    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)

        model = RepNetBaselineModel(**params)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)

        logger.info("Trial %d | fold %d/%d | AUROC=%.4f",
                     trial.number, fold_idx + 1, len(folds), auroc)

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    logger.info("Trial %d | lr=%.5f dropout=%.3f | AUROC=%.4f (±%.4f)",
                trial.number, lr, dropout, mean_auroc, std_auroc)
    return mean_auroc


def main():
    parser = argparse.ArgumentParser(description="Optuna: RepNet Baseline lr+dropout search")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data/seniordesign_upload_balanced")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create dated output directory
    run_dir = Path("optuna_baseline") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "results.log"),
        ],
    )

    logger.info("Output directory: %s", run_dir)

    # Load & preprocess
    X, y = load_seniordesign(args.data_dir)
    X = preprocess(X)

    # 80/20 holdout
    X_dev, X_test, y_dev, y_test = split_holdout(X, y, test_size=0.20, seed=args.seed)
    folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=args.seed)

    logger.info("Dev: %d samples (%d-fold CV), Test: %d samples",
                len(y_dev), args.n_folds, len(y_test))

    # Run search with SQLite storage for resume-ability
    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="baseline_lr_dropout",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds),
        n_trials=args.n_trials,
    )

    # Save best params
    best_params_full = {**FIXED_PARAMS,
                        "lr": study.best_params["lr"],
                        "dropout": study.best_params["dropout"]}

    with open(run_dir / "best_params.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_cv_auroc": study.best_value,
            "best_params": study.best_params,
            "fixed_params": FIXED_PARAMS,
            "full_params": best_params_full,
        }, f, indent=2, default=str)

    # Retrain best on full dev → save weights + test eval
    from sklearn.model_selection import train_test_split
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=args.seed
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=args.seed)

    model = RepNetBaselineModel(**best_params_full)
    model.fit(X_tr, y_tr, X_es, y_es)

    # Save model weights
    torch.save(model.model.state_dict(), run_dir / "best_model.pt")
    logger.info("Saved best model weights to %s", run_dir / "best_model.pt")

    # Test evaluation
    proba = model.predict_proba(X_test)
    test_auroc = roc_auc_score(y_test, proba)
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    youden_thresh = float(thresholds[np.argmax(tpr - fpr)])

    # Build summary
    lines = []
    lines.append(f"{'='*60}")
    lines.append("OPTIMIZATION COMPLETE — RepNet Baseline: lr + dropout")
    lines.append(f"{'='*60}")
    lines.append(f"Best trial: #{study.best_trial.number}")
    lines.append(f"Best CV AUROC: {study.best_value:.4f}")
    lines.append(f"Best lr:      {study.best_params['lr']:.6f}")
    lines.append(f"Best dropout: {study.best_params['dropout']:.4f}")
    lines.append("")
    lines.append(f"{'='*60}")
    lines.append("HOLDOUT TEST")
    lines.append(f"{'='*60}")
    lines.append(f"  AUROC: {test_auroc:.4f}")
    lines.append(f"\n  Threshold = 0.50:")
    lines.append(classification_report(y_test, (proba >= 0.5).astype(int),
                                       target_names=["No PE", "PE"]))
    lines.append(f"  Threshold = {youden_thresh:.3f} (Youden's J):")
    lines.append(classification_report(y_test, (proba >= youden_thresh).astype(int),
                                       target_names=["No PE", "PE"]))
    lines.append("")
    lines.append("All trials:")
    for trial in study.trials:
        lines.append(f"  #{trial.number:3d} | lr={trial.params['lr']:.6f} "
                     f"dropout={trial.params['dropout']:.4f} | "
                     f"AUROC={trial.value:.4f}")

    summary = "\n".join(lines)

    # Save summary to file
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)

    # Print to console
    print(summary)

    # Generate Optuna visualization plots
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
        plot_parallel_coordinate(study, params=["lr", "dropout"]).write_html(
            str(run_dir / "parallel_coordinate.html"))
        plot_param_importances(study).write_html(
            str(run_dir / "param_importances.html"))
        plot_slice(study, params=["lr", "dropout"]).write_html(
            str(run_dir / "slice_plot.html"))
        logger.info("Saved plots to %s", run_dir)
    except Exception as e:
        logger.warning("Could not generate plots: %s", e)

    # Save training curves for best retrained model
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        epochs_axis = list(range(1, len(model.history["train_loss"]) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Val AUROC"))
        fig.add_trace(
            go.Scatter(x=epochs_axis, y=model.history["train_loss"], name="Train Loss"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=epochs_axis, y=model.history["val_auroc"], name="Val AUROC"),
            row=1, col=2,
        )
        fig.update_layout(
            title=f"Best model training curves (lr={study.best_params['lr']:.5f}, "
                  f"dropout={study.best_params['dropout']:.3f})",
            xaxis_title="Epoch", xaxis2_title="Epoch",
        )
        fig.write_html(str(run_dir / "training_curves.html"))
        logger.info("Saved training curves to %s", run_dir / "training_curves.html")

        # Save raw training history as JSON
        with open(run_dir / "training_history.json", "w") as f:
            json.dump({
                "epochs": epochs_axis,
                "train_loss": model.history["train_loss"],
                "val_auroc": model.history["val_auroc"],
            }, f, indent=2)
        logger.info("Saved training history to %s", run_dir / "training_history.json")
    except Exception as e:
        logger.warning("Could not save training curves: %s", e)

    print(f"\nResults saved to: {run_dir}/")


if __name__ == "__main__":
    main()
