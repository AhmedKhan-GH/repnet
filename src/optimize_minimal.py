"""Minimal Optuna study: ECG-AI ResNet, tune learning rate + dropout only.

Usage:
    python -m src.optimize_minimal --n-trials 10
"""

import argparse
import logging

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.data.dataset import load_nightingale, split_holdout
from src.models.ecg_ai_resnet import ECGAIResNetModel
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)


def objective(trial, X_train, y_train, X_val, y_val):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    model = ECGAIResNetModel(
        stage_filters=(16, 32, 64),  # paper defaults
        kernel_size=3,               # paper default
        dropout=dropout,
        lr=lr,
        batch_size=32,
        epochs=20,
    )
    model.fit(X_train, y_train, X_val, y_val)
    return model.score(X_val, y_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Load and normalize
    X, y = load_nightingale("data/Nightingale Dataset")
    zscore = ZScoreNormalization(per_lead=True)
    X, _ = zscore.transform(X)

    # Simple 80/20 split
    X_train, X_val, y_train, y_val = split_holdout(X, y, test_size=0.20, seed=args.seed)
    logger.info("Train: %d, Val: %d", len(y_train), len(y_val))

    # Run study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
    )
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
    )

    # Results
    print("\n" + "=" * 50)
    print(f"Best AUROC: {study.best_value:.4f}")
    print(f"Best LR:      {study.best_params['lr']:.6f}")
    print(f"Best Dropout: {study.best_params['dropout']:.4f}")
    print("=" * 50)

    # Contour plot
    try:
        from optuna.visualization import plot_contour
        fig = plot_contour(study, params=["lr", "dropout"])
        fig.write_html("optuna_contour.html")
        print("\nContour plot saved to optuna_contour.html")
    except ImportError:
        print("\nInstall plotly for contour plot: pip install plotly")


if __name__ == "__main__":
    main()
