"""RepNet Baseline — 80/20 + 5-fold CV evaluation.

Evaluation protocol:
  1. Stratified 80/20 holdout split — test set is never touched until final eval
  2. 5-fold stratified CV on the 80% dev set
     - Signal conditioning (notch 60 Hz + bandpass 0.5–40 Hz + z-score) applied once
     - Augmentation (noise, amplitude scaling, time shift) applied per-fold on train only
  3. Final model retrained on all 80% dev data, evaluated on 20% test holdout
  4. Results saved to timestamped directory (logs, plots, weights, metrics)

Outputs (saved to cv_results/YYYY-MM-DD_HH-MM-SS/):
  - results.log       — full training log
  - config.json       — hyperparameters and data info
  - cv_results.json   — per-fold and test metrics
  - summary.txt       — formatted results report
  - model_*.pt        — trained model weights (one per model)
  - training_curves_*.html — loss/AUROC plots (one per model)

Usage:
    python -m src.train_cv
    python -m src.train_cv --data-dir "data/seniordesign_upload"
    python -m src.train_cv --n-folds 5 --epochs 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from src.data.dataset import kfold_cv_indices, load_seniordesign, split_holdout
from src.models.repnet_baseline import RepNetBaselineModel
from src.models.repnet_crosslead import RepNetCrossLeadModel
from src.models.repnet_temporal import RepNetTemporalModel
from src.models.repnet_crosslead_temporal import RepNetCrossLeadTemporalModel
from src.preprocessing.filters import NotchFilter, BaselineWanderFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.augmentation import GaussianNoise, AmplitudeScaling, RandomTimeShift

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed hyperparameters (EDA-informed choices)
# ---------------------------------------------------------------------------
REPNET_BASELINE_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.15,
    lr=5e-4,
    batch_size=64,
    loss_fn="focal",
    focal_alpha=0.75,        # Clinical asymmetry: missing PE costlier than false alarm
    focal_gamma=2.0,         # Focus on hard examples (EDA silhouette=0.015, heavy overlap)
)

REPNET_CROSSLEAD_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.15,
    n_heads=4,               # cross-lead attention heads
    lr=5e-4,
    batch_size=64,
    loss_fn="focal",
    focal_alpha=0.75,        # Clinical asymmetry: missing PE costlier than false alarm
    focal_gamma=2.0,         # Focus on hard examples (EDA silhouette=0.015, heavy overlap)
)

REPNET_TEMPORAL_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.15,
    n_heads=4,               # temporal attention heads
    lr=5e-4,
    batch_size=64,
    loss_fn="focal",
    focal_alpha=0.75,        # Clinical asymmetry: missing PE costlier than false alarm
    focal_gamma=2.0,         # Focus on hard examples (EDA silhouette=0.015, heavy overlap)
)

REPNET_CROSSLEAD_TEMPORAL_PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.15,
    n_heads=4,               # both temporal and cross-lead attention heads
    lr=5e-4,
    batch_size=64,
    loss_fn="focal",
    focal_alpha=0.75,        # Clinical asymmetry: missing PE costlier than false alarm
    focal_gamma=2.0,         # Focus on hard examples (EDA silhouette=0.015, heavy overlap)
)

SEED = 42


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X: np.ndarray) -> np.ndarray:
    """Signal conditioning applied once to all data.

    Order matters:
      1. High-pass 0.5 Hz — remove baseline wander
      2. Notch 60 Hz — remove US powerline interference
      3. Z-score per lead — normalise amplitude scale
    """
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def augment_train(X: np.ndarray, y: np.ndarray,
                  seed: int = SEED, n_copies: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Concatenative augmentation: keep originals + append n_copies augmented versions.

    Each copy gets independently randomised augmentation:
      - GaussianNoise: measurement noise / electrode contact
      - AmplitudeScaling: inter-patient amplitude variation
      - RandomTimeShift: alignment variation across recordings

    With n_copies=2 and 236 training samples → 708 total (236 original + 472 augmented).
    """
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

    # Shuffle so originals and augmented copies are interleaved
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_out))
    return X_out[idx], y_out[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_cv(params: dict, X_dev: np.ndarray, y_dev: np.ndarray,
           folds, epochs: int, model_cls) -> list[float]:
    """Run k-fold CV, return per-fold validation AUROCs."""
    aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        # Augmentation on train fold only
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)
        logger.info(
            "  Fold %d/%d — train=%d (pos=%d neg=%d)  val=%d",
            fold_idx + 1, len(folds),
            len(y_tr), int((y_tr == 1).sum()), int((y_tr == 0).sum()), len(y_val),
        )

        model = model_cls(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = model.score(X_val, y_val)
        aurocs.append(auc)
        print(f"  → Fold {fold_idx+1} AUROC: {auc:.4f}")

    return aurocs


def train_final(params: dict, X_dev: np.ndarray, y_dev: np.ndarray,
                epochs: int, seed: int, model_cls):
    """Retrain on full dev set for final test evaluation.

    Stratified 90/10 split BEFORE augmentation to avoid leakage.
    Early-stop set is clean (no augmented copies of training samples).
    """
    from sklearn.model_selection import train_test_split
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=seed
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=seed)
    logger.info(
        "Final training: %d train (augmented) + %d early-stop (clean)",
        len(y_tr), len(y_es),
    )
    model = model_cls(**params, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    return model


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Find threshold that maximises Youden's J = sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def format_results(name: str, cv_aurocs: list[float],
                   test_auc: float, y_test: np.ndarray, probs_test: np.ndarray) -> tuple[str, dict]:
    """Format CV summary and test evaluation, return text and metrics dict."""
    cv_arr = np.array(cv_aurocs)

    # Calculate metrics at different thresholds
    preds_05 = (probs_test >= 0.5).astype(int)
    thresh_j = youden_threshold(y_test, probs_test)
    preds_j = (probs_test >= thresh_j).astype(int)

    # Build text output
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {name}")
    lines.append(f"{'='*60}")
    lines.append(f"  5-fold CV AUROC: {cv_arr.mean():.4f} ± {cv_arr.std():.4f}")
    lines.append(f"  Per-fold:        {[f'{v:.4f}' for v in cv_aurocs]}")
    lines.append(f"  Test AUROC:      {test_auc:.4f}")
    lines.append("")
    lines.append("  Classification report (threshold=0.50):")
    lines.append(classification_report(y_test, preds_05, target_names=["No PE", "PE"]))
    lines.append(f"  Classification report (Youden's J threshold={thresh_j:.3f}):")
    lines.append(classification_report(y_test, preds_j, target_names=["No PE", "PE"]))

    text_output = "\n".join(lines)

    # Build metrics dict
    from sklearn.metrics import precision_recall_fscore_support
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(y_test, preds_05, average='binary', pos_label=1)
    prec_j, rec_j, f1_j, _ = precision_recall_fscore_support(y_test, preds_j, average='binary', pos_label=1)

    metrics = {
        "cv_auroc_mean": float(cv_arr.mean()),
        "cv_auroc_std": float(cv_arr.std()),
        "cv_aurocs_per_fold": [float(x) for x in cv_aurocs],
        "test_auroc": float(test_auc),
        "threshold_0.5": {
            "precision": float(prec_05),
            "recall": float(rec_05),
            "f1": float(f1_05),
        },
        "threshold_youden_j": {
            "threshold": float(thresh_j),
            "precision": float(prec_j),
            "recall": float(rec_j),
            "f1": float(f1_j),
        }
    }

    return text_output, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RepNet Baseline 5-fold CV evaluation")
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Create timestamped output directory
    # ------------------------------------------------------------------
    run_dir = Path("cv_results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    logger.info("Loading data from %s …", args.data_dir)
    X, y = load_seniordesign(args.data_dir)
    logger.info("Preprocessing (high-pass 0.5 Hz + notch 60 Hz + z-score) …")
    X = preprocess(X)

    # ------------------------------------------------------------------
    # 2. 80/20 stratified holdout
    # ------------------------------------------------------------------
    X_dev, X_test, y_dev, y_test = split_holdout(X, y, test_size=0.20, seed=SEED)
    logger.info("Dev: %d  Test: %d  (pos rate dev=%.1f%%  test=%.1f%%)",
                len(y_dev), len(y_test),
                100 * y_dev.mean(), 100 * y_test.mean())

    # Save configuration
    config = {
        "data_dir": args.data_dir,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "seed": SEED,
        "dataset_size": {
            "total": len(y),
            "dev": len(y_dev),
            "test": len(y_test),
            "dev_pos_rate": float(y_dev.mean()),
            "test_pos_rate": float(y_test.mean()),
        },
        "repnet_baseline_params": REPNET_BASELINE_PARAMS,
        "repnet_crosslead_params": REPNET_CROSSLEAD_PARAMS,
        "repnet_temporal_params": REPNET_TEMPORAL_PARAMS,
        "repnet_crosslead_temporal_params": REPNET_CROSSLEAD_TEMPORAL_PARAMS,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to %s", run_dir / "config.json")

    # ------------------------------------------------------------------
    # 3. 5-fold CV
    # ------------------------------------------------------------------
    folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=SEED)

    models = [
        ("RepNet Baseline",            RepNetBaselineModel,            REPNET_BASELINE_PARAMS),
        ("RepNet CrossLead",           RepNetCrossLeadModel,           REPNET_CROSSLEAD_PARAMS),
        ("RepNet Temporal",            RepNetTemporalModel,            REPNET_TEMPORAL_PARAMS),
        ("RepNet CrossLead-Temporal",  RepNetCrossLeadTemporalModel,   REPNET_CROSSLEAD_TEMPORAL_PARAMS),
    ]

    results = {}
    all_metrics = {}

    for name, model_cls, params in models:
        print(f"\n{'#'*60}")
        print(f"  {name} — {args.n_folds}-fold CV")
        print(f"{'#'*60}")
        cv_aurocs = run_cv(params, X_dev, y_dev, folds, epochs=args.epochs,
                           model_cls=model_cls)

        print(f"\n  Retraining {name} on full dev set …")
        final_model = train_final(params, X_dev, y_dev, epochs=args.epochs,
                                  seed=SEED, model_cls=model_cls)
        probs_test = final_model.predict_proba(X_test)
        test_auc = roc_auc_score(y_test, probs_test)

        # Save model weights
        model_filename = f"model_{name.replace(' ', '_').lower()}.pt"
        torch.save(final_model.model.state_dict(), run_dir / model_filename)
        logger.info("Saved %s weights to %s", name, run_dir / model_filename)

        results[name] = (cv_aurocs, test_auc, probs_test, final_model)

    # ------------------------------------------------------------------
    # 4. Results per model & save artifacts
    # ------------------------------------------------------------------
    summary_lines = []

    for name, (cv_aurocs, test_auc, probs_test, final_model) in results.items():
        text_output, metrics = format_results(name, cv_aurocs, test_auc, y_test, probs_test)
        print(text_output)
        summary_lines.append(text_output)
        all_metrics[name] = metrics

        # Save training curves
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            epochs_axis = list(range(1, len(final_model.history["train_loss"]) + 1))
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Val AUROC"))
            fig.add_trace(
                go.Scatter(x=epochs_axis, y=final_model.history["train_loss"], name="Train Loss"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=epochs_axis, y=final_model.history["val_auroc"], name="Val AUROC"),
                row=1, col=2,
            )
            fig.update_layout(
                title=f"{name} Training Curves",
                xaxis_title="Epoch", xaxis2_title="Epoch",
            )
            curves_filename = f"training_curves_{name.replace(' ', '_').lower()}.html"
            fig.write_html(str(run_dir / curves_filename))
            logger.info("Saved %s training curves to %s", name, run_dir / curves_filename)
        except Exception as e:
            logger.warning("Could not save training curves for %s: %s", name, e)

    # ------------------------------------------------------------------
    # 5. Summary comparison
    # ------------------------------------------------------------------
    summary_header = f"\n{'='*60}\n  SUMMARY\n{'='*60}\n"
    summary_header += f"  {'Model':<22} {'CV AUROC (mean±std)':<24} {'Test AUROC'}\n"
    summary_header += f"  {'-'*58}\n"

    summary_table = []
    for name, (cv_aurocs, test_auc, _, _) in results.items():
        cv_arr = np.array(cv_aurocs)
        line = f"  {name:<22} {cv_arr.mean():.4f} ± {cv_arr.std():.4f}         {test_auc:.4f}"
        summary_table.append(line)

    summary_text = summary_header + "\n".join(summary_table) + "\n"
    print(summary_text)
    summary_lines.append(summary_text)

    # Save summary to file
    with open(run_dir / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    logger.info("Saved summary to %s", run_dir / "summary.txt")

    # Save metrics JSON
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics to %s", run_dir / "cv_results.json")

    print(f"\nAll results saved to: {run_dir}/")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
