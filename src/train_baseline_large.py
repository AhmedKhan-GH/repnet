"""RepNet Baseline Large — 80/20 + 5-fold CV evaluation for 3-layer model.

Evaluation protocol:
  1. Stratified 80/20 holdout split — test set is never touched until final eval
  2. 5-fold stratified CV on the 80% dev set
     - Signal conditioning (notch 60 Hz + bandpass 0.5–40 Hz + z-score) applied once
     - Augmentation (noise, amplitude scaling, time shift) applied per-fold on train only
  3. Final model retrained on all 80% dev data, evaluated on 20% test holdout
  4. Results printed (AUROC + classification metrics)

Usage:
    python -m src.train_baseline_large
    python -m src.train_baseline_large --data-dir "data/seniordesign_upload"
    python -m src.train_baseline_large --n-folds 5 --epochs 50
"""

import argparse
import logging

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from src.data.dataset import kfold_cv_indices, load_seniordesign, split_holdout
from src.models.repnet_baseline_large import RepNetBaselineLargeModel
from src.preprocessing.filters import NotchFilter, BaselineWanderFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.augmentation import GaussianNoise, AmplitudeScaling, RandomTimeShift

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed hyperparameters (EDA-informed choices, extended to 3 layers)
# ---------------------------------------------------------------------------
REPNET_BASELINE_LARGE_PARAMS = dict(
    stage_filters=(32, 64, 128),
    wide_kernel=7,
    narrow_kernel=5,
    narrow_kernel_2=3,
    dropout=0.15,
    lr=5e-4,
    batch_size=64,
    loss_fn="focal",
    focal_alpha=0.75,        # 75% weight on PE class (false negatives are costly)
    focal_gamma=2.0,         # Standard focal loss gamma
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
           folds, epochs: int) -> list[float]:
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

        model = RepNetBaselineLargeModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = model.score(X_val, y_val)
        aurocs.append(auc)
        print(f"  → Fold {fold_idx+1} AUROC: {auc:.4f}")

    return aurocs


def train_final(params: dict, X_dev: np.ndarray, y_dev: np.ndarray,
                epochs: int, seed: int = SEED):
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
    model = RepNetBaselineLargeModel(**params, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    return model


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Find threshold that maximises Youden's J = sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def print_results(name: str, cv_aurocs: list[float],
                  test_auc: float, y_test: np.ndarray, probs_test: np.ndarray):
    """Print CV summary and test evaluation."""
    cv_arr = np.array(cv_aurocs)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  5-fold CV AUROC: {cv_arr.mean():.4f} ± {cv_arr.std():.4f}")
    print(f"  Per-fold:        {[f'{v:.4f}' for v in cv_aurocs]}")
    print(f"  Test AUROC:      {test_auc:.4f}")
    print()
    # 0.5 threshold
    preds_05 = (probs_test >= 0.5).astype(int)
    print("  Classification report (threshold=0.50):")
    print(classification_report(y_test, preds_05, target_names=["No PE", "PE"]))
    # Youden's J threshold
    thresh_j = youden_threshold(y_test, probs_test)
    preds_j = (probs_test >= thresh_j).astype(int)
    print(f"  Classification report (Youden's J threshold={thresh_j:.3f}):")
    print(classification_report(y_test, preds_j, target_names=["No PE", "PE"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RepNet Baseline Large 5-fold CV evaluation")
    parser.add_argument("--data-dir", default="data/seniordesign_upload_balanced")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

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

    # ------------------------------------------------------------------
    # 3. 5-fold CV
    # ------------------------------------------------------------------
    folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=SEED)

    print(f"\n{'#'*60}")
    print(f"  RepNet Baseline Large — {args.n_folds}-fold CV")
    print(f"{'#'*60}")
    cv_aurocs = run_cv(REPNET_BASELINE_LARGE_PARAMS, X_dev, y_dev, folds, epochs=args.epochs)

    print(f"\n  Retraining RepNet Baseline Large on full dev set …")
    final_model = train_final(REPNET_BASELINE_LARGE_PARAMS, X_dev, y_dev, epochs=args.epochs)
    probs_test = final_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, probs_test)

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    print_results("RepNet Baseline Large", cv_aurocs, test_auc, y_test, probs_test)


if __name__ == "__main__":
    main()
