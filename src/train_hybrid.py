"""RepNet Hybrid — 80/20 + 5-fold CV evaluation.

Usage:
    python -m src.train_hybrid
"""

import argparse
import logging

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.data.dataset import kfold_cv_indices, load_seniordesign, split_holdout
from src.models.repnet_hybrid import RepNetHybridModel
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.augmentation import GaussianNoise, AmplitudeScaling, RandomTimeShift

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

PARAMS = dict(
    stage_filters=(32, 64),
    wide_kernel=7,
    narrow_kernel=5,
    dropout=0.15,
    n_heads=4,
    lr=5e-4,
    batch_size=64,
    epochs=50,
    loss_fn="weighted",
)


def preprocess(X):
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def augment_train(X, y, seed=SEED, n_copies=0):
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
    X_out = np.concatenate(parts_X)
    y_out = np.concatenate(parts_y)
    idx = np.random.default_rng(seed).permutation(len(y_out))
    return X_out[idx], y_out[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    X, y = load_seniordesign(args.data_dir)
    X = preprocess(X)

    X_dev, X_test, y_dev, y_test = split_holdout(X, y, test_size=0.20, seed=SEED)
    folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=SEED)

    logger.info("Dev: %d  Test: %d", len(y_dev), len(y_test))

    params = {**PARAMS, "epochs": args.epochs}

    print(f"\n{'#'*60}")
    print(f"  RepNet Hybrid — {args.n_folds}-fold CV")
    print(f"{'#'*60}")
    aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)

        model = RepNetHybridModel(**params)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = model.score(X_val, y_val)
        aurocs.append(auc)
        print(f"  → Fold {fold_idx+1} AUROC: {auc:.4f}")

    print(f"\n  Retraining on full dev set …")
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=SEED
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED)
    model = RepNetHybridModel(**params)
    model.fit(X_tr, y_tr, X_es, y_es)

    proba = model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, proba)

    cv = np.array(aurocs)
    print(f"\n{'='*60}")
    print(f"  RepNet Hybrid")
    print(f"{'='*60}")
    print(f"  5-fold CV AUROC: {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"  Per-fold:        {[f'{v:.4f}' for v in aurocs]}")
    print(f"  Test AUROC:      {test_auc:.4f}")
    print()

    preds_05 = (proba >= 0.5).astype(int)
    print("  Classification report (threshold=0.50):")
    print(classification_report(y_test, preds_05, target_names=["No PE", "PE"]))

    fpr, tpr, thresholds = roc_curve(y_test, proba)
    thresh_j = float(thresholds[np.argmax(tpr - fpr)])
    preds_j = (proba >= thresh_j).astype(int)
    print(f"  Classification report (Youden's J threshold={thresh_j:.3f}):")
    print(classification_report(y_test, preds_j, target_names=["No PE", "PE"]))


if __name__ == "__main__":
    main()
