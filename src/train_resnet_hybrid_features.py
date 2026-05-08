"""Train RepNetResNetHybridFeatures: 4-block ResNet + HRV features at head.

Domain motivation: PE/HDP is an indirect ECG signal -- the disease primarily
affects vascular and renal physiology, with autonomic dysfunction (HRV
changes) and mild repolarization abnormalities as the ECG signature. Pure
conv nets struggle with HRV because R-peak detection is a discrete operation
they're poor at. Adding hand-crafted HRV features at the head (Attia et al.
2019, Mayo Clinic LVEF model) is the standard pattern for this regime.

Pipeline:
  1. Load ECG, apply standard preprocessing (filter + z-score).
  2. Extract HRV features from lead II (cached to disk -- runs once).
  3. Patient-grouped 80/20 holdout + 5-fold CV.
  4. Per fold: z-score features using train statistics, apply to val.
  5. Train ResNet hybrid + features at head with weighted CE.

Outputs (saved to resnet_hybrid_features/<timestamp>/):
  - config.json
  - cv_results.json
  - summary.txt
  - results.log
  - best_model.pt
  - feature_stats.json   (train-fold means/stds used for normalization)

Usage:
    python -m src.train_resnet_hybrid_features
    python -m src.train_resnet_hybrid_features --no-grouped
    python -m src.train_resnet_hybrid_features --data-dir data/seniordesign_upload_balanced
    python -m src.train_resnet_hybrid_features --force-feature-recompute
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
from src.models.repnet_resnet_hybrid_features import RepNetResNetHybridFeaturesModel
from src.preprocessing.augmentation import (
    AmplitudeScaling,
    GaussianNoise,
    RandomTimeShift,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.hrv_features import (
    FEATURE_NAMES,
    N_FEATURES,
    cached_extract,
)
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

PARAMS = dict(
    n_features=N_FEATURES,
    f1=32, f2=64, f3=128, f4=128,
    wide_kernel=7,
    narrow_kernel=5,
    narrow_kernel_2=3,
    dropout=0.2,
    head_hidden=64,
    lr=5e-4,
    weight_decay=1e-3,
    batch_size=64,
    loss_fn="weighted",
)


def preprocess(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def augment_train_with_features(X, y, F, seed: int = SEED, n_copies: int = 2):
    """Concatenative augmentation: keep original X+F + n_copies augmented copies.

    Features are duplicated (not recomputed) for augmented copies. This is fine
    because HRV is time-shift-invariant by construction (RR intervals don't
    change when the whole signal shifts), and amplitude scaling / Gaussian
    noise have negligible effect on R-peak timing.
    """
    parts_X, parts_y, parts_F = [X], [y], [F]
    for i in range(n_copies):
        rng_state = np.random.get_state()
        np.random.seed(seed + i)
        X_aug = X.copy()
        X_aug, _ = GaussianNoise(sigma=0.02).transform(X_aug, None)
        X_aug, _ = AmplitudeScaling(scale_range=0.1).transform(X_aug, None)
        X_aug, _ = RandomTimeShift(max_shift=100).transform(X_aug, None)
        parts_X.append(X_aug)
        parts_y.append(y.copy())
        parts_F.append(F.copy())
        np.random.set_state(rng_state)

    X_out = np.concatenate(parts_X, axis=0)
    y_out = np.concatenate(parts_y, axis=0)
    F_out = np.concatenate(parts_F, axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_out))
    return X_out[idx], y_out[idx], F_out[idx]


def fit_feature_normalizer(F_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-score normalizer on training features. Robust to zero-variance dims."""
    mean = F_train.mean(axis=0)
    std = F_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)   # avoid divide-by-zero
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_normalizer(F: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((F - mean) / std).astype(np.float32)


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def run_cv(params: dict, X_dev, y_dev, F_dev, folds, epochs: int) -> list[float]:
    aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr, F_tr_raw = X_dev[train_idx], y_dev[train_idx], F_dev[train_idx]
        X_val, y_val, F_val_raw = X_dev[val_idx], y_dev[val_idx], F_dev[val_idx]

        # Fit feature normalizer on TRAIN only, apply to val
        mean, std = fit_feature_normalizer(F_tr_raw)
        F_tr = apply_feature_normalizer(F_tr_raw, mean, std)
        F_val = apply_feature_normalizer(F_val_raw, mean, std)

        # Augment after normalization (features are duplicated for aug copies)
        X_tr, y_tr, F_tr = augment_train_with_features(
            X_tr, y_tr, F_tr, seed=SEED + fold_idx,
        )

        logger.info(
            "  Fold %d/%d - train=%d (pos=%d neg=%d) val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr),
            int((y_tr == 1).sum()), int((y_tr == 0).sum()),
            len(y_val), int((y_val == 1).sum()), int((y_val == 0).sum()),
        )
        model = RepNetResNetHybridFeaturesModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val, F_train=F_tr, F_val=F_val)
        auc = model.score(X_val, y_val, F=F_val)
        aurocs.append(auc)
        print(f"  -> Fold {fold_idx+1} AUROC: {auc:.4f}")
    return aurocs


def train_final(params: dict, X_dev, y_dev, F_dev, epochs: int):
    X_tr, X_es, y_tr, y_es, F_tr_raw, F_es_raw = train_test_split(
        X_dev, y_dev, F_dev,
        test_size=0.10, stratify=y_dev, random_state=SEED,
    )
    mean, std = fit_feature_normalizer(F_tr_raw)
    F_tr = apply_feature_normalizer(F_tr_raw, mean, std)
    F_es = apply_feature_normalizer(F_es_raw, mean, std)

    X_tr, y_tr, F_tr = augment_train_with_features(X_tr, y_tr, F_tr, seed=SEED)
    logger.info(
        "Final training: %d train (augmented) + %d early-stop (clean)",
        len(y_tr), len(y_es),
    )
    model = RepNetResNetHybridFeaturesModel(**params, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es, F_train=F_tr, F_val=F_es)
    return model, mean, std


def evaluate(model, X_test, y_test, F_test_normalized) -> dict:
    probs = model.predict_proba(X_test, F=F_test_normalized)
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
        description="Train RepNetResNetHybridFeatures (ResNet + HRV features at head)",
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-grouped", action="store_true",
                        help="Disable patient-grouped splits (default: grouped)")
    parser.add_argument("--force-feature-recompute", action="store_true",
                        help="Recompute HRV features even if cache exists")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    args = parser.parse_args()

    grouped = not args.no_grouped
    params = dict(PARAMS)
    if args.dropout is not None: params["dropout"] = args.dropout
    if args.lr is not None: params["lr"] = args.lr
    if args.weight_decay is not None: params["weight_decay"] = args.weight_decay

    run_dir = Path("resnet_hybrid_features") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    if grouped:
        X, y, groups = load_seniordesign(args.data_dir, return_patient_ids=True)
        valid = ~np.isnan(groups.astype(float))
        n_drop = int((~valid).sum())
        if n_drop:
            logger.info("Dropping %d rows with missing Pat_Obfus_MRN", n_drop)
            X, y, groups = X[valid], y[valid], groups[valid]
    else:
        X, y = load_seniordesign(args.data_dir)
        groups = None

    logger.info(
        "Loaded %d samples (pos=%d, neg=%d, pos_rate=%.1f%%)",
        len(y), int((y == 1).sum()), int((y == 0).sum()), 100 * y.mean(),
    )
    X = preprocess(X)

    # Cache HRV features. Cache key: data_dir + sample count (content-hashed).
    cache_key = args.data_dir.replace("/", "_").replace("\\", "_")
    feature_cache = Path(".cache_hrv") / f"{cache_key}_n{len(y)}.npy"
    F = cached_extract(X, feature_cache, force=args.force_feature_recompute)
    logger.info(
        "HRV features: shape=%s  per-feature stats:\n%s",
        F.shape,
        "\n".join(
            f"  {name:<14} mean={F[:, i].mean():>10.3f}  std={F[:, i].std():>10.3f}  "
            f"min={F[:, i].min():>10.3f}  max={F[:, i].max():>10.3f}"
            for i, name in enumerate(FEATURE_NAMES)
        ),
    )

    if grouped:
        # split_holdout_grouped operates on (X, y, groups). We'll align F by index.
        idx_all = np.arange(len(y))
        from src.data.dataset import split_holdout_grouped, kfold_cv_indices_grouped
        # Use a 1-shot grouped split that returns INDICES so F follows along
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
        dev_idx, test_idx = next(splitter.split(X, y, groups))
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        F_dev, F_test = F[dev_idx], F[test_idx]
        g_dev = groups[dev_idx]
        # Sanity: patient sets must be disjoint
        if set(g_dev).intersection(set(groups[test_idx])):
            raise RuntimeError("Patient leakage between dev and test")
        folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)
    else:
        from src.data.dataset import split_holdout, kfold_cv_indices
        from sklearn.model_selection import train_test_split as tts
        X_dev, X_test, y_dev, y_test, F_dev, F_test = tts(
            X, y, F, test_size=0.20, stratify=y, random_state=args.seed,
        )
        folds = kfold_cv_indices(y_dev, n_folds=args.n_folds, seed=args.seed)

    logger.info(
        "Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)  splits=%s",
        len(y_dev), 100 * y_dev.mean(), len(y_test), 100 * y_test.mean(),
        "grouped" if grouped else "stratified",
    )

    config = {
        "data_dir": args.data_dir,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "grouped": grouped,
        "n_total": int(len(y)),
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "pos_rate_dev": float(y_dev.mean()),
        "pos_rate_test": float(y_test.mean()),
        "feature_names": list(FEATURE_NAMES),
        "params": params,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'#'*72}\n  RepNetResNetHybridFeatures - 4-block ResNet + HRV features\n{'#'*72}")
    cv_aurocs = run_cv(params, X_dev, y_dev, F_dev, folds, epochs=args.epochs)

    print("\n  Retraining on full dev ...")
    final_model, mean_final, std_final = train_final(
        params, X_dev, y_dev, F_dev, epochs=args.epochs,
    )
    F_test_norm = apply_feature_normalizer(F_test, mean_final, std_final)
    eval_metrics = evaluate(final_model, X_test, y_test, F_test_norm)

    torch.save(final_model.model.state_dict(), run_dir / "best_model.pt")
    with open(run_dir / "feature_stats.json", "w") as f:
        json.dump({
            "feature_names": list(FEATURE_NAMES),
            "mean": mean_final.tolist(),
            "std": std_final.tolist(),
        }, f, indent=2)

    results = {
        "cv_aurocs": cv_aurocs,
        "cv_mean": float(np.mean(cv_aurocs)),
        "cv_std": float(np.std(cv_aurocs)),
        **eval_metrics,
    }
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "RepNetResNetHybridFeatures - 4-block ResNet + HRV/QT features at head",
        f"  Splits:        {'PATIENT-GROUPED' if grouped else 'STRATIFIED (ungrouped)'}",
        f"  Data:          {args.data_dir}",
        f"  Total/Dev/Test: {len(y)} / {len(y_dev)} / {len(y_test)}",
        f"  Dev pos rate:  {100 * y_dev.mean():.1f}%",
        f"  Folds:         {args.n_folds}",
        f"  Epochs/fold:   {args.epochs}",
        f"  HRV features:  {', '.join(FEATURE_NAMES)}",
        f"  Filters:       F1={params['f1']} F2={params['f2']} F3={params['f3']} F4={params['f4']}",
        f"  Head hidden:   {params['head_hidden']}",
        f"  lr:            {params['lr']:.6f}",
        f"  weight_decay:  {params['weight_decay']:.6f}",
        f"  dropout:       {params['dropout']:.4f}",
        f"  loss_fn:       {params['loss_fn']}",
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
