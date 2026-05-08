"""Train RepNetResNetHybrid (4-block 1D ResNet, first block per-lead).

No attention. Strong regularization defaults (dropout=0.2, weight_decay=1e-3,
lr=5e-4) appropriate for ~1700 dev samples with channel-mixed mid/late convs.

Default protocol:
  - Full unbalanced dataset (data/seniordesign_upload, ~85/15)
  - Patient-grouped 80/20 holdout + 5-fold CV
  - Weighted CE loss
  - Basic 3x augmentation, no over/undersampling

Outputs (saved to resnet_hybrid/<timestamp>/):
  - config.json
  - cv_results.json
  - summary.txt
  - results.log
  - best_model.pt    (retrained on full dev)

Usage:
    python -m src.train_resnet_hybrid
    python -m src.train_resnet_hybrid --no-grouped
    python -m src.train_resnet_hybrid --dropout 0.3 --weight-decay 5e-3
    python -m src.train_resnet_hybrid --data-dir data/seniordesign_upload_balanced
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
from src.models.repnet_resnet_hybrid import RepNetResNetHybridModel
from src.preprocessing.augmentation import (
    AmplitudeScaling,
    GaussianNoise,
    RandomTimeShift,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

RESNET_PARAMS = dict(
    f1=32, f2=64, f3=128, f4=128,
    wide_kernel=7,
    narrow_kernel=5,
    narrow_kernel_2=3,
    dropout=0.2,
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
    return float(thresholds[np.argmax(tpr - fpr)])


def run_cv(params: dict, X_dev, y_dev, folds, epochs: int) -> list[float]:
    aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)
        logger.info(
            "  Fold %d/%d - train=%d (pos=%d neg=%d) val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr),
            int((y_tr == 1).sum()), int((y_tr == 0).sum()),
            len(y_val), int((y_val == 1).sum()), int((y_val == 0).sum()),
        )
        model = RepNetResNetHybridModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = model.score(X_val, y_val)
        aurocs.append(auc)
        print(f"  -> Fold {fold_idx+1} AUROC: {auc:.4f}")
    return aurocs


def train_final(params: dict, X_dev, y_dev, epochs: int):
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=SEED,
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED)
    logger.info(
        "Final training: %d train (augmented) + %d early-stop (clean)",
        len(y_tr), len(y_es),
    )
    model = RepNetResNetHybridModel(**params, epochs=epochs)
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
        description="Train RepNetResNetHybrid (4-block, first block per-lead, no attention)",
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-grouped", action="store_true",
                        help="Disable patient-grouped splits (default: grouped)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout (default 0.2)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default 5e-4)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight decay (default 1e-3)")
    args = parser.parse_args()

    grouped = not args.no_grouped
    params = dict(RESNET_PARAMS)
    if args.dropout is not None:
        params["dropout"] = args.dropout
    if args.lr is not None:
        params["lr"] = args.lr
    if args.weight_decay is not None:
        params["weight_decay"] = args.weight_decay

    run_dir = Path("resnet_hybrid") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    if grouped:
        X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
            X, y, groups, test_size=0.20, seed=args.seed,
        )
        folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)
    else:
        X_dev, X_test, y_dev, y_test = split_holdout(X, y, test_size=0.20, seed=args.seed)
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
        "params": params,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'#'*72}\n  RepNetResNetHybrid - 4-block ResNet, first block per-lead, no attention\n{'#'*72}")
    cv_aurocs = run_cv(params, X_dev, y_dev, folds, epochs=args.epochs)

    print("\n  Retraining on full dev ...")
    final_model = train_final(params, X_dev, y_dev, epochs=args.epochs)
    eval_metrics = evaluate(final_model, X_test, y_test)

    torch.save(final_model.model.state_dict(), run_dir / "best_model.pt")
    logger.info("Saved final weights to %s", run_dir / "best_model.pt")

    results = {
        "cv_aurocs": cv_aurocs,
        "cv_mean": float(np.mean(cv_aurocs)),
        "cv_std": float(np.std(cv_aurocs)),
        **eval_metrics,
    }
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "RepNetResNetHybrid - 4-block ResNet, first block per-lead, no attention",
        f"  Splits:        {'PATIENT-GROUPED' if grouped else 'STRATIFIED (ungrouped)'}",
        f"  Data:          {args.data_dir}",
        f"  Total/Dev/Test: {len(y)} / {len(y_dev)} / {len(y_test)}",
        f"  Dev pos rate:  {100 * y_dev.mean():.1f}%",
        f"  Folds:         {args.n_folds}",
        f"  Epochs/fold:   {args.epochs}",
        f"  Filters:       F1={params['f1']} F2={params['f2']} F3={params['f3']} F4={params['f4']}",
        f"  Kernels:       {params['wide_kernel']} | {params['narrow_kernel']} | {params['narrow_kernel']} | {params['narrow_kernel_2']}",
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
