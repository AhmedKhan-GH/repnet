"""Cross-lead-attention placement Optuna study (3-stage CrossLead Deeper).

Searches which stages should have CrossLeadAttention enabled. Hypothesis from
the per-class attention diagnostic: stage-1 attention is class-agnostic noise
because the features at that depth (RF~28 ms = sub-QRS) aren't yet meaningful
enough for cross-lead mixing to encode anything class-relevant.

Search space (categorical over per-stage attention bitmaps)
----------------------------------------------------------
  T-T-T   attention at all 3 stages (CURRENT)
  F-T-T   no attention at stage 1 (drop the 'noise' layer)
  F-F-T   attention only at the deepest stage (last 100K params)
  T-F-T   skip middle (unusual but cheap to test)
  T-T-F   attention everywhere except last (controls for the F-F-T contrast)
  F-F-F   pure conv baseline (no cross-lead mixing at all)

Fixed (most-performant config from prior studies)
-------------------------------------------------
  stage_filters: (48, 96, 192)
  kernels:       (7, 5, 3)
  Optimizer:     lr=2.465e-3, dropout=0.0546, weight_decay=1.67e-4
  Augmentation:  aug_sigma=0.060, aug_max_shift=276
  Attention:     n_heads=4
  Loss:          cross_entropy
  Batch / epochs: 64 / 50
  Splits:        patient-grouped 80/20 holdout + 5-fold patient-grouped CV

Usage:
    python -m src.optimize_crosslead_3stage_attn --n-trials 12 --n-folds 5
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

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_crosslead_deeper import (
    RepNetCrossLeadDeeper,
    RepNetCrossLeadDeeperModel,
)
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED = 42

FIXED_PARAMS = dict(
    stage_filters = (48, 96, 192),     # filter study best
    kernels       = (7, 5, 3),
    n_heads       = 4,
    batch_size    = 64,
    epochs        = 50,
    loss_fn       = "cross_entropy",
    lr            = 2.465e-3,
    dropout       = 0.0546,
    weight_decay  = 1.67e-4,
)
AUG_SIGMA     = 0.060
AUG_MAX_SHIFT = 276

# String keys for SQLite-safe categorical storage.
ATTN_CHOICES = ["T-T-T", "F-T-T", "F-F-T", "T-F-T", "T-T-F", "F-F-F"]


def _parse_attn(key: str) -> tuple[bool, ...]:
    return tuple(c == "T" for c in key.split("-"))


def _count_params(attn_stages: tuple[bool, ...]) -> int:
    net = RepNetCrossLeadDeeper(
        stage_filters=FIXED_PARAMS["stage_filters"],
        kernels=FIXED_PARAMS["kernels"],
        dropout=FIXED_PARAMS["dropout"],
        n_heads=FIXED_PARAMS["n_heads"],
        attn_stages=attn_stages,
    )
    return sum(p.numel() for p in net.parameters())


def preprocess(X):
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def quality_filter(X, y, patient_ids, flat_std_thresh=1e-4):
    flat_mask = (X.std(axis=2) < flat_std_thresh).any(axis=1)
    n_flat = int(flat_mask.sum())
    keep = ~flat_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]
    try:
        nan_mask = np.isnan(patient_ids.astype(float))
    except (ValueError, TypeError):
        nan_mask = np.array([str(p).strip() in ("", "nan", "None") for p in patient_ids])
    n_missing = int(nan_mask.sum())
    keep = ~nan_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]
    return X, y, patient_ids, {"flat_lead_dropped": n_flat, "missing_id_dropped": n_missing}


def augment_balance_train(X, y, seed=SEED):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    X_pos_g, _ = GaussianNoise(sigma=AUG_SIGMA).transform(X_pos.copy())
    X_pos_t, _ = RandomTimeShift(max_shift=AUG_MAX_SHIFT).transform(X_pos.copy())
    X_aug = np.concatenate([X_neg, X_pos, X_pos_g, X_pos_t], axis=0)
    y_aug = np.concatenate([
        np.zeros(len(X_neg), dtype=y.dtype),
        np.ones(len(X_pos),  dtype=y.dtype),
        np.ones(len(X_pos),  dtype=y.dtype),
        np.ones(len(X_pos),  dtype=y.dtype),
    ], axis=0)
    X_bal, y_bal = MajorityUndersampling(ratio=1.0, seed=seed).transform(X_aug, y_aug)
    np.random.set_state(rng_state)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


def objective(trial, X_dev, y_dev, folds):
    attn_key = trial.suggest_categorical("attn_stages", ATTN_CHOICES)
    attn_stages = _parse_attn(attn_key)
    n_params = _count_params(attn_stages)

    model_params = {**FIXED_PARAMS, "attn_stages": attn_stages}

    fold_aurocs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr,  y_tr  = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]
        X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=SEED + fold_idx)

        model = RepNetCrossLeadDeeperModel(**model_params)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)

        logger.info("Trial %d attn=%s | fold %d/%d | AUROC=%.4f",
                    trial.number, attn_key, fold_idx + 1, len(folds), auroc)

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc  = float(np.std(fold_aurocs))
    logger.info("Trial %d attn=%s params=%dK | AUROC=%.4f (+/-%.4f)",
                trial.number, attn_key, n_params // 1000, mean_auroc, std_auroc)
    return mean_auroc


def main():
    parser = argparse.ArgumentParser(
        description="Optuna attention-placement search for 3-stage CrossLead Deeper"
    )
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--n-folds",  type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data/seniordesign_upload")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    run_dir = Path("optuna_crosslead_3stage_attn") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    logger.info("Attention-placement candidates and parameter counts:")
    for key in ATTN_CHOICES:
        a = _parse_attn(key)
        logger.info("  %-7s -> attn=%s = %dK params", key, a, _count_params(a) // 1000)

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d",
                len(y), int(y.sum()), int((y == 0).sum()))
    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=0.20, seed=args.seed,
    )
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(),
                len(y_test), 100 * y_test.mean())

    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="crosslead_3stage_attn",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    # Seed current full-attention baseline.
    study.enqueue_trial({"attn_stages": "T-T-T"})

    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds),
        n_trials=args.n_trials,
    )

    best_attn = _parse_attn(study.best_params["attn_stages"])
    best_full = {**FIXED_PARAMS, "attn_stages": best_attn}

    with open(run_dir / "best_params.json", "w") as f:
        json.dump({
            "best_trial":       study.best_trial.number,
            "best_cv_auroc":    study.best_value,
            "best_attn_stages": list(best_attn),
            "best_param_count": _count_params(best_attn),
            "fixed_params":     FIXED_PARAMS,
            "model_params":     {**best_full, "attn_stages": list(best_attn)},
            "aug_sigma":        AUG_SIGMA,
            "aug_max_shift":    AUG_MAX_SHIFT,
        }, f, indent=2, default=str)

    # Final retrain on full dev with patient-grouped 90/10 early-stop split.
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]
    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=args.seed)

    model = RepNetCrossLeadDeeperModel(**best_full)
    model.fit(X_tr, y_tr, X_es, y_es)
    torch.save(model.model.state_dict(), run_dir / "best_model.pt")

    proba = model.predict_proba(X_test)
    test_auroc = roc_auc_score(y_test, proba)
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    youden_thresh = float(thresholds[np.argmax(tpr - fpr)])

    lines = [
        "=" * 60,
        "OPTIMIZATION COMPLETE — RepNet CrossLead Deeper: attention placement",
        "=" * 60,
        f"Best trial:       #{study.best_trial.number}",
        f"Best CV AUROC:    {study.best_value:.4f}",
        f"Best attn_stages: {best_attn}  ({_count_params(best_attn):,} params)",
        "",
        "=" * 60,
        "HOLDOUT TEST",
        "=" * 60,
        f"  AUROC: {test_auroc:.4f}",
        "\n  Threshold = 0.50:",
        classification_report(y_test, (proba >= 0.5).astype(int),
                              target_names=["No PE", "PE"]),
        f"  Threshold = {youden_thresh:.3f} (Youden's J):",
        classification_report(y_test, (proba >= youden_thresh).astype(int),
                              target_names=["No PE", "PE"]),
        "",
        "All trials:",
    ]
    for trial in study.trials:
        if trial.value is None:
            continue
        a = _parse_attn(trial.params["attn_stages"])
        nP = _count_params(a)
        lines.append(
            f"  #{trial.number:3d} | attn={trial.params['attn_stages']} params={nP//1000:4d}K "
            f"| AUROC={trial.value:.4f}"
        )

    summary = "\n".join(lines)
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(summary)

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )
        plot_optimization_history(study).write_html(str(run_dir / "optimization_history.html"))
        plot_slice(study).write_html(str(run_dir / "slice_plot.html"))
        plot_param_importances(study).write_html(str(run_dir / "param_importances.html"))
        logger.info("Saved Optuna visualization plots to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save Optuna plots: %s", e)


if __name__ == "__main__":
    main()
