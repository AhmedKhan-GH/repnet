"""Optuna study: loss function + sampling ratio for PE-optimised CrossLead Deeper.

Fixes the best architecture (48,96,192) and optimizer (lr=2.465e-3) from prior
studies and searches over the two axes most likely to improve PE recall given
the new symmetric augmentation:

  loss_fn          cross_entropy | weighted | focal
  focal_gamma      [0.5, 3.0]   (only when focal)
  focal_alpha      [0.1, 0.9]   (only when focal)
  undersample_ratio [0.3, 4.0]  — <1 = more PE than Normal, >1 = more Normal
  aug_sigma        [0.005, 0.20] log-uniform
  aug_max_shift    [25, 600]

Fixed (well-tuned from prior studies):
  stage_filters=(48,96,192), kernels=(7,5,3), n_heads=4,
  lr=2.465e-3, dropout=0.0546, weight_decay=1.67e-4, batch_size=64

Augmentation strategy (from multiseed_pe):
  Both PE and Normal are doubled with seeded noise+shift, then the majority
  class is undersampled to `undersample_ratio * n_minority`.

Early stopping: patience on val AUPRC (not AUROC).
Objective:      mean val AUPRC across --n-folds folds (default 3).

Outputs (optuna_pe_loss_sampling/YYYY-MM-DD_HH-MM-SS/):
  study.db, results.log, best_params.json, best_model.pt, summary.txt,
  optuna_*_plot.html

Usage:
    python -m src.optimize_crosslead_pe_loss_sampling
    python -m src.optimize_crosslead_pe_loss_sampling --n-trials 60 --n-folds 3
    python -m src.optimize_crosslead_pe_loss_sampling --n-trials 20 --n-folds 1  # fast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.models.repnet_crosslead_deeper import (
    RepNetCrossLeadDeeper,
    RepNetCrossLeadDeeperModel,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SPLIT_SEED = 42

FIXED_PARAMS = dict(
    stage_filters = (48, 96, 192),
    kernels       = (7, 5, 3),
    n_heads       = 4,
    lr            = 2.465e-3,
    dropout       = 0.0546,
    weight_decay  = 1.67e-4,
    batch_size    = 64,
)

MAX_EPOCHS = 80
PATIENCE   = 10


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for binary classification via 2-class softmax logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: weight for the positive (PE) class.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        p_t   = probs[torch.arange(len(targets)), targets]
        alpha_t = torch.where(targets == 1,
                              torch.full_like(p_t, self.alpha),
                              torch.full_like(p_t, 1.0 - self.alpha))
        loss = -alpha_t * (1.0 - p_t) ** self.gamma * torch.log(p_t.clamp(min=1e-8))
        return loss.mean()


def _build_criterion(loss_fn: str, y_train: np.ndarray, device,
                     focal_gamma: float = 2.0, focal_alpha: float = 0.25):
    if loss_fn == "weighted":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=w)
    if loss_fn == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Augmentation (symmetric, seeded)
# ---------------------------------------------------------------------------

def _augment_batch(X: np.ndarray, rng: np.random.Generator,
                   sigma: float, max_shift: int) -> np.ndarray:
    X_aug = X.copy() + rng.normal(0, sigma, X.shape).astype(X.dtype)
    N, C, T = X_aug.shape
    shifts = rng.integers(-max_shift, max_shift + 1, size=N)
    out = np.zeros_like(X_aug)
    for i, s in enumerate(shifts):
        if s > 0:
            out[i, :, s:] = X_aug[i, :, :T - s]
        elif s < 0:
            out[i, :, :T + s] = X_aug[i, :, -s:]
        else:
            out[i] = X_aug[i]
    return out


def augment_balance(X: np.ndarray, y: np.ndarray,
                    sigma: float, max_shift: int,
                    undersample_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Double both classes symmetrically, then undersample to target ratio."""
    rng = np.random.default_rng(seed)
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    X_combined = np.concatenate([
        X_neg, _augment_batch(X_neg, rng, sigma, max_shift),
        X_pos, _augment_batch(X_pos, rng, sigma, max_shift),
    ], axis=0)
    y_combined = np.concatenate([
        np.zeros(len(X_neg) * 2, dtype=y.dtype),
        np.ones(len(X_pos)  * 2, dtype=y.dtype),
    ], axis=0)
    X_bal, y_bal = MajorityUndersampling(
        ratio=undersample_ratio, seed=seed,
    ).transform(X_combined, y_combined)
    idx = rng.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def quality_filter(X, y, patient_ids, flat_std_thresh=1e-4):
    flat_mask = (X.std(axis=2) < flat_std_thresh).any(axis=1)
    keep = ~flat_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]
    try:
        nan_mask = np.isnan(patient_ids.astype(float))
    except (ValueError, TypeError):
        nan_mask = np.array([str(p).strip() in ("", "nan", "None") for p in patient_ids])
    keep = ~nan_mask
    return X[keep], y[keep], patient_ids[keep]


def preprocess(X):
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X.copy())
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


# ---------------------------------------------------------------------------
# Trial trainer (AUPRC early stopping)
# ---------------------------------------------------------------------------

def train_fold(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> float:
    """Train one fold and return val AUPRC of the best-checkpoint model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    net = RepNetCrossLeadDeeper(
        stage_filters = FIXED_PARAMS["stage_filters"],
        kernels       = FIXED_PARAMS["kernels"],
        n_heads       = FIXED_PARAMS["n_heads"],
        dropout       = FIXED_PARAMS["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=FIXED_PARAMS["lr"], betas=(0.9, 0.999), eps=1e-7,
        weight_decay=FIXED_PARAMS["weight_decay"],
    )
    criterion = _build_criterion(
        params["loss_fn"], y_tr, device,
        focal_gamma=params.get("focal_gamma", 2.0),
        focal_alpha=params.get("focal_alpha", 0.25),
    )

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=FIXED_PARAMS["batch_size"], shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available(),
        persistent_workers=torch.cuda.is_available(), generator=gen,
    )
    Xv_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_auprc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        net.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        net.eval()
        with torch.no_grad():
            val_probs = torch.softmax(net(Xv_t), dim=1)[:, 1].cpu().numpy()
        avg_loss  = epoch_loss / n_batches
        val_auprc = average_precision_score(y_val, val_probs)
        val_auroc = roc_auc_score(y_val, val_probs)

        marker = ""
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
            marker     = " *"
        else:
            no_improve += 1

        print(f"    Epoch {epoch+1:3d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | "
              f"val_AUROC={val_auroc:.4f} | val_AUPRC={val_auprc:.4f}{marker}"
              + (f"  [pat {no_improve}/{PATIENCE}]" if no_improve > 0 else ""))

        if no_improve >= PATIENCE:
            print(f"    Early stop — no AUPRC improvement for {PATIENCE} epochs")
            break

    # Restore best weights and return best val AUPRC
    if best_state is not None:
        net.load_state_dict(best_state)
    return best_auprc, net


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, X_dev, y_dev, g_dev, folds) -> float:
    loss_fn = trial.suggest_categorical("loss_fn", ["cross_entropy", "weighted", "focal"])
    params  = {"loss_fn": loss_fn}

    if loss_fn == "focal":
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 3.0)
        params["focal_alpha"] = trial.suggest_float("focal_alpha", 0.1, 0.9)

    undersample_ratio = trial.suggest_float("undersample_ratio", 0.3, 4.0, log=True)
    aug_sigma         = trial.suggest_float("aug_sigma",         0.005, 0.20, log=True)
    aug_max_shift     = trial.suggest_int("aug_max_shift",       25, 600)

    fold_auprcs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr,  y_tr  = X_dev[tr_idx], y_dev[tr_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        g_tr         = g_dev[tr_idx]

        # Inner early-stop split (10% of fold-train, grouped)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10,
                                     random_state=SPLIT_SEED + fold_idx)
        train_idx2, es_idx = next(splitter.split(X_tr, y_tr, g_tr))
        X_tr2, y_tr2 = X_tr[train_idx2], y_tr[train_idx2]
        X_es,  y_es  = X_tr[es_idx],     y_tr[es_idx]

        X_tr2, y_tr2 = augment_balance(
            X_tr2, y_tr2, aug_sigma, aug_max_shift,
            undersample_ratio, seed=SPLIT_SEED + fold_idx,
        )

        seed = SPLIT_SEED + trial.number * 7 + fold_idx
        val_auprc, _ = train_fold(X_tr2, y_tr2, X_es, y_es, params, seed=seed)
        fold_auprcs.append(val_auprc)

        logger.info(
            "Trial %3d %s ratio=%.2f sigma=%.4f shift=%d | fold %d/%d | AUPRC=%.4f",
            trial.number, loss_fn, undersample_ratio, aug_sigma, aug_max_shift,
            fold_idx + 1, len(folds), val_auprc,
        )

        # Intermediate pruning
        trial.report(float(np.mean(fold_auprcs)), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_auprc = float(np.mean(fold_auprcs))
    std_auprc  = float(np.std(fold_auprcs))
    logger.info(
        "Trial %3d %s ratio=%.2f sigma=%.4f shift=%d | AUPRC=%.4f +/-%.4f",
        trial.number, loss_fn, undersample_ratio, aug_sigma, aug_max_shift,
        mean_auprc, std_auprc,
    )
    return mean_auprc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna PE-loss + sampling search for CrossLead Deeper"
    )
    parser.add_argument("--n-trials",  type=int, default=50)
    parser.add_argument("--n-folds",   type=int, default=5,
                        help="CV folds per trial (1=single GroupShuffleSplit, faster).")
    parser.add_argument("--data-dir",  default="data/seniordesign_upload")
    parser.add_argument("--seed",      type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    run_dir = (
        Path("optuna_pe_loss_sampling") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "results.log", encoding="utf-8"),
        ],
    )
    logger.info("Output: %s", run_dir)
    logger.info("Fixed arch: %s", FIXED_PARAMS)
    logger.info("max_epochs=%d  patience=%d", MAX_EPOCHS, PATIENCE)

    X, y, pids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, pids = quality_filter(X, y, pids)
    logger.info("After QC: N=%d  pos=%d  neg=%d", len(y), int(y.sum()), int((y == 0).sum()))
    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, pids, test_size=0.20, seed=args.seed,
    )
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(), len(y_test), 100 * y_test.mean())

    # Build folds once, reused for every trial
    if args.n_folds == 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
        tr_idx, val_idx = next(splitter.split(X_dev, y_dev, g_dev))
        folds = [(tr_idx, val_idx)]
    else:
        from src.data.dataset import kfold_cv_indices_grouped
        folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)

    storage = f"sqlite:///{run_dir / 'study.db'}"
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(
        study_name    = "pe_loss_sampling",
        direction     = "maximize",
        sampler       = TPESampler(seed=args.seed),
        pruner        = pruner,
        storage       = storage,
        load_if_exists= True,
    )

    # Enqueue baseline (current best config) as first trial
    study.enqueue_trial({
        "loss_fn":           "cross_entropy",
        "undersample_ratio": 1.0,
        "aug_sigma":         0.060,
        "aug_max_shift":     276,
    })
    # Enqueue focal baseline
    study.enqueue_trial({
        "loss_fn":           "focal",
        "focal_gamma":       2.0,
        "focal_alpha":       0.25,
        "undersample_ratio": 1.0,
        "aug_sigma":         0.060,
        "aug_max_shift":     276,
    })
    # Enqueue weighted baseline
    study.enqueue_trial({
        "loss_fn":           "weighted",
        "undersample_ratio": 1.0,
        "aug_sigma":         0.060,
        "aug_max_shift":     276,
    })

    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, g_dev, folds),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    logger.info("Best trial #%d | AUPRC=%.4f | params=%s",
                best.number, best.value, best.params)

    # Save best params
    with open(run_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "trial_number": best.number,
            "val_auprc":    best.value,
            "params":       best.params,
            "fixed":        {k: list(v) if isinstance(v, tuple) else v
                             for k, v in FIXED_PARAMS.items()},
        }, f, indent=2)

    # Retrain best config on full dev set, evaluate on test
    best_p = best.params
    aug_sigma     = best_p["aug_sigma"]
    aug_max_shift = best_p["aug_max_shift"]
    us_ratio      = best_p["undersample_ratio"]

    # Inner val split for early stopping on full dev retrain
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    tr_idx2, es_idx2 = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr_full, y_tr_full = X_dev[tr_idx2], y_dev[tr_idx2]
    X_es_full,  y_es_full = X_dev[es_idx2],  y_dev[es_idx2]
    X_tr_full, y_tr_full = augment_balance(
        X_tr_full, y_tr_full, aug_sigma, aug_max_shift, us_ratio, seed=args.seed,
    )

    params_full = {
        "loss_fn":     best_p["loss_fn"],
        "focal_gamma": best_p.get("focal_gamma", 2.0),
        "focal_alpha": best_p.get("focal_alpha", 0.25),
    }
    test_auprc_val, best_net = train_fold(
        X_tr_full, y_tr_full, X_es_full, y_es_full, params_full, seed=args.seed,
    )
    logger.info("Best config retrain — val AUPRC=%.4f", test_auprc_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_net.to(device).eval()
    Xt_test = torch.tensor(X_test, dtype=torch.float32)
    dl_test = DataLoader(TensorDataset(Xt_test), batch_size=64)
    test_probs = []
    with torch.no_grad():
        for (xb,) in dl_test:
            test_probs.append(torch.softmax(best_net(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    test_probs = np.concatenate(test_probs)

    test_auroc = float(roc_auc_score(y_test, test_probs))
    test_auprc = float(average_precision_score(y_test, test_probs))
    logger.info("Test — AUROC=%.4f  AUPRC=%.4f", test_auroc, test_auprc)

    torch.save({k: v.cpu() for k, v in best_net.state_dict().items()},
               run_dir / "best_model.pt")

    summary = "\n".join([
        "",
        "=" * 60,
        "  Optuna PE loss + sampling study",
        "=" * 60,
        f"  Trials completed : {len(study.trials)}",
        f"  Best trial       : #{best.number}",
        f"  Best val AUPRC   : {best.value:.4f}",
        f"  Best params      : {best.params}",
        "",
        f"  Retrain on full dev — test AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}",
        "",
        f"  Test : N={len(y_test)}  pos={int(y_test.sum())}  neg={int((y_test==0).sum())}",
    ])
    print(summary)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    # Plotly visualisations
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
        plot_optimization_history(study).write_html(str(run_dir / "optuna_history.html"))
        plot_param_importances(study).write_html(str(run_dir / "optuna_importance.html"))
        plot_parallel_coordinate(study).write_html(str(run_dir / "optuna_parallel.html"))
        plot_slice(study).write_html(str(run_dir / "optuna_slice.html"))
        logger.info("Saved Optuna plots to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save Optuna plots: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
