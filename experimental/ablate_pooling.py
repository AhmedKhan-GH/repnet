"""Ablation: Global Average Pool vs Max Pool vs Avg+Max concat.

Modifies only the pooling layer in RepNetCrossLeadDeeper and runs the
same 3-fold patient-grouped CV used in the main training pipeline.

Usage (from repo root):
    python -m experimental.ablate_pooling
    python -m experimental.ablate_pooling --n-folds 5 --epochs 30
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
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeper
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED = 42

BEST_PARAMS = dict(
    stage_filters=(48, 96, 192),
    kernels=(7, 5, 3),
    dropout=0.0546,
    n_heads=4,
    lr=0.002465,
    weight_decay=0.000167,
    batch_size=64,
    epochs=50,
    loss_fn="cross_entropy",
)

AUG_SIGMA = 0.06
AUG_MAX_SHIFT = 276


# ---------------------------------------------------------------------------
#  Modified model with configurable pooling
# ---------------------------------------------------------------------------

class RepNetCrossLeadDeeperPooling(RepNetCrossLeadDeeper):
    """RepNetCrossLeadDeeper with swappable global pooling strategy."""

    def __init__(self, pool_mode: str = "avg", **kwargs):
        super().__init__(**kwargs)
        self.pool_mode = pool_mode

        f_last = kwargs.get("stage_filters", (32, 64, 128))[-1]

        if pool_mode == "avg":
            self.gap = nn.AdaptiveAvgPool1d(1)
            self._pool_out = f_last
        elif pool_mode == "max":
            self.gap = nn.AdaptiveMaxPool1d(1)
            self._pool_out = f_last
        elif pool_mode == "avg+max":
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self._pool_out = f_last * 2
            self.fc = nn.Linear(self._pool_out, kwargs.get("n_classes", 2))
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        for stage in self.stages:
            x = stage["conv"](x)
            if "attn" in stage:
                x = stage["attn"](x)
        B, L, F, T_out = x.shape
        x = x.reshape(B, L * F, T_out)
        x = self.fuse(x)

        if self.pool_mode == "avg+max":
            x_avg = self.avg_pool(x).squeeze(-1)
            x_max = self.max_pool(x).squeeze(-1)
            x = torch.cat([x_avg, x_max], dim=1)
        else:
            x = self.gap(x).squeeze(-1)

        return self.fc(self.head_drop(x))


# ---------------------------------------------------------------------------
#  Thin training wrapper (mirrors RepNetCrossLeadDeeperModel)
# ---------------------------------------------------------------------------

class PoolingAblationModel:
    def __init__(self, pool_mode: str, **kwargs):
        self.pool_mode = pool_mode
        self.net_params = {
            k: kwargs[k]
            for k in ("stage_filters", "kernels", "dropout", "n_heads")
            if k in kwargs
        }
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 50)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = RepNetCrossLeadDeeperPooling(
            pool_mode=self.pool_mode, **self.net_params,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            betas=(0.9, 0.999), eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        from torch.utils.data import DataLoader, TensorDataset

        use_cuda = self.device.type == "cuda"
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        train_dl = DataLoader(
            TensorDataset(Xt, yt), batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 if use_cuda else 0,
            pin_memory=use_cuda,
            persistent_workers=use_cuda,
        )
        Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        best_val_auc, best_state = 0.0, None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in train_dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            val_auc = self._score(Xv, y_val)
            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            print(f"    Epoch {epoch+1:3d}/{self.epochs} | "
                  f"loss={epoch_loss/n_batches:.4f} | val_AUROC={val_auc:.4f}{marker}")

            if patience_counter >= 10:
                print("    Early stop")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    @torch.no_grad()
    def _score(self, Xv, y_val):
        self.model.eval()
        probs = torch.softmax(self.model(Xv), dim=1)[:, 1].cpu().numpy()
        return roc_auc_score(y_val, probs)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset
        use_cuda = self.device.type == "cuda"
        Xt = torch.tensor(X, dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt), batch_size=self.batch_size,
                        num_workers=2 if use_cuda else 0,
                        pin_memory=use_cuda)
        probs = []
        for (xb,) in dl:
            xb = xb.to(self.device, non_blocking=True)
            probs.append(torch.softmax(self.model(xb), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)


# ---------------------------------------------------------------------------
#  Pipeline helpers (same as main training)
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


def augment_balance(X, y, seed):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    X_pos, X_neg = X[y == 1], X[y == 0]
    X_pos_g, _ = GaussianNoise(sigma=AUG_SIGMA).transform(X_pos.copy())
    X_pos_t, _ = RandomTimeShift(max_shift=AUG_MAX_SHIFT).transform(X_pos.copy())
    X_aug = np.concatenate([X_neg, X_pos, X_pos_g, X_pos_t])
    y_aug = np.concatenate([
        np.zeros(len(X_neg), dtype=y.dtype),
        np.ones(len(X_pos) * 3, dtype=y.dtype),
    ])
    X_bal, y_bal = MajorityUndersampling(ratio=1.0, seed=seed).transform(X_aug, y_aug)
    np.random.set_state(rng_state)
    idx = np.random.default_rng(seed).permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pooling ablation: avg vs max vs avg+max")
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    run_dir = Path("experimental/results") / f"pooling_ablation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")],
    )

    # --- Report compute device ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  Compute device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Load and prepare data ---
    logger.info("Loading data from %s", args.data_dir)
    X, y, pids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, pids = quality_filter(X, y, pids)
    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, pids, test_size=0.20, seed=SEED,
    )
    X_dev = preprocess(X_dev)
    X_test = preprocess(X_test)
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)

    logger.info("Dev: %d  Test: %d  Folds: %d", len(y_dev), len(y_test), args.n_folds)

    # --- Run ablation ---
    pool_modes = ["max"]
    all_results = {}

    for mode in pool_modes:
        print(f"\n{'='*60}")
        print(f"  Pooling: {mode}")
        print(f"{'='*60}")

        fold_aurocs = []
        for fi, (tr_idx, val_idx) in enumerate(folds):
            X_tr, y_tr = augment_balance(X_dev[tr_idx], y_dev[tr_idx], seed=SEED + fi)
            X_val, y_val = X_dev[val_idx], y_dev[val_idx]

            print(f"\n  Fold {fi+1}/{args.n_folds} — train={len(y_tr)} val={len(y_val)}")

            model = PoolingAblationModel(
                pool_mode=mode,
                stage_filters=BEST_PARAMS["stage_filters"],
                kernels=BEST_PARAMS["kernels"],
                dropout=BEST_PARAMS["dropout"],
                n_heads=BEST_PARAMS["n_heads"],
                lr=BEST_PARAMS["lr"],
                weight_decay=BEST_PARAMS["weight_decay"],
                batch_size=BEST_PARAMS["batch_size"],
                epochs=args.epochs,
            )
            model.fit(X_tr, y_tr, X_val, y_val)

            probs = model.predict_proba(X_val)
            auc = float(roc_auc_score(y_val, probs))
            fold_aurocs.append(auc)
            print(f"  → Fold {fi+1} AUROC = {auc:.4f}")

        arr = np.array(fold_aurocs)
        all_results[mode] = {
            "fold_aurocs": [float(v) for v in fold_aurocs],
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        print(f"\n  {mode} — CV AUROC: {arr.mean():.4f} ± {arr.std():.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  POOLING ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Mode':<10} {'CV AUROC':>12}  {'Per-fold':}")
    print(f"  {'-'*55}")
    for mode in pool_modes:
        r = all_results[mode]
        folds_str = "  ".join(f"{v:.4f}" for v in r["fold_aurocs"])
        print(f"  {mode:<10} {r['mean']:.4f} ± {r['std']:.4f}  [{folds_str}]")

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
