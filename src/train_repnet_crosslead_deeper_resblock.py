"""Best filter model with channel-independent PerLeadConvBlock replaced by a
cross-lead ResBlock that treats the 12 leads as additional channels.

Architecture stays at filter-search winner config (48, 96, 192) / kernels (7, 5, 3),
but each stage's conv block is now:
  Input:  (B, 12, C_in, T)
  Reshape (B, 12*C_in, T)
  Conv1d (12*C_in -> 12*C_out, k) + BN + ReLU
  Conv1d (12*C_out -> 12*C_out, k) + BN + skip + ReLU + MaxPool(2)
  Reshape (B, 12, C_out, T/2)

Effect: lead identity is destroyed inside the conv block — every lead's
features get mixed with every other lead's, with independent weights for each
(lead, channel) pair. Cross-lead mixing now happens in BOTH the conv and the
attention layer (was: only attention).

Parameter cost: ~140M parameters total (vs. 965K for the per-lead variant).
This is a huge overparameterization for ~370 effective training samples — the
result is mostly diagnostic (will overfit massively, train AUROC near 1.0,
test AUROC may collapse).

Usage:
    python -m src.train_repnet_crosslead_deeper_resblock
    python -m src.train_repnet_crosslead_deeper_resblock --epochs 30 --n-folds 3
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_crosslead import CrossLeadAttention
from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeperModel
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED = 42

PARAMS = dict(
    stage_filters  = (48, 96, 192),
    kernels        = (7, 5, 3),
    n_heads        = 4,
    lr             = 2.465e-3,
    dropout        = 0.0546,
    weight_decay   = 1.67e-4,
    batch_size     = 64,
    loss_fn        = "cross_entropy",
)
AUG_SIGMA     = 0.060
AUG_MAX_SHIFT = 276


# ------------------------------------------------------------------ #
# Cross-lead ResBlock (replaces PerLeadConvBlock).                    #
# ------------------------------------------------------------------ #

class ResBlockMixed(nn.Module):
    """Standard 1D ResBlock that treats the 12 leads as part of the channel dim.

    Parameter count scales as (n_leads * C)^2 * k vs. C^2 * k for the per-lead
    variant — i.e. ~144x more parameters per layer for 12 leads.

    Same I/O shape as PerLeadConvBlock so it can drop in.
    """

    def __init__(self, n_leads: int, in_ch: int, out_ch: int,
                 kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        p = kernel_size // 2
        self.n_leads = n_leads
        self.out_ch = out_ch
        in_full  = n_leads * in_ch
        out_full = n_leads * out_ch

        self.conv1 = nn.Conv1d(in_full,  out_full, kernel_size, padding=p)
        self.bn1   = nn.BatchNorm1d(out_full)
        self.conv2 = nn.Conv1d(out_full, out_full, kernel_size, padding=p)
        self.bn2   = nn.BatchNorm1d(out_full)
        self.act   = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout(dropout)

        if in_full != out_full:
            self.skip = nn.Sequential(
                nn.Conv1d(in_full, out_full, kernel_size=1),
                nn.BatchNorm1d(out_full),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C_in, T)
        B, L, C, T = x.shape
        h = x.reshape(B, L * C, T)
        residual = self.skip(h)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.bn2(self.conv2(h))
        h = self.act(h + residual)
        h = self.drop(self.pool(h))
        T_out = h.shape[-1]
        return h.reshape(B, L, self.out_ch, T_out)


class RepNetCrossLeadDeeperResBlock(nn.Module):
    """Same shape as RepNetCrossLeadDeeper but with ResBlockMixed in place of PerLeadConvBlock."""

    def __init__(
        self,
        n_leads:        int = 12,
        stage_filters:  tuple[int, ...] = (48, 96, 192),
        kernels:        tuple[int, ...] = (7, 5, 3),
        dropout:        float = 0.0546,
        n_heads:        int = 4,
        attn_stages:    tuple[bool, ...] | None = None,
        attn_tokens:    tuple[int, ...] | int = 1,
        n_classes:      int = 2,
        **kwargs,
    ):
        super().__init__()
        if attn_stages is None:
            attn_stages = tuple([True] * len(stage_filters))
        if isinstance(attn_tokens, int):
            attn_tokens = tuple([attn_tokens] * len(stage_filters))

        in_c = 1
        stages = []
        for f, k, use_attn, n_tok in zip(stage_filters, kernels, attn_stages, attn_tokens):
            stage = nn.ModuleDict({
                "conv": ResBlockMixed(n_leads, in_c, f, k, dropout),
            })
            if use_attn:
                stage["attn"] = CrossLeadAttention(f, n_heads, dropout, n_tokens=int(n_tok))
            stages.append(stage)
            in_c = f
        self.stages = nn.ModuleList(stages)

        f_last = stage_filters[-1]
        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f_last, f_last, kernel_size=1),
            nn.BatchNorm1d(f_last),
            nn.ReLU(inplace=True),
        )
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc        = nn.Linear(f_last, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)                # (B, 12, 1, T)
        for stage in self.stages:
            x = stage["conv"](x)
            if "attn" in stage:
                x = stage["attn"](x)
        B, L, F, T_out = x.shape
        x = x.reshape(B, L * F, T_out)
        x = self.fuse(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


# ------------------------------------------------------------------ #
# Wrapper that builds RepNetCrossLeadDeeperResBlock instead of the    #
# per-lead variant; everything else (training loop, early stop=10) is #
# inherited from RepNetCrossLeadDeeperModel.                          #
# ------------------------------------------------------------------ #

class _ResBlockDeeperModel(RepNetCrossLeadDeeperModel):
    """Subclass that swaps the network class. Reuses the parent's training loop."""

    def fit(self, X_train, y_train, X_val, y_val):
        # Build the resblock variant; signature matches the per-lead variant.
        self.model = RepNetCrossLeadDeeperResBlock(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        train_dl = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True, num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=torch.cuda.is_available(),
        )
        Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.history = {"train_loss": [], "val_auroc": []}
        best_val_auc, best_state = 0.0, None
        patience_counter, patience = 0, 10

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in train_dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            val_auc  = self._score_device(Xv, y_val)
            self.history["train_loss"].append(avg_loss)
            self.history["val_auroc"].append(val_auc)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            print(f"  Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"val_AUROC={val_auc:.4f}{marker}")

            if patience_counter >= patience:
                print("  Early stop")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)


# ------------------------------------------------------------------ #
# Pipeline (matches train_repnet_crosslead_deeper_optimized.py)       #
# ------------------------------------------------------------------ #

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


def preprocess(X):
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X.copy())
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


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


def run_cv(X_dev, y_dev, folds, epochs):
    aurocs, auprcs = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr   = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]
        X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=SEED + fold_idx)
        logger.info("  Fold %d/%d - train=%d val=%d (pos=%d neg=%d)",
                    fold_idx + 1, len(folds), len(y_tr), len(y_val),
                    int((y_val == 1).sum()), int((y_val == 0).sum()))
        model = _ResBlockDeeperModel(**PARAMS, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        probs = model.predict_proba(X_val)
        aurocs.append(float(roc_auc_score(y_val, probs)))
        auprcs.append(float(average_precision_score(y_val, probs)))
        print(f"  -> Fold {fold_idx+1} AUROC={aurocs[-1]:.4f}  AUPRC={auprcs[-1]:.4f}")
    return aurocs, auprcs


def train_final(X_dev, y_dev, g_dev, epochs, seed):
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]
    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=seed)
    logger.info("Final training: %d train + %d early-stop", len(y_tr), len(y_es))
    model = _ResBlockDeeperModel(**PARAMS, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    return model


def youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def sensitivity_threshold(y_true, probs, target_sens=0.90):
    fpr, tpr, thr = roc_curve(y_true, probs)
    valid = tpr >= target_sens
    return float(thr[valid][-1]) if valid.any() else 0.0


def operating_point_metrics(y_true, probs, tau):
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    f1   = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return {"threshold": float(tau), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv), "f1": float(f1)}


def bootstrap_ci(y_true, probs, groups, metric_fn, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    unique_pats = np.unique(groups)
    point = float(metric_fn(y_true, probs))
    samples = []
    for _ in range(n_boot):
        pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
        idx = np.concatenate([np.where(groups == p)[0] for p in pats])
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(float(metric_fn(y_true[idx], probs[idx])))
        except ValueError:
            pass
    samples = np.asarray(samples)
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def format_test_report(y_test, probs, g_test):
    auroc_p, auroc_lo, auroc_hi = bootstrap_ci(y_test, probs, g_test, roc_auc_score)
    auprc_p, auprc_lo, auprc_hi = bootstrap_ci(y_test, probs, g_test, average_precision_score)
    brier = float(brier_score_loss(y_test, probs))
    tau_youden = youden_threshold(y_test, probs)
    tau_sens   = sensitivity_threshold(y_test, probs, target_sens=0.90)
    ops = {
        "tau=0.50":       operating_point_metrics(y_test, probs, 0.5),
        "tau=Youden":     operating_point_metrics(y_test, probs, tau_youden),
        "tau=sens>=0.90": operating_point_metrics(y_test, probs, tau_sens),
    }
    lines = [
        f"  AUROC  : {auroc_p:.4f}  [{auroc_lo:.4f}, {auroc_hi:.4f}]   (95% bootstrap, patient-level)",
        f"  AUPRC  : {auprc_p:.4f}  [{auprc_lo:.4f}, {auprc_hi:.4f}]",
        f"  Brier  : {brier:.4f}",
        f"  Test   : N={len(y_test)}  pos={int(y_test.sum())}  neg={int((y_test==0).sum())}",
        "",
        f"  {'Operating point':<18} {'tau':>7}  {'sens':>6}  {'spec':>6}  {'PPV':>5}  {'NPV':>5}  {'F1':>5}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}",
    ]
    for name, m in ops.items():
        lines.append(
            f"  {name:<18} {m['threshold']:>7.3f}  {m['sensitivity']:>6.3f}  "
            f"{m['specificity']:>6.3f}  {m['ppv']:>5.3f}  {m['npv']:>5.3f}  "
            f"{m['f1']:>5.3f}  {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}"
        )
    metrics = {
        "auroc": {"point": auroc_p, "ci_low": auroc_lo, "ci_high": auroc_hi},
        "auprc": {"point": auprc_p, "ci_low": auprc_lo, "ci_high": auprc_hi},
        "brier": brier, "n_test": int(len(y_test)),
        "operating_points": ops,
    }
    return "\n".join(lines), metrics


def main():
    parser = argparse.ArgumentParser(
        description="Filter arch with PerLeadConvBlock replaced by cross-lead ResBlock"
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds",  type=int, default=3)
    parser.add_argument("--epochs",   type=int, default=50)
    args = parser.parse_args()

    run_dir = Path("cv_results") / f"deeper_resblock_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")])
    logger.info("Output: %s", run_dir)
    logger.info("Variant: PerLeadConvBlock -> ResBlockMixed (cross-lead)")
    logger.info("PARAMS: %s", {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()})

    # Print parameter count up-front so the user sees the cost.
    test_net = RepNetCrossLeadDeeperResBlock(
        stage_filters=PARAMS["stage_filters"], kernels=PARAMS["kernels"],
        dropout=PARAMS["dropout"], n_heads=PARAMS["n_heads"],
    )
    n_params = sum(p.numel() for p in test_net.parameters())
    logger.info("ResBlock variant total parameters: %s (%.1fM)",
                f"{n_params:,}", n_params / 1e6)
    del test_net

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d  (%.1f%% pos)",
                len(y), int(y.sum()), int((y == 0).sum()), 100*y.mean())

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=0.20, seed=SEED)
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(),
                len(y_test), 100 * y_test.mean())

    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":      args.data_dir,
            "test_size":     0.20,
            "n_folds":       args.n_folds,
            "epochs":        args.epochs,
            "seed":          SEED,
            "params":        {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "aug_sigma":     AUG_SIGMA,
            "aug_max_shift": AUG_MAX_SHIFT,
            "qc":            qc,
            "n_dev":         int(len(y_dev)),
            "n_test":        int(len(y_test)),
            "dev_pos_rate":  float(y_dev.mean()),
            "test_pos_rate": float(y_test.mean()),
            "variant":       "ResBlockMixed (cross-lead)",
            "n_parameters":  int(n_params),
        }, f, indent=2)

    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)
    print(f"\n{'#'*60}\n  ResBlockMixed variant - {args.n_folds}-fold CV (80/20 split)\n{'#'*60}")
    cv_aurocs, cv_auprcs = run_cv(X_dev, y_dev, folds, epochs=args.epochs)

    print("\n  Retraining on full dev set ...")
    final_model = train_final(X_dev, y_dev, g_dev, epochs=args.epochs, seed=SEED)
    probs_test = final_model.predict_proba(X_test)
    torch.save(final_model.model.state_dict(), run_dir / "model.pt")

    test_text, test_metrics = format_test_report(y_test, probs_test, g_test)
    cv_arr_auroc = np.array(cv_aurocs)
    cv_arr_auprc = np.array(cv_auprcs)
    summary = "\n".join([
        f"\n{'='*60}",
        f"  ResBlockMixed variant ({n_params:,} params, {n_params/1e6:.1f}M)",
        f"{'='*60}",
        f"  CV AUROC : {cv_arr_auroc.mean():.4f} +/- {cv_arr_auroc.std():.4f}    "
        f"per-fold: {[f'{v:.3f}' for v in cv_aurocs]}",
        f"  CV AUPRC : {cv_arr_auprc.mean():.4f} +/- {cv_arr_auprc.std():.4f}    "
        f"per-fold: {[f'{v:.3f}' for v in cv_auprcs]}",
        f"\n  Test set evaluation",
        f"  {'-'*40}",
        test_text,
    ])
    print(summary)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    with open(run_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "cv_auroc_mean": float(cv_arr_auroc.mean()),
            "cv_auroc_std":  float(cv_arr_auroc.std()),
            "cv_aurocs":     [float(v) for v in cv_aurocs],
            "cv_auprc_mean": float(cv_arr_auprc.mean()),
            "cv_auprc_std":  float(cv_arr_auprc.std()),
            "cv_auprcs":     [float(v) for v in cv_auprcs],
            "test":          test_metrics,
        }, f, indent=2)

    np.savez(run_dir / "test_predictions.npz",
             y_true=y_test, probs=probs_test, patient_ids=g_test)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
