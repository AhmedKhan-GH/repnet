"""Best filter model with CrossLeadAttention replaced by LeadMixConv1x1.

Hypothesis from the per-class attention diagnostic: Stage 1 attention scores
converge to uniform (~0.083 = 1/12), with the residual+gate doing the actual
work. A learned linear cross-lead mix achieves the same lead-mixing capacity
without the attention computation.

Architecture: same (48, 96, 192) filter-search winner config, but the chosen
stages have CrossLeadAttention replaced by a residual Linear(12, 12) mix
across the lead dimension.

Pipeline matches train_repnet_crosslead_deeper_optimized.py (80/20 patient-grouped
split, 3-fold patient-grouped CV, augmentation on, undersample to 1:1, cross_entropy,
early stopping at patience=10).

Usage:
    # Replace only Stage 1 (default — directly tests the diagnostic finding)
    python -m src.train_repnet_crosslead_deeper_lead_mix

    # Replace all stages
    python -m src.train_repnet_crosslead_deeper_lead_mix --replace-stages 1 2 3

    # Replace nothing (smoke test — should match the best filter model results)
    python -m src.train_repnet_crosslead_deeper_lead_mix --replace-stages
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
from src.models.repnet_crosslead_deeper import (
    RepNetCrossLeadDeeper,
    RepNetCrossLeadDeeperModel,
)
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED      = 42
TEST_SIZE = 0.20

PARAMS = dict(
    stage_filters  = (48, 96, 192),
    kernels        = (7, 5, 3),
    n_heads        = 4,
    attn_stages    = (True, True, True),     # build with attention; we'll swap selectively
    lr             = 2.465e-3,
    dropout        = 0.0546,
    weight_decay   = 1.67e-4,
    batch_size     = 64,
    loss_fn        = "cross_entropy",
)

AUG_SIGMA     = 0.060
AUG_MAX_SHIFT = 276


# ------------------------------------------------------------------ #
# LeadMixConv1x1 — residual learned linear mix across the lead axis. #
# ------------------------------------------------------------------ #

class LeadMixConv1x1(nn.Module):
    """Residual Linear(L -> L) over the lead dimension.

    Same I/O shape as CrossLeadAttention so it can replace it inside the
    stage's ModuleDict without changing anything else.
    """
    def __init__(self, n_leads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.mix  = nn.Linear(n_leads, n_leads)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, T)
        h = x.permute(0, 2, 3, 1)        # (B, C, T, L)
        h = self.mix(h)                  # Linear over last dim
        h = self.drop(h)
        h = h.permute(0, 3, 1, 2)        # (B, L, C, T)
        return x + h                     # residual


class _LeadMixDeeperModel(RepNetCrossLeadDeeperModel):
    """Same as RepNetCrossLeadDeeperModel but swaps attention -> LeadMixConv1x1
    at the requested stages (after model construction, before training).

    Early-stop patience=10 retained from parent class.
    """
    def __init__(self, replace_stages=(1,), **kwargs):
        super().__init__(**kwargs)
        # 1-indexed list of stages whose attention to replace
        self.replace_stages = tuple(int(s) for s in replace_stages)

    def fit(self, X_train, y_train, X_val, y_val):
        # Build full model with attention everywhere
        self.model = RepNetCrossLeadDeeper(**self.net_params).to(self.device)

        # Replace attention with LeadMixConv1x1 at requested stages.
        for stage_idx in self.replace_stages:
            i = stage_idx - 1  # 1-indexed -> 0-indexed
            if i < 0 or i >= len(self.model.stages):
                logger.warning("replace_stages index %d out of range; skipping", stage_idx)
                continue
            if "attn" not in self.model.stages[i]:
                logger.warning("stage %d has no attn module to replace; skipping", stage_idx)
                continue
            self.model.stages[i]["attn"] = LeadMixConv1x1(
                n_leads=12,
                dropout=self.net_params.get("dropout", 0.0),
            ).to(self.device)
            logger.info("Replaced stage %d CrossLeadAttn with LeadMixConv1x1", stage_idx)

        # Now reuse the parent's training loop logic — but it builds a new model
        # at the start, so we need to re-implement it here without the rebuild.
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
                n_batches += 1

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
# Pipeline (matches train_repnet_crosslead_deeper_optimized.py)     #
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


def run_cv(X_dev, y_dev, folds, epochs, replace_stages):
    aurocs, auprcs = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr   = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]
        X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=SEED + fold_idx)

        logger.info("  Fold %d/%d - train=%d val=%d (pos=%d neg=%d)",
                    fold_idx + 1, len(folds), len(y_tr), len(y_val),
                    int((y_val == 1).sum()), int((y_val == 0).sum()))

        model = _LeadMixDeeperModel(replace_stages=replace_stages, **PARAMS, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        probs = model.predict_proba(X_val)
        aurocs.append(float(roc_auc_score(y_val, probs)))
        auprcs.append(float(average_precision_score(y_val, probs)))
        print(f"  -> Fold {fold_idx+1} AUROC={aurocs[-1]:.4f}  AUPRC={auprcs[-1]:.4f}")

    return aurocs, auprcs


def train_final(X_dev, y_dev, g_dev, epochs, replace_stages, seed):
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))

    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]
    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=seed)
    logger.info("Final training: %d train + %d early-stop", len(y_tr), len(y_es))

    model = _LeadMixDeeperModel(replace_stages=replace_stages, **PARAMS, epochs=epochs)
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
        description="Filter model with LeadMixConv1x1 replacing CrossLeadAttention at chosen stages"
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds",  type=int, default=3)
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--replace-stages", type=int, nargs="*", default=[1],
                        help="1-indexed stages to replace attn -> LeadMixConv1x1. "
                             "Default: [1] (only Stage 1). Pass nothing to keep all attention.")
    args = parser.parse_args()

    rs_str = "_".join(str(s) for s in args.replace_stages) if args.replace_stages else "none"
    run_dir = Path("cv_results") / f"deeper_lead_mix_s{rs_str}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")])
    logger.info("Output: %s", run_dir)
    logger.info("Replace stages: %s -> LeadMixConv1x1", args.replace_stages or "(none)")
    logger.info("PARAMS: %s", {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()})

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d  (%.1f%% pos)",
                len(y), int(y.sum()), int((y == 0).sum()), 100*y.mean())

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=TEST_SIZE, seed=SEED)
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)  [80/20 split]",
                len(y_dev), 100 * y_dev.mean(),
                len(y_test), 100 * y_test.mean())

    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":        args.data_dir,
            "test_size":       TEST_SIZE,
            "n_folds":         args.n_folds,
            "epochs":          args.epochs,
            "seed":            SEED,
            "params":          {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "aug_sigma":       AUG_SIGMA,
            "aug_max_shift":   AUG_MAX_SHIFT,
            "replace_stages":  list(args.replace_stages),
            "qc":              qc,
            "n_dev":           int(len(y_dev)),
            "n_test":          int(len(y_test)),
            "dev_pos_rate":    float(y_dev.mean()),
            "test_pos_rate":   float(y_test.mean()),
        }, f, indent=2)

    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)
    print(f"\n{'#'*60}\n  Filter arch + LeadMixConv1x1 at stages {args.replace_stages or '(none)'}\n  {args.n_folds}-fold CV (80/20 split)\n{'#'*60}")
    cv_aurocs, cv_auprcs = run_cv(X_dev, y_dev, folds, epochs=args.epochs,
                                   replace_stages=args.replace_stages)

    print("\n  Retraining on full dev set ...")
    final_model = train_final(X_dev, y_dev, g_dev, epochs=args.epochs,
                               replace_stages=args.replace_stages, seed=SEED)
    probs_test = final_model.predict_proba(X_test)
    torch.save(final_model.model.state_dict(), run_dir / "model.pt")

    test_text, test_metrics = format_test_report(y_test, probs_test, g_test)
    cv_arr_auroc = np.array(cv_aurocs)
    cv_arr_auprc = np.array(cv_auprcs)
    summary = "\n".join([
        f"\n{'='*60}",
        f"  Filter arch + LeadMixConv1x1 (replaced stages: {args.replace_stages or 'none'})",
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
