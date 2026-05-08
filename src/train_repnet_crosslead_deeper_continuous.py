"""RepNet CrossLead Deeper — Optuna config + CONTINUOUS positive augmentation.

Same architecture and optimizer as `train_repnet_crosslead_deeper_optimized.py`:
  stage_filters=(48,96,192)  kernels=(7,5,3)  lr=2.465e-3  dropout=0.0546
  weight_decay=1.67e-4

Augmentation pipeline (the only thing changed from the optimized baseline):

  Optimized baseline pre-builds 3 frozen variants per positive (orig, orig+noise,
  orig+shift) before each fold's training. The model sees the same noise
  realization and shift offset every epoch — effective synthetic data is capped
  at 3× per positive for the entire run.

  This script applies a fresh GaussianNoise(σ=AUG_SIGMA) AND a fresh
  RandomTimeShift(±AUG_MAX_SHIFT) on every __getitem__ call. Repeated access to
  the same positive index returns a different realization, so effective
  diversity grows linearly with epochs. Negatives are returned unmodified
  (positives only — preserves the optimized baseline's class-conditional aug).

  Class balance is preserved by replicating positive indices 3× per epoch and
  undersampling negatives to 1:1, so per-epoch sample counts and gradient steps
  match the baseline (apples-to-apples comparison vs the frozen variant).

Usage:
    python -m src.train_repnet_crosslead_deeper_continuous
    python -m src.train_repnet_crosslead_deeper_continuous --n-folds 5 --epochs 60
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_crosslead_deeper import (
    RepNetCrossLeadDeeper,
    RepNetCrossLeadDeeperModel,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

# Optuna-optimized configuration — combines best from completed studies.
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

# Augmentation strength (carried over from optimized study; may be re-tunable
# under continuous resampling — fresh-each-epoch noise is a stronger regularizer
# than frozen noise, so optimal sigma may shift down. Worth a follow-up sweep.)
AUG_SIGMA     = 0.060
AUG_MAX_SHIFT = 276


class ContinuousAugmentTrainDataset(Dataset):
    """Class-balanced training set with on-the-fly positive augmentation.

    Construction-time bookkeeping:
      - Replicate positive sample indices `pos_replicas` times.
      - Undersample negative indices to `neg_ratio * len(replicated_positives)`.
      - Concatenate + shuffle the index list once (fixes per-epoch sample count).

    Per-`__getitem__` behaviour:
      - Negatives are returned unmodified.
      - Positives receive a fresh combination of GaussianNoise(σ=aug_sigma) and
        RandomTimeShift(±aug_max_shift). Each draw is independent — repeated
        access to the same positive index returns a different realization.

    Each DataLoader worker seeds its own RNG so augmentations stay reproducible
    given (seed, worker_id) and don't collide across workers.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        pos_replicas: int = 3,
        neg_ratio: float = 1.0,
        aug_sigma: float = 0.060,
        aug_max_shift: int = 276,
        seed: int = 42,
    ):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.aug_sigma = float(aug_sigma)
        self.aug_max_shift = int(aug_max_shift)
        self._base_seed = int(seed)
        self._rng = None

        rng = np.random.default_rng(seed)
        pos_idx = np.where(self.y == 1)[0]
        neg_idx = np.where(self.y == 0)[0]

        pos_indices = np.tile(pos_idx, int(pos_replicas))
        target_neg = int(round(float(neg_ratio) * len(pos_indices)))
        if 0 < target_neg < len(neg_idx):
            neg_indices = rng.choice(neg_idx, size=target_neg, replace=False)
        else:
            neg_indices = neg_idx

        indices = np.concatenate([pos_indices, neg_indices])
        rng.shuffle(indices)
        self.indices = indices

    def _rng_(self) -> np.random.Generator:
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            if wi is not None:
                # `torch.initial_seed()` is refreshed per-worker-per-epoch by
                # PyTorch's default worker_init logic, so mixing it in keeps
                # augmentations differing across epochs even when
                # persistent_workers=False (CPU-only training case).
                seed = (self._base_seed + 7919 * (wi.id + 1) + int(torch.initial_seed())) & 0xFFFFFFFF
            else:
                seed = self._base_seed
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        x = self.X[i]
        y = int(self.y[i])

        if y != 1:
            return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)

        rng = self._rng_()
        T = x.shape[1]

        shift = int(rng.integers(-self.aug_max_shift, self.aug_max_shift + 1))
        if shift > 0:
            x_shifted = np.zeros_like(x)
            x_shifted[:, shift:] = x[:, : T - shift]
        elif shift < 0:
            x_shifted = np.zeros_like(x)
            x_shifted[:, : T + shift] = x[:, -shift:]
        else:
            x_shifted = x.copy()

        noise = rng.normal(0.0, self.aug_sigma, x_shifted.shape).astype(np.float32)
        x_aug = x_shifted + noise
        return torch.from_numpy(x_aug), torch.tensor(y, dtype=torch.long)


class _ContinuousAugDeeperModel(RepNetCrossLeadDeeperModel):
    """RepNetCrossLeadDeeperModel with continuous-augmentation training.

    - Builds a `ContinuousAugmentTrainDataset` so every positive sample is freshly
      augmented (Gaussian noise + random time shift) on every epoch instead of
      being drawn from a small set of pre-computed variants.
    - Disables early stopping: trains the full `epochs` count, then restores
      best-val-AUROC weights at the end.
    """

    def __init__(
        self,
        *args,
        aug_sigma: float = 0.060,
        aug_max_shift: int = 276,
        pos_replicas: int = 3,
        neg_ratio: float = 1.0,
        fold_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aug_sigma = float(aug_sigma)
        self.aug_max_shift = int(aug_max_shift)
        self.pos_replicas = int(pos_replicas)
        self.neg_ratio = float(neg_ratio)
        self.fold_seed = int(fold_seed)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = RepNetCrossLeadDeeper(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-7,
            weight_decay=self.weight_decay,
        )

        # Synthesize a y array matching the dataset's actual class distribution
        # (only used by `weighted` / `focal` losses; a no-op for cross_entropy).
        n_pos_eff = int((y_train == 1).sum()) * self.pos_replicas
        n_neg_eff = int(round(self.neg_ratio * n_pos_eff))
        criterion = self._build_criterion(
            np.concatenate([
                np.ones(n_pos_eff, dtype=np.int64),
                np.zeros(n_neg_eff, dtype=np.int64),
            ])
        )

        train_ds = ContinuousAugmentTrainDataset(
            X_train, y_train,
            pos_replicas=self.pos_replicas,
            neg_ratio=self.neg_ratio,
            aug_sigma=self.aug_sigma,
            aug_max_shift=self.aug_max_shift,
            seed=self.fold_seed,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size, shuffle=True, num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=torch.cuda.is_available(),
        )
        Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.history = {"train_loss": [], "val_auroc": []}
        best_val_auc, best_state = 0.0, None

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
            val_auc = self._score_device(Xv, y_val)
            self.history["train_loss"].append(avg_loss)
            self.history["val_auroc"].append(val_auc)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                marker = " *"

            print(f"  Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"val_AUROC={val_auc:.4f}{marker}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)


def quality_filter(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
    flat_std_thresh: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    flat_mask = (X.std(axis=2) < flat_std_thresh).any(axis=1)
    n_flat    = int(flat_mask.sum())
    keep      = ~flat_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]

    try:
        nan_mask = np.isnan(patient_ids.astype(float))
    except (ValueError, TypeError):
        nan_mask = np.array([str(p).strip() in ("", "nan", "None") for p in patient_ids])
    n_missing = int(nan_mask.sum())
    keep      = ~nan_mask
    X, y, patient_ids = X[keep], y[keep], patient_ids[keep]

    return X, y, patient_ids, {"flat_lead_dropped": n_flat, "missing_id_dropped": n_missing}


def preprocess(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X.copy())
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def run_cv(X_dev, y_dev, folds, epochs):
    aurocs, auprcs = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr   = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]

        n_pos_tr = int((y_tr == 1).sum())
        n_neg_tr = int((y_tr == 0).sum())
        epoch_size = 2 * 3 * n_pos_tr  # pos*3 + neg undersampled to match
        logger.info(
            "  Fold %d/%d — train=%d (pos=%d neg=%d → epoch=%d, 1:1, continuous aug)  "
            "val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr), n_pos_tr, n_neg_tr, epoch_size,
            len(y_val),
            int((y_val == 1).sum()), int((y_val == 0).sum()),
        )

        model = _ContinuousAugDeeperModel(
            **PARAMS, epochs=epochs,
            aug_sigma=AUG_SIGMA, aug_max_shift=AUG_MAX_SHIFT,
            fold_seed=SEED + fold_idx,
        )
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

    n_pos_tr = int((y_tr == 1).sum())
    logger.info(
        "Final training: %d train (pos=%d → epoch=%d, 1:1, continuous aug) + %d early-stop (clean)",
        len(y_tr), n_pos_tr, 2 * 3 * n_pos_tr, len(y_es),
    )
    model = _ContinuousAugDeeperModel(
        **PARAMS, epochs=epochs,
        aug_sigma=AUG_SIGMA, aug_max_shift=AUG_MAX_SHIFT,
        fold_seed=seed,
    )
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
        "brier": brier,
        "n_test": int(len(y_test)),
        "operating_points": ops,
    }
    return "\n".join(lines), metrics


def render_architecture(net_params, run_dir):
    model = RepNetCrossLeadDeeper(**net_params).eval()
    dummy = torch.randn(1, 12, 2500)
    try:
        from torchinfo import summary as ti_summary
        info = ti_summary(model, input_data=dummy, depth=4, verbose=0,
                          col_names=("input_size", "output_size", "num_params"))
        text = str(info)
    except ImportError:
        n_params = sum(p.numel() for p in model.parameters())
        text = f"{model}\n\nTotal params: {n_params:,}"

    print("\n" + "=" * 60)
    print("  RepNet CrossLead Deeper — Optuna config + CONTINUOUS aug")
    print("=" * 60)
    print(text)
    (run_dir / "architecture.txt").write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(
        description="RepNet CrossLead Deeper (3-stage) — continuous on-the-fly augmentation"
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds",  type=int, default=3)
    parser.add_argument("--epochs",   type=int, default=50)
    args = parser.parse_args()

    run_dir = Path("cv_results") / f"repnet_crosslead_deeper_continuous_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")],
    )
    logger.info("Output: %s", run_dir)
    logger.info("PARAMS: %s", PARAMS)
    logger.info(
        "Augmentation (continuous, positives only): sigma=%.4f  max_shift=%d  pos_replicas=3  neg_ratio=1.0",
        AUG_SIGMA, AUG_MAX_SHIFT,
    )

    net_params = {k: PARAMS[k] for k in
                  ("stage_filters", "kernels", "dropout", "n_heads")}
    render_architecture(net_params, run_dir)

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d  (dropped %d flat-lead, %d missing-ID)",
                len(y), int(y.sum()), int((y == 0).sum()),
                qc["flat_lead_dropped"], qc["missing_id_dropped"])

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=0.20, seed=SEED,
    )
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(),
                len(y_test), 100 * y_test.mean())

    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":          args.data_dir,
            "n_folds":           args.n_folds,
            "epochs":            args.epochs,
            "seed":              SEED,
            "params":            {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "augmentation_mode": "continuous_positives_only",
            "aug_sigma":         AUG_SIGMA,
            "aug_max_shift":     AUG_MAX_SHIFT,
            "pos_replicas":      3,
            "neg_ratio":         1.0,
            "qc":                qc,
            "n_dev":             int(len(y_dev)),
            "n_test":            int(len(y_test)),
            "dev_pos_rate":      float(y_dev.mean()),
            "test_pos_rate":     float(y_test.mean()),
        }, f, indent=2)

    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)

    print(f"\n{'#'*60}\n  RepNet CrossLead Deeper (CONTINUOUS AUG) — {args.n_folds}-fold patient-grouped CV\n{'#'*60}")
    cv_aurocs, cv_auprcs = run_cv(X_dev, y_dev, folds, epochs=args.epochs)

    print("\n  Retraining on full dev set ...")
    final_model = train_final(X_dev, y_dev, g_dev, epochs=args.epochs, seed=SEED)
    probs_test = final_model.predict_proba(X_test)
    torch.save(final_model.model.state_dict(), run_dir / "model_repnet_crosslead_deeper_continuous.pt")

    test_text, test_metrics = format_test_report(y_test, probs_test, g_test)

    cv_arr_auroc = np.array(cv_aurocs)
    cv_arr_auprc = np.array(cv_auprcs)
    summary = "\n".join([
        f"\n{'='*60}",
        f"  RepNet CrossLead Deeper (Optuna config + continuous aug, 3-stage)",
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
