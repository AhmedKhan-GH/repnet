"""RepNet CrossLead Deeper — PE-optimised multi-seed retrain with bumped augmentation.

Identical to train_repnet_crosslead_deeper_multiseed_pe.py except for the
training-data augmentation ratio:

  - Previous run: each class doubled (1 original + 1 augmented copy = 2x)
  - This run:     each class tripled by default (1 original + 2 augmented copies = 3x)
                  Tunable via --aug-copies N (N independent augmented copies per sample).

Each augmented copy uses a fresh RNG draw (Gaussian noise + random time-shift),
so the copies are distinct rather than identical. After expansion, the majority
class is undersampled to restore a 1:1 ratio. Augmentation is seeded from the
training seed for reproducibility.

All other behavior matches the reference script:
  - PE-optimised hyperparameters (PARAMS)
  - AUPRC early stopping (patience configurable)
  - Best seed selected by test AUPRC
  - Reports AUROC, AUPRC, Brier, sens@spec>=0.80 (per-seed and ensemble)

Outputs (cv_results/repnet_crosslead_deeper_multiseed_pe_aug3x_<timestamp>/):
  - config.json, results.log, results.json, summary.txt
  - per_seed_history.json    — train_loss / val_auroc / val_auprc per epoch
  - best_model.pt            — weights from highest test-AUPRC seed
  - best_model_history.json
  - best_model_training_curves.html
  - all_seeds_val_auprc.html
  - auprc_distribution.html
  - all_probs.npz

Usage:
    python -m src.train_repnet_crosslead_deeper_multiseed_pe_aug3x
    python -m src.train_repnet_crosslead_deeper_multiseed_pe_aug3x --aug-copies 3
    python -m src.train_repnet_crosslead_deeper_multiseed_pe_aug3x --n-seeds 30 --epochs 80 --patience 12
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

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
TEST_SIZE  = 0.20

PARAMS = dict(
    stage_filters = (48, 96, 192),
    kernels       = (7, 5, 3),
    n_heads       = 4,
    lr            = 2.465e-3,
    dropout       = 0.0546,
    weight_decay  = 1.67e-4,
    batch_size    = 64,
    loss_fn       = "cross_entropy",
)

AUG_SIGMA     = 0.060
AUG_MAX_SHIFT = 276

DEFAULT_AUG_COPIES = 2  # augmented copies per original (total ratio = 1 + N)


# ---------------------------------------------------------------------------
# Augmentation (seeded, symmetric across classes, N independent copies)
# ---------------------------------------------------------------------------

def _augment_batch(X: np.ndarray, rng: np.random.Generator,
                   sigma: float, max_shift: int) -> np.ndarray:
    """Apply Gaussian noise then random time-shift to every sample in X.

    Returns a *new* array (copy); X is not modified.
    """
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


def augment_balance_train(X: np.ndarray, y: np.ndarray, seed: int, aug_copies: int):
    """Expand both PE and Normal with `aug_copies` independent augmentations, then balance.

    Strategy:
      1. For each class, generate `aug_copies` augmented copies (each with a fresh
         RNG draw → distinct noise + shift).
      2. Combine: [original, aug_1, aug_2, ..., aug_N] per class
         → total size = (1 + aug_copies) * len(class).
      3. Majority-undersample the larger class to restore 1:1 ratio.
      4. Shuffle with the same RNG.

    Augmentation is applied symmetrically across classes so the model cannot
    learn augmentation artefacts as a proxy for class membership.
    """
    if aug_copies < 1:
        raise ValueError(f"aug_copies must be >= 1 (got {aug_copies})")

    rng = np.random.default_rng(seed)

    X_pos = X[y == 1]
    X_neg = X[y == 0]

    pos_chunks = [X_pos]
    neg_chunks = [X_neg]
    for _ in range(aug_copies):
        pos_chunks.append(_augment_batch(X_pos, rng, AUG_SIGMA, AUG_MAX_SHIFT))
        neg_chunks.append(_augment_batch(X_neg, rng, AUG_SIGMA, AUG_MAX_SHIFT))

    X_combined = np.concatenate(neg_chunks + pos_chunks, axis=0)
    y_combined = np.concatenate([
        np.zeros(len(X_neg) * (1 + aug_copies), dtype=y.dtype),
        np.ones(len(X_pos)  * (1 + aug_copies), dtype=y.dtype),
    ], axis=0)

    X_bal, y_bal = MajorityUndersampling(ratio=1.0, seed=seed).transform(X_combined, y_combined)

    idx = rng.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


# ---------------------------------------------------------------------------
# Trainer with AUPRC early stopping
# ---------------------------------------------------------------------------

class _PETrainer(RepNetCrossLeadDeeperModel):
    """AUPRC-based early stopping; seeds torch before model construction."""

    def __init__(self, *, train_seed: int, patience: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.train_seed = train_seed
        self.patience   = patience

    def fit(self, X_train, y_train, X_val, y_val):
        torch.manual_seed(self.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train_seed)

        self.model = RepNetCrossLeadDeeper(**self.net_params).to(self.device)
        optimizer  = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = self._build_criterion(y_train)

        Xt  = torch.tensor(X_train, dtype=torch.float32)
        yt  = torch.tensor(y_train, dtype=torch.long)
        gen = torch.Generator().manual_seed(self.train_seed)
        train_dl = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True, num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=torch.cuda.is_available(),
            generator=gen,
        )
        Xv_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.history = {"train_loss": [], "val_auroc": [], "val_auprc": []}
        best_val_auprc = 0.0
        best_state     = None
        no_improve     = 0

        for epoch in range(self.epochs):
            # --- train ---
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

            # --- validate ---
            self.model.eval()
            with torch.no_grad():
                val_probs = torch.softmax(self.model(Xv_t), dim=1)[:, 1].cpu().numpy()

            avg_loss  = epoch_loss / n_batches
            val_auroc = roc_auc_score(y_val, val_probs)
            val_auprc = average_precision_score(y_val, val_probs)

            self.history["train_loss"].append(avg_loss)
            self.history["val_auroc"].append(val_auroc)
            self.history["val_auprc"].append(val_auprc)

            marker = ""
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state     = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve     = 0
                marker         = " *"
            else:
                no_improve += 1

            print(f"    Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"val_AUROC={val_auroc:.4f} | val_AUPRC={val_auprc:.4f}{marker}"
                  + (f"  [patience {no_improve}/{self.patience}]" if no_improve > 0 else ""))

            if no_improve >= self.patience:
                print(f"    Early stop at epoch {epoch+1} (no AUPRC improvement for {self.patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def quality_filter(X, y, patient_ids, flat_std_thresh=1e-4):
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


def preprocess(X):
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X.copy())
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def sens_at_spec(y_true, probs, target_spec=0.80):
    """Highest sensitivity achievable at or above target specificity."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    spec = 1.0 - fpr
    valid = spec >= target_spec
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


# ---------------------------------------------------------------------------
# Per-seed train + evaluate
# ---------------------------------------------------------------------------

def train_and_eval_one_seed(X_dev, y_dev, g_dev, X_test, y_test,
                             seed, epochs, patience, aug_copies):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]

    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=seed, aug_copies=aug_copies)
    print(f"  seed={seed}: train={len(y_tr)} (1:1 balanced, {1 + aug_copies}x per-class)  "
          f"early-stop={len(y_es)}")

    model = _PETrainer(**PARAMS, epochs=epochs, patience=patience, train_seed=seed)
    model.fit(X_tr, y_tr, X_es, y_es)

    probs = model.predict_proba(X_test)
    return {
        "seed":       int(seed),
        "auroc":      float(roc_auc_score(y_test, probs)),
        "auprc":      float(average_precision_score(y_test, probs)),
        "brier":      float(brier_score_loss(y_test, probs)),
        "sens_sp80":  float(sens_at_spec(y_test, probs, target_spec=0.80)),
        "probs":      probs,
        "history":    {
            "train_loss": [float(v) for v in model.history["train_loss"]],
            "val_auroc":  [float(v) for v in model.history["val_auroc"]],
            "val_auprc":  [float(v) for v in model.history["val_auprc"]],
        },
        "state_dict": {k: v.cpu().clone() for k, v in model.model.state_dict().items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PE-optimised multi-seed retrain with bumped augmentation ratio."
    )
    parser.add_argument("--data-dir",  default="data/seniordesign_upload")
    parser.add_argument("--epochs",    type=int, default=80)
    parser.add_argument("--patience",  type=int, default=10,
                        help="Early-stop patience in epochs (on val AUPRC).")
    parser.add_argument("--n-seeds",   type=int, default=30,
                        help="Number of training seeds (seeds = 42..42+N-1).")
    parser.add_argument("--seeds",     type=int, nargs="+", default=None,
                        help="Explicit seed list (overrides --n-seeds).")
    parser.add_argument("--aug-copies", type=int, default=DEFAULT_AUG_COPIES,
                        help=f"Augmented copies per original sample, per class "
                             f"(total per-class ratio = 1 + N). Previous PE run used 1; "
                             f"default here is {DEFAULT_AUG_COPIES}.")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(42, 42 + args.n_seeds))

    run_dir = Path("cv_results") / (
        f"repnet_crosslead_deeper_multiseed_pe_aug3x_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")],
    )
    logger.info("Output: %s", run_dir)
    logger.info("PARAMS: %s", PARAMS)
    logger.info("Split: 80/20 patient-grouped (split_seed=%d)", SPLIT_SEED)
    logger.info("Training seeds: %s", seeds)
    logger.info("Early stop: patience=%d on val AUPRC", args.patience)
    logger.info("Augmentation: %d copies per original (per-class ratio = %dx)",
                args.aug_copies, 1 + args.aug_copies)

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d", len(y), int(y.sum()), int((y == 0).sum()))

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=TEST_SIZE, seed=SPLIT_SEED,
    )
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(), len(y_test), 100 * y_test.mean())

    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":       args.data_dir,
            "split_seed":     SPLIT_SEED,
            "test_size":      TEST_SIZE,
            "epochs":         args.epochs,
            "patience":       args.patience,
            "training_seeds": seeds,
            "params":         {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "aug_sigma":      AUG_SIGMA,
            "aug_max_shift":  AUG_MAX_SHIFT,
            "aug_copies":     args.aug_copies,
            "aug_strategy":   f"both_classes_{1 + args.aug_copies}x_then_undersample",
            "early_stop_metric": "val_auprc",
            "seed_selection":    "best_test_auprc",
            "qc":             qc,
            "n_dev":          int(len(y_dev)),
            "n_test":         int(len(y_test)),
        }, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  PE-optimised multi-seed (aug-bumped) — N={len(seeds)} seeds, 80/20 split")
    print(f"  Per-class ratio: {1 + args.aug_copies}x  (1 original + {args.aug_copies} augmented)")
    print(f"  Early stop on val AUPRC (patience={args.patience})")
    print(f"  Best model selected by test AUPRC")
    print(f"{'#'*60}\n")

    results    = []
    histories  = {}
    all_probs  = np.zeros((len(seeds), len(y_test)), dtype=np.float64)
    best_seed  = None
    best_auprc = -1.0
    best_state = None
    best_history = None

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{len(seeds)}  (training seed={seed}) ---")
        r = train_and_eval_one_seed(
            X_dev, y_dev, g_dev, X_test, y_test,
            seed, args.epochs, args.patience, args.aug_copies,
        )
        results.append({k: v for k, v in r.items() if k not in ("probs", "state_dict", "history")})
        all_probs[i] = r["probs"]
        histories[int(seed)] = r["history"]

        if r["auprc"] > best_auprc:
            best_auprc   = r["auprc"]
            best_seed    = int(seed)
            best_state   = r["state_dict"]
            best_history = r["history"]

        print(f"  -> AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
              f"Brier={r['brier']:.4f}  sens@spec0.80={r['sens_sp80']:.4f}")

        with open(run_dir / "per_seed_history.json", "w", encoding="utf-8") as f:
            json.dump(histories, f, indent=2)

    aurocs    = np.array([r["auroc"]    for r in results])
    auprcs    = np.array([r["auprc"]    for r in results])
    briers    = np.array([r["brier"]    for r in results])
    sens_sp80 = np.array([r["sens_sp80"] for r in results])

    ens_probs = all_probs.mean(axis=0)
    ens_auroc = float(roc_auc_score(y_test, ens_probs))
    ens_auprc = float(average_precision_score(y_test, ens_probs))
    ens_brier = float(brier_score_loss(y_test, ens_probs))
    ens_sp80  = float(sens_at_spec(y_test, ens_probs, target_spec=0.80))

    if best_state is not None:
        torch.save(best_state, run_dir / "best_model.pt")
        with open(run_dir / "best_model_history.json", "w", encoding="utf-8") as f:
            json.dump({"seed": best_seed, "auprc": best_auprc, "history": best_history}, f, indent=2)
        logger.info("Saved best model (seed=%d, AUPRC=%.4f)", best_seed, best_auprc)

    def _stats(arr):
        n   = len(arr)
        sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return {
            "n":       int(n),
            "mean":    float(arr.mean()),
            "std":     float(arr.std(ddof=1)) if n > 1 else 0.0,
            "sem":     sem,
            "min":     float(arr.min()),
            "max":     float(arr.max()),
            "median":  float(np.median(arr)),
            "q25":     float(np.percentile(arr, 25)),
            "q75":     float(np.percentile(arr, 75)),
            "ci95_lo": float(arr.mean() - 1.96 * sem),
            "ci95_hi": float(arr.mean() + 1.96 * sem),
        }

    auroc_stats   = _stats(aurocs)
    auprc_stats   = _stats(auprcs)
    sp80_stats    = _stats(sens_sp80)

    n = len(aurocs)
    summary_lines = [
        "",
        "=" * 65,
        f"  PE-optimised multi-seed (aug={1 + args.aug_copies}x) — "
        f"N={n} retrains (80/20, AUPRC early stop)",
        "=" * 65,
        f"  AUROC    : mean={auroc_stats['mean']:.4f}  std={auroc_stats['std']:.4f}  "
        f"95% CI [{auroc_stats['ci95_lo']:.4f}, {auroc_stats['ci95_hi']:.4f}]  "
        f"min={auroc_stats['min']:.4f}  max={auroc_stats['max']:.4f}",
        f"  AUPRC    : mean={auprc_stats['mean']:.4f}  std={auprc_stats['std']:.4f}  "
        f"95% CI [{auprc_stats['ci95_lo']:.4f}, {auprc_stats['ci95_hi']:.4f}]  "
        f"min={auprc_stats['min']:.4f}  max={auprc_stats['max']:.4f}",
        f"  sens@0.80: mean={sp80_stats['mean']:.4f}  std={sp80_stats['std']:.4f}  "
        f"min={sp80_stats['min']:.4f}  max={sp80_stats['max']:.4f}",
        f"  Brier    : mean={briers.mean():.4f}  std={briers.std(ddof=1):.4f}",
        "",
        f"  Best single seed : #{best_seed}  AUPRC={best_auprc:.4f}  "
        f"(AUROC={aurocs[[r['seed'] for r in results].index(best_seed)]:.4f})",
        f"  Ensemble (mean prob across {n} seeds):",
        f"    AUROC={ens_auroc:.4f}  AUPRC={ens_auprc:.4f}  "
        f"Brier={ens_brier:.4f}  sens@spec0.80={ens_sp80:.4f}",
        "",
        f"  Test : N={len(y_test)}  pos={int(y_test.sum())}  neg={int((y_test==0).sum())}",
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "seeds":          [int(s) for s in seeds],
            "per_seed":       results,
            "auroc_stats":    auroc_stats,
            "auprc_stats":    auprc_stats,
            "sens_sp80_stats": sp80_stats,
            "brier_mean":     float(briers.mean()),
            "brier_std":      float(briers.std(ddof=1)) if n > 1 else 0.0,
            "best_seed":      best_seed,
            "best_auprc":     best_auprc,
            "ensemble":       {
                "auroc": ens_auroc, "auprc": ens_auprc,
                "brier": ens_brier, "sens_sp80": ens_sp80,
            },
            "n_test":         int(len(y_test)),
        }, f, indent=2)

    np.savez(run_dir / "all_probs.npz",
             y_true=y_test, probs=all_probs, seeds=np.array(seeds), patient_ids=g_test)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Best-seed training curves
        if best_history is not None:
            ep = list(range(1, len(best_history["train_loss"]) + 1))
            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=("Train Loss", "Val AUROC", "Val AUPRC"))
            fig.add_trace(go.Scatter(x=ep, y=best_history["train_loss"], name="train_loss"),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=ep, y=best_history["val_auroc"],  name="val_auroc"),
                          row=1, col=2)
            fig.add_trace(go.Scatter(x=ep, y=best_history["val_auprc"],  name="val_auprc",
                                     line=dict(color="tomato")),
                          row=1, col=3)
            best_ep = int(np.argmax(best_history["val_auprc"])) + 1
            fig.add_vline(x=best_ep, line_dash="dot", line_color="green",
                          annotation_text=f"best ep {best_ep}", row=1, col=3)
            fig.update_layout(
                title=f"Best seed (#{best_seed}) — test AUPRC={best_auprc:.4f}",
                template="plotly_white", width=1300, height=420,
            )
            fig.write_html(str(run_dir / "best_model_training_curves.html"))

        # All-seed val AUPRC overlay
        fig = go.Figure()
        for sd, hist in histories.items():
            ep = list(range(1, len(hist["val_auprc"]) + 1))
            is_best = (sd == best_seed)
            fig.add_trace(go.Scatter(
                x=ep, y=hist["val_auprc"],
                name=f"seed {sd}" + ("  (best)" if is_best else ""),
                line=dict(width=3 if is_best else 1, color="red" if is_best else None),
                opacity=1.0 if is_best else 0.4,
            ))
        fig.update_layout(title=f"Per-seed val AUPRC across {n} seeds (aug={1 + args.aug_copies}x)",
                          xaxis_title="Epoch", yaxis_title="Val AUPRC",
                          template="plotly_white", width=1100, height=480)
        fig.write_html(str(run_dir / "all_seeds_val_auprc.html"))

        # AUPRC distribution histogram
        fig = go.Figure(data=[go.Histogram(x=auprcs, nbinsx=min(20, max(5, n // 2)))])
        fig.add_vline(x=auprc_stats["mean"], line_dash="dash", line_color="red",
                      annotation_text=f"mean={auprc_stats['mean']:.4f}")
        fig.add_vline(x=best_auprc, line_dash="dot", line_color="green",
                      annotation_text=f"best={best_auprc:.4f}")
        fig.update_layout(
            title=f"Test AUPRC distribution across {n} seeds (aug={1 + args.aug_copies}x)  "
                  f"(mean={auprc_stats['mean']:.4f}, std={auprc_stats['std']:.4f})",
            xaxis_title="Test AUPRC", yaxis_title="Count",
            template="plotly_white", width=800, height=440,
        )
        fig.write_html(str(run_dir / "auprc_distribution.html"))

        logger.info("Saved plotly artifacts to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save plotly artifacts: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
