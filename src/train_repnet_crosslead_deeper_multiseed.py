"""RepNet CrossLead Deeper — best Optuna config, multi-seed retrain on 80/20.

Fixes the 80/20 patient-grouped data split (split_seed=42), then retrains the
best-config model N times (default 30) with different *training* seeds (init,
mini-batch order, dropout mask, augmentation, early-stop split). Reports the
distribution of test AUROC across seeds.

Purpose: separate training noise from data noise. The single 0.7952 reported
in the filter-study summary is one draw from this distribution — this script
estimates its mean, std, and 95% CI.

The data split is *not* varied across seeds. We want to characterize how much
the same architecture, on the same train/test partition, varies purely due to
SGD non-determinism. Bootstrap CIs on a single point estimate already cover
test-set sampling.

Outputs (cv_results/repnet_crosslead_deeper_multiseed_<timestamp>/):
  - config.json                    — full configuration
  - results.log                    — training log
  - results.json                   — per-seed metrics + aggregate stats
  - per_seed_history.json          — train_loss / val_auroc per epoch, all seeds
  - best_model.pt                  — weights from highest-AUROC seed
  - best_model_history.json        — that seed's training curves + AUROC
  - best_model_training_curves.html
  - all_seeds_val_auroc.html       — overlay of all seeds (best highlighted)
  - auroc_distribution.html        — histogram of test AUROC
  - all_probs.npz                  — test probs from every seed (for ensembling)
  - summary.txt                    — human-readable

Usage:
    python -m src.train_repnet_crosslead_deeper_multiseed
    python -m src.train_repnet_crosslead_deeper_multiseed --n-seeds 30 --epochs 50
    python -m src.train_repnet_crosslead_deeper_multiseed --seeds 42 43 44 45 46
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
)
from sklearn.model_selection import GroupShuffleSplit

from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.models.repnet_crosslead_deeper import (
    RepNetCrossLeadDeeper,
    RepNetCrossLeadDeeperModel,
)
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SPLIT_SEED = 42       # data split fixed across all training seeds
TEST_SIZE  = 0.20     # 80/20, same as the filter-study holdout

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


class _SeededTrainer(RepNetCrossLeadDeeperModel):
    """Sets torch + cuda seeds before model construction so init varies per run."""

    def __init__(self, *, train_seed: int, **kwargs):
        super().__init__(**kwargs)
        self.train_seed = train_seed

    def fit(self, X_train, y_train, X_val, y_val):
        torch.manual_seed(self.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train_seed)

        self.model = RepNetCrossLeadDeeper(**self.net_params).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        gen = torch.Generator()
        gen.manual_seed(self.train_seed)
        train_dl = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True, num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=torch.cuda.is_available(),
            generator=gen,
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
            val_auc  = self._score_device(Xv, y_val)
            self.history["train_loss"].append(avg_loss)
            self.history["val_auroc"].append(val_auc)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                marker = " *"

            print(f"    Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"val_AUROC={val_auc:.4f}{marker}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)


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


def augment_balance_train(X, y, seed):
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


def train_and_eval_one_seed(X_dev, y_dev, g_dev, X_test, y_test, seed, epochs):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))
    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]

    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=seed)
    print(f"  seed={seed}: train={len(y_tr)} (1:1)  early-stop={len(y_es)}")

    model = _SeededTrainer(**PARAMS, epochs=epochs, train_seed=seed)
    model.fit(X_tr, y_tr, X_es, y_es)

    probs = model.predict_proba(X_test)
    return {
        "seed":   int(seed),
        "auroc":  float(roc_auc_score(y_test, probs)),
        "auprc":  float(average_precision_score(y_test, probs)),
        "brier":  float(brier_score_loss(y_test, probs)),
        "probs":  probs,
        "history": {
            "train_loss": [float(v) for v in model.history["train_loss"]],
            "val_auroc":  [float(v) for v in model.history["val_auroc"]],
        },
        "state_dict": {k: v.cpu().clone() for k, v in model.model.state_dict().items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed retrain of best CrossLead Deeper config on 80/20."
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument(
        "--n-seeds", type=int, default=30,
        help="Number of training seeds (used if --seeds not given). Seeds = 42..42+N-1.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit list of training seeds (overrides --n-seeds).",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(42, 42 + args.n_seeds))

    run_dir = Path("cv_results") / (
        f"repnet_crosslead_deeper_multiseed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d", len(y), int(y.sum()), int((y == 0).sum()))

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=TEST_SIZE, seed=SPLIT_SEED,
    )
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(),
                len(y_test), 100 * y_test.mean())

    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":      args.data_dir,
            "split_seed":    SPLIT_SEED,
            "test_size":     TEST_SIZE,
            "epochs":        args.epochs,
            "training_seeds": seeds,
            "params":        {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "aug_sigma":     AUG_SIGMA,
            "aug_max_shift": AUG_MAX_SHIFT,
            "qc":            qc,
            "n_dev":         int(len(y_dev)),
            "n_test":        int(len(y_test)),
        }, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  Multi-seed retrain — N={len(seeds)} training seeds, 80/20 split")
    print(f"  Test set is identical across all seeds (split_seed=42)")
    print(f"{'#'*60}\n")

    results = []
    histories = {}
    all_probs = np.zeros((len(seeds), len(y_test)), dtype=np.float64)
    best_seed = None
    best_auroc = -1.0
    best_state = None
    best_history = None

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{len(seeds)}  (training seed={seed}) ---")
        r = train_and_eval_one_seed(X_dev, y_dev, g_dev, X_test, y_test, seed, args.epochs)
        results.append({k: v for k, v in r.items() if k not in ("probs", "state_dict", "history")})
        all_probs[i] = r["probs"]
        histories[int(seed)] = r["history"]

        if r["auroc"] > best_auroc:
            best_auroc = r["auroc"]
            best_seed = int(seed)
            best_state = r["state_dict"]
            best_history = r["history"]

        print(f"  -> AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  Brier={r['brier']:.4f}")

        # Persist incrementally so partial runs are still useful
        with open(run_dir / "per_seed_history.json", "w", encoding="utf-8") as f:
            json.dump(histories, f, indent=2)

    aurocs = np.array([r["auroc"] for r in results])
    auprcs = np.array([r["auprc"] for r in results])
    briers = np.array([r["brier"] for r in results])

    # Ensemble: averaged probabilities across all seeds
    ens_probs = all_probs.mean(axis=0)
    ens_auroc = float(roc_auc_score(y_test, ens_probs))
    ens_auprc = float(average_precision_score(y_test, ens_probs))
    ens_brier = float(brier_score_loss(y_test, ens_probs))

    # Save best model weights + its training curve
    if best_state is not None:
        torch.save(best_state, run_dir / "best_model.pt")
        with open(run_dir / "best_model_history.json", "w", encoding="utf-8") as f:
            json.dump({"seed": best_seed, "auroc": best_auroc, "history": best_history},
                      f, indent=2)
        logger.info("Saved best model (seed=%d, AUROC=%.4f) to %s",
                    best_seed, best_auroc, run_dir / "best_model.pt")

    # Distribution statistics
    n = len(aurocs)
    sem = float(aurocs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    auroc_stats = {
        "n": int(n),
        "mean":  float(aurocs.mean()),
        "std":   float(aurocs.std(ddof=1)) if n > 1 else 0.0,
        "sem":   sem,
        "min":   float(aurocs.min()),
        "max":   float(aurocs.max()),
        "median": float(np.median(aurocs)),
        "q25":   float(np.percentile(aurocs, 25)),
        "q75":   float(np.percentile(aurocs, 75)),
        "ci95_lo": float(aurocs.mean() - 1.96 * sem),
        "ci95_hi": float(aurocs.mean() + 1.96 * sem),
    }

    summary_lines = [
        "",
        "=" * 60,
        f"  Multi-seed test AUROC over N={n} retrains (80/20 split)",
        "=" * 60,
        f"  AUROC : mean={auroc_stats['mean']:.4f}  std={auroc_stats['std']:.4f}  "
        f"sem={auroc_stats['sem']:.4f}",
        f"          95% CI: [{auroc_stats['ci95_lo']:.4f}, {auroc_stats['ci95_hi']:.4f}]   "
        f"min={auroc_stats['min']:.4f}  max={auroc_stats['max']:.4f}   "
        f"median={auroc_stats['median']:.4f}  IQR=[{auroc_stats['q25']:.4f}, {auroc_stats['q75']:.4f}]",
        f"          per-seed: {['%.4f' % v for v in aurocs]}",
        f"  AUPRC : mean={auprcs.mean():.4f}  std={auprcs.std(ddof=1):.4f}  "
        f"min={auprcs.min():.4f}  max={auprcs.max():.4f}",
        f"  Brier : mean={briers.mean():.4f}  std={briers.std(ddof=1):.4f}",
        "",
        f"  Best single seed: #{best_seed}  AUROC={best_auroc:.4f}  "
        f"(saved to best_model.pt)",
        f"  Ensemble (mean prob across seeds): "
        f"AUROC={ens_auroc:.4f}  AUPRC={ens_auprc:.4f}  Brier={ens_brier:.4f}",
        "",
        f"  Test : N={len(y_test)}  pos={int(y_test.sum())}  neg={int((y_test==0).sum())}",
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "seeds":         [int(s) for s in seeds],
            "per_seed":      results,
            "auroc_stats":   auroc_stats,
            "auprc_mean":    float(auprcs.mean()),
            "auprc_std":     float(auprcs.std(ddof=1)) if n > 1 else 0.0,
            "brier_mean":    float(briers.mean()),
            "brier_std":     float(briers.std(ddof=1)) if n > 1 else 0.0,
            "best_seed":     best_seed,
            "best_auroc":    best_auroc,
            "ensemble":      {"auroc": ens_auroc, "auprc": ens_auprc, "brier": ens_brier},
            "n_test":        int(len(y_test)),
        }, f, indent=2)

    np.savez(run_dir / "all_probs.npz",
             y_true=y_test, probs=all_probs, seeds=np.array(seeds), patient_ids=g_test)

    # Plotly artifacts: per-seed training curves + AUROC distribution
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Best-seed training curves (the one whose weights we saved)
        if best_history is not None:
            ep = list(range(1, len(best_history["train_loss"]) + 1))
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Train Loss", "Val AUROC (early-stop set)"))
            fig.add_trace(go.Scatter(x=ep, y=best_history["train_loss"], name="train_loss"),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=ep, y=best_history["val_auroc"], name="val_auroc"),
                          row=1, col=2)
            fig.update_layout(
                title=f"Best seed (#{best_seed}) training curves — test AUROC={best_auroc:.4f}",
                xaxis_title="Epoch", xaxis2_title="Epoch",
            )
            fig.write_html(str(run_dir / "best_model_training_curves.html"))

        # All-seed val_auroc overlay (best seed highlighted)
        fig = go.Figure()
        for sd, hist in histories.items():
            ep = list(range(1, len(hist["val_auroc"]) + 1))
            is_best = (sd == best_seed)
            fig.add_trace(go.Scatter(
                x=ep, y=hist["val_auroc"],
                name=f"seed {sd}" + ("  (best)" if is_best else ""),
                line=dict(width=3 if is_best else 1, color="red" if is_best else None),
                opacity=1.0 if is_best else 0.4,
            ))
        fig.update_layout(
            title=f"Per-seed val AUROC across {n} seeds",
            xaxis_title="Epoch", yaxis_title="Val AUROC",
        )
        fig.write_html(str(run_dir / "all_seeds_val_auroc.html"))

        # AUROC histogram
        fig = go.Figure(data=[go.Histogram(x=aurocs, nbinsx=min(20, max(5, n // 2)))])
        fig.add_vline(x=auroc_stats["mean"], line_dash="dash", line_color="red",
                      annotation_text=f"mean={auroc_stats['mean']:.4f}")
        fig.add_vline(x=best_auroc, line_dash="dot", line_color="green",
                      annotation_text=f"best={best_auroc:.4f}")
        fig.update_layout(
            title=f"Test AUROC distribution across {n} seeds  "
                  f"(mean={auroc_stats['mean']:.4f}, std={auroc_stats['std']:.4f})",
            xaxis_title="Test AUROC", yaxis_title="Count",
        )
        fig.write_html(str(run_dir / "auroc_distribution.html"))

        logger.info("Saved plotly artifacts to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save plotly artifacts: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
