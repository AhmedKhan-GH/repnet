"""RepNet CrossLead — patient-grouped 80/20 + 3-fold CV.

Same pipeline as src/train_baseline_resnet1d.py, but with the cross-lead
attention model (per-lead conv + MultiheadAttention across the 12 leads).

Usage:
    python -m src.train_repnet_crosslead
    python -m src.train_repnet_crosslead --n-folds 5 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Windows consoles default to cp1252 — torchinfo's tree chars and ONNX's
# success emoji both crash on it. UTF-8 stdout fixes both.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
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
from src.models.repnet_crosslead import RepNetCrossLead, RepNetCrossLeadModel
from src.preprocessing.augmentation import GaussianNoise, RandomTimeShift
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED = 42

PARAMS = dict(
    stage_filters  = (32, 64),
    wide_kernel    = 7,
    narrow_kernel  = 5,
    dropout        = 0.0636,            # optuna 2026-04-17 best
    n_heads        = 4,
    lr             = 8.76e-4,           # optuna 2026-04-17 best
    batch_size     = 64,                # optuna 2026-04-17 fixed
    loss_fn        = "cross_entropy",   # train set is 1:1 after augment+undersample
)


# ---------------------------------------------------------------------------
# Quality, preprocessing, augmentation
# ---------------------------------------------------------------------------

def quality_filter(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
    flat_std_thresh: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Drop recordings with flat leads or missing patient IDs."""
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


def augment_balance_train(
    X: np.ndarray, y: np.ndarray, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment positives only, then undersample negatives to 1:1 balance."""
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X_pos = X[y == 1]
    X_neg = X[y == 0]

    X_pos_g, _ = GaussianNoise(sigma=0.02).transform(X_pos.copy())
    X_pos_t, _ = RandomTimeShift(max_shift=200).transform(X_pos.copy())

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


# ---------------------------------------------------------------------------
# CV / training loop
# ---------------------------------------------------------------------------

def run_cv(
    X_dev: np.ndarray, y_dev: np.ndarray, folds: list, epochs: int,
) -> tuple[list[float], list[float]]:
    """Patient-grouped k-fold CV. Returns per-fold (AUROC, AUPRC)."""
    aurocs, auprcs = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=SEED + fold_idx)
        logger.info(
            "  Fold %d/%d — train=%d (1:1 balanced)  val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr), len(y_val),
            int((y_val == 1).sum()), int((y_val == 0).sum()),
        )

        model = RepNetCrossLeadModel(**PARAMS, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)

        probs = model.predict_proba(X_val)
        aurocs.append(float(roc_auc_score(y_val, probs)))
        auprcs.append(float(average_precision_score(y_val, probs)))
        print(f"  → Fold {fold_idx+1} AUROC={aurocs[-1]:.4f}  AUPRC={auprcs[-1]:.4f}")

    return aurocs, auprcs


def train_final(
    X_dev: np.ndarray, y_dev: np.ndarray, g_dev: np.ndarray,
    epochs: int, seed: int,
) -> RepNetCrossLeadModel:
    """Retrain on full dev set with patient-grouped 90/10 early-stop split."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))

    X_tr,  y_tr  = X_dev[tr_idx], y_dev[tr_idx]
    X_es,  y_es  = X_dev[es_idx], y_dev[es_idx]

    X_tr, y_tr = augment_balance_train(X_tr, y_tr, seed=seed)
    logger.info(
        "Final training: %d train (augmented+balanced) + %d early-stop (clean)",
        len(y_tr), len(y_es),
    )
    model = RepNetCrossLeadModel(**PARAMS, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    return model


# ---------------------------------------------------------------------------
# Test report — operating points, bootstrap CIs
# ---------------------------------------------------------------------------

def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def sensitivity_threshold(y_true: np.ndarray, probs: np.ndarray, target_sens: float = 0.90) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    valid = tpr >= target_sens
    if not valid.any():
        return 0.0
    return float(thresholds[valid][-1])


def operating_point_metrics(y_true: np.ndarray, probs: np.ndarray, tau: float) -> dict:
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    f1   = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return {
        "threshold": float(tau),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": float(sens), "specificity": float(spec),
        "ppv": float(ppv), "npv": float(npv), "f1": float(f1),
    }


def bootstrap_ci(
    y_true: np.ndarray, probs: np.ndarray, groups: np.ndarray,
    metric_fn, n_boot: int = 1000, seed: int = SEED,
) -> tuple[float, float, float]:
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


def format_test_report(
    y_test: np.ndarray, probs: np.ndarray, g_test: np.ndarray,
) -> tuple[str, dict]:
    auroc_p, auroc_lo, auroc_hi = bootstrap_ci(y_test, probs, g_test, roc_auc_score)
    auprc_p, auprc_lo, auprc_hi = bootstrap_ci(y_test, probs, g_test, average_precision_score)
    brier = float(brier_score_loss(y_test, probs))

    tau_youden = youden_threshold(y_test, probs)
    tau_sens   = sensitivity_threshold(y_test, probs, target_sens=0.90)

    ops = {
        "tau=0.50":           operating_point_metrics(y_test, probs, 0.5),
        "tau=Youden":         operating_point_metrics(y_test, probs, tau_youden),
        "tau=sens>=0.90":     operating_point_metrics(y_test, probs, tau_sens),
    }

    lines = []
    lines.append(f"  AUROC  : {auroc_p:.4f}  [{auroc_lo:.4f}, {auroc_hi:.4f}]   (95% bootstrap, patient-level)")
    lines.append(f"  AUPRC  : {auprc_p:.4f}  [{auprc_lo:.4f}, {auprc_hi:.4f}]")
    lines.append(f"  Brier  : {brier:.4f}")
    lines.append(f"  Test   : N={len(y_test)}  pos={int(y_test.sum())}  neg={int((y_test==0).sum())}")
    lines.append("")
    lines.append(f"  {'Operating point':<18} {'τ':>7}  {'sens':>6}  {'spec':>6}  {'PPV':>5}  {'NPV':>5}  {'F1':>5}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
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


# ---------------------------------------------------------------------------
# Architecture rendering
# ---------------------------------------------------------------------------

def render_architecture(net_params: dict, run_dir: Path) -> str:
    model = RepNetCrossLead(**net_params).eval()
    dummy = torch.randn(1, 12, 2500)

    try:
        from torchinfo import summary
        info = summary(
            model, input_data=dummy, depth=4, verbose=0,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            row_settings=("var_names",),
        )
        text = str(info)
    except ImportError:
        n_params = sum(p.numel() for p in model.parameters())
        text = f"{model}\n\nTotal params: {n_params:,}\n(install `torchinfo` for layer table)"

    print("\n" + "="*60)
    print("  Model architecture — RepNet CrossLead")
    print("="*60)
    print(text)

    (run_dir / "architecture.txt").write_text(text, encoding="utf-8")

    onnx_path = run_dir / "repnet_crosslead.onnx"
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["ecg"], output_names=["logits"],
            dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        print(f"\n  ONNX exported: {onnx_path}")
        print(f"  → drag this file onto https://netron.app for an interactive view\n")
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RepNet CrossLead — patient-grouped CV")
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds",  type=int, default=3)
    parser.add_argument("--epochs",   type=int, default=50)
    args = parser.parse_args()

    run_dir = Path("cv_results") / f"repnet_crosslead_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")],
    )
    logger.info("Output: %s", run_dir)

    # 0. Render architecture
    net_params = {k: PARAMS[k] for k in
                  ("stage_filters", "wide_kernel", "narrow_kernel", "dropout", "n_heads")}
    render_architecture(net_params, run_dir)

    # 1. Load + quality filter
    logger.info("Loading from %s", args.data_dir)
    X, y, patient_ids = load_seniordesign(args.data_dir, return_patient_ids=True)
    X, y, patient_ids, qc = quality_filter(X, y, patient_ids)
    logger.info("After QC: N=%d  pos=%d  neg=%d  (dropped %d flat-lead, %d missing-ID)",
                len(y), int(y.sum()), int((y == 0).sum()),
                qc["flat_lead_dropped"], qc["missing_id_dropped"])

    # 2. Patient-grouped 80/20 holdout
    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, patient_ids, test_size=0.20, seed=SEED,
    )
    logger.info("Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)",
                len(y_dev), 100 * y_dev.mean(),
                len(y_test), 100 * y_test.mean())

    # 3. Preprocess (once, on all data — augmentation/balancing are fold-local)
    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    # Save config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":       args.data_dir,
            "n_folds":        args.n_folds,
            "epochs":         args.epochs,
            "seed":           SEED,
            "params":         {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "qc":             qc,
            "n_dev":          int(len(y_dev)),
            "n_test":         int(len(y_test)),
            "dev_pos_rate":   float(y_dev.mean()),
            "test_pos_rate":  float(y_test.mean()),
        }, f, indent=2)

    # 4. Patient-grouped k-fold CV
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)

    print(f"\n{'#'*60}\n  RepNet CrossLead — {args.n_folds}-fold patient-grouped CV\n{'#'*60}")
    cv_aurocs, cv_auprcs = run_cv(X_dev, y_dev, folds, epochs=args.epochs)

    # 5. Final retrain + test eval
    print(f"\n  Retraining on full dev set …")
    final_model = train_final(X_dev, y_dev, g_dev, epochs=args.epochs, seed=SEED)
    probs_test = final_model.predict_proba(X_test)
    torch.save(final_model.model.state_dict(), run_dir / "model_repnet_crosslead.pt")

    # 6. Test report
    test_text, test_metrics = format_test_report(y_test, probs_test, g_test)

    cv_arr_auroc = np.array(cv_aurocs)
    cv_arr_auprc = np.array(cv_auprcs)
    summary_lines = [
        f"\n{'='*60}",
        f"  RepNet CrossLead",
        f"{'='*60}",
        f"  CV AUROC : {cv_arr_auroc.mean():.4f} ± {cv_arr_auroc.std():.4f}    "
        f"per-fold: {[f'{v:.3f}' for v in cv_aurocs]}",
        f"  CV AUPRC : {cv_arr_auprc.mean():.4f} ± {cv_arr_auprc.std():.4f}    "
        f"per-fold: {[f'{v:.3f}' for v in cv_auprcs]}",
        f"\n  Test set evaluation",
        f"  {'-'*40}",
        test_text,
    ]
    summary = "\n".join(summary_lines)
    print(summary)

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

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

    # Training curves
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        epochs_axis = list(range(1, len(final_model.history["train_loss"]) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Val AUROC"))
        fig.add_trace(go.Scatter(x=epochs_axis, y=final_model.history["train_loss"], name="Train Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs_axis, y=final_model.history["val_auroc"], name="Val AUROC"),    row=1, col=2)
        fig.update_layout(title="RepNet CrossLead — Training Curves",
                          xaxis_title="Epoch", xaxis2_title="Epoch")
        fig.write_html(str(run_dir / "training_curves.html"))
    except Exception as e:
        logger.warning("Could not save training curves: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
