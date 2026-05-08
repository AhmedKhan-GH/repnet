"""RepNet Baseline — patient-grouped 80/20 + 5-fold CV.

Single training run using the original RepNet baseline defaults
(lr=1e-3, dropout=0.1) with patient-grouped splits — fixes the leakage
issue in the old optuna_baseline study (which used non-grouped splits
on the pre-balanced dataset).

Usage:
    python -m src.train_repnet_baseline
    python -m src.train_repnet_baseline --n-folds 5 --epochs 50
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
from src.models.repnet_baseline import RepNet, RepNetBaselineModel
from src.preprocessing.augmentation import (
    AmplitudeScaling,
    GaussianNoise,
    RandomTimeShift,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

PARAMS = dict(
    stage_filters = (32, 64),       # original RepNet baseline (2-stage)
    wide_kernel   = 7,
    narrow_kernel = 5,
    dropout       = 0.1,            # RepNet baseline default
    lr            = 1e-3,           # RepNet baseline default
    batch_size    = 64,
    loss_fn       = "weighted",     # handles class imbalance on unbalanced dataset
)


# ---------------------------------------------------------------------------
# QC + preprocessing + augmentation
# ---------------------------------------------------------------------------

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


def augment_train(
    X: np.ndarray, y: np.ndarray, seed: int = SEED, n_copies: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenative augmentation matching the original baseline pipeline."""
    parts_X = [X]
    parts_y = [y]
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


# ---------------------------------------------------------------------------
# CV / final retrain
# ---------------------------------------------------------------------------

def run_cv(
    X_dev: np.ndarray, y_dev: np.ndarray, folds: list, epochs: int,
) -> tuple[list[float], list[float]]:
    aurocs, auprcs = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)
        logger.info(
            "  Fold %d/%d — train=%d (concat-augmented)  val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr), len(y_val),
            int((y_val == 1).sum()), int((y_val == 0).sum()),
        )

        model = RepNetBaselineModel(**PARAMS, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)

        probs = model.predict_proba(X_val)
        aurocs.append(float(roc_auc_score(y_val, probs)))
        auprcs.append(float(average_precision_score(y_val, probs)))
        print(f"  → Fold {fold_idx+1} AUROC={aurocs[-1]:.4f}  AUPRC={auprcs[-1]:.4f}")

    return aurocs, auprcs


def train_final(
    X_dev: np.ndarray, y_dev: np.ndarray, g_dev: np.ndarray,
    epochs: int, seed: int,
) -> RepNetBaselineModel:
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))

    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]

    X_tr, y_tr = augment_train(X_tr, y_tr, seed=seed)
    logger.info(
        "Final training: %d train (concat-augmented) + %d early-stop (clean)",
        len(y_tr), len(y_es),
    )
    model = RepNetBaselineModel(**PARAMS, epochs=epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    return model


# ---------------------------------------------------------------------------
# Test report
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


# ---------------------------------------------------------------------------
# Architecture rendering
# ---------------------------------------------------------------------------

def render_architecture(net_params: dict, run_dir: Path) -> str:
    model = RepNet(**net_params).eval()
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

    print("\n" + "=" * 60)
    print("  Model architecture — RepNet Baseline (2-stage)")
    print("=" * 60)
    print(text)

    (run_dir / "architecture.txt").write_text(text, encoding="utf-8")

    onnx_path = run_dir / "repnet_baseline.onnx"
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["ecg"], output_names=["logits"],
            dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        print(f"\n  ONNX exported: {onnx_path}\n")
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RepNet Baseline — patient-grouped CV")
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-folds",  type=int, default=5)
    parser.add_argument("--epochs",   type=int, default=50)
    args = parser.parse_args()

    run_dir = Path("cv_results") / f"repnet_baseline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")],
    )
    logger.info("Output: %s", run_dir)

    net_params = {k: PARAMS[k] for k in
                  ("stage_filters", "wide_kernel", "narrow_kernel", "dropout")}
    render_architecture(net_params, run_dir)

    logger.info("Loading from %s", args.data_dir)
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
            "data_dir":      args.data_dir,
            "n_folds":       args.n_folds,
            "epochs":        args.epochs,
            "seed":          SEED,
            "params":        {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            "qc":            qc,
            "n_dev":         int(len(y_dev)),
            "n_test":        int(len(y_test)),
            "dev_pos_rate":  float(y_dev.mean()),
            "test_pos_rate": float(y_test.mean()),
        }, f, indent=2)

    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)

    print(f"\n{'#'*60}\n  RepNet Baseline — {args.n_folds}-fold patient-grouped CV\n{'#'*60}")
    cv_aurocs, cv_auprcs = run_cv(X_dev, y_dev, folds, epochs=args.epochs)

    print(f"\n  Retraining on full dev set ...")
    final_model = train_final(X_dev, y_dev, g_dev, epochs=args.epochs, seed=SEED)
    probs_test = final_model.predict_proba(X_test)
    torch.save(final_model.model.state_dict(), run_dir / "model_repnet_baseline.pt")

    test_text, test_metrics = format_test_report(y_test, probs_test, g_test)

    cv_arr_auroc = np.array(cv_aurocs)
    cv_arr_auprc = np.array(cv_auprcs)
    summary_lines = [
        f"\n{'='*60}",
        f"  RepNet Baseline (lr={PARAMS['lr']}, dropout={PARAMS['dropout']})",
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

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        epochs_axis = list(range(1, len(final_model.history["train_loss"]) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Val AUROC"))
        fig.add_trace(go.Scatter(x=epochs_axis, y=final_model.history["train_loss"], name="Train Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs_axis, y=final_model.history["val_auroc"],   name="Val AUROC"),  row=1, col=2)
        fig.update_layout(title="RepNet Baseline — Training Curves",
                          xaxis_title="Epoch", xaxis2_title="Epoch")
        fig.write_html(str(run_dir / "training_curves.html"))
    except Exception as e:
        logger.warning("Could not save training curves: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
