"""SimpleResNet1D — majority-undersample only, no augmentation, slow training.

Pipeline:
  1. Load with patient IDs → drop NaN/Inf, flat leads, missing IDs
  2. Patient-grouped 80/20 holdout split (no patient on both sides)
  3. Preprocess (BWF 0.5 Hz HP + Notch 60 Hz + Z-score per lead)
  4. Patient-grouped 3-fold CV on the dev set
       - Per fold: MajorityUndersampling(ratio=1.0) on TRAIN only (no augmentation)
       - Val fold: preprocessed only
  5. Final retrain on full dev (with the same balance pipeline) → score on test
  6. Test report: AUROC/AUPRC with patient-level bootstrap CIs.

Usage:
    python -m src.train_simple_resnet1d
    python -m src.train_simple_resnet1d --epochs 300 --lr 5e-5 --dropout 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 stdout so torchinfo's tree chars render on Windows consoles.
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
from src.models.simple_resnet1d import SimpleResNet1D, SimpleResNet1DModel
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.preprocessing.sampling import MajorityUndersampling

logger = logging.getLogger(__name__)

SEED = 42

# Defaults: very low LR, low dropout, long training (overridable via CLI).
PARAMS = dict(
    stem_channels   = 32,
    block_channels  = (32, 64, 128, 256),
    block_strides   = (1, 2, 2, 1),
    kernel_size     = 11,
    stem_kernel     = 11,
    stem_stride     = 1,
    dropout         = 0.1,
    block_dropout   = 0.0,
    lr              = 5e-5,
    weight_decay    = 1e-4,
    batch_size      = 32,
    patience        = 50,
    loss_fn         = "cross_entropy",
    use_cosine_lr   = False,
)


# ---------------------------------------------------------------------------
# QC + preprocessing + balancing
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


def balance_train(
    X: np.ndarray, y: np.ndarray, seed: int = SEED, undersample: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Majority-undersample to 1:1 — no augmentation.

    If ``undersample=False`` (e.g. when using weighted CE loss), the full
    unbalanced set is returned, just shuffled.
    """
    if undersample:
        X, y = MajorityUndersampling(ratio=1.0, seed=seed).transform(X, y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# CV / training
# ---------------------------------------------------------------------------

def run_cv(
    X_dev: np.ndarray, y_dev: np.ndarray, folds: list, params: dict, epochs: int,
) -> tuple[list[float], list[float]]:
    undersample = params["loss_fn"] != "weighted"
    aurocs, auprcs = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr   = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx],   y_dev[val_idx]

        X_tr, y_tr = balance_train(X_tr, y_tr, seed=SEED + fold_idx, undersample=undersample)
        balance_label = "1:1, undersampled" if undersample else f"unbalanced, weighted CE"
        logger.info(
            "  Fold %d/%d — train=%d (%s, no aug)  val=%d (pos=%d neg=%d)",
            fold_idx + 1, len(folds), len(y_tr), balance_label, len(y_val),
            int((y_val == 1).sum()), int((y_val == 0).sum()),
        )

        model = SimpleResNet1DModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)

        probs = model.predict_proba(X_val)
        aurocs.append(float(roc_auc_score(y_val, probs)))
        auprcs.append(float(average_precision_score(y_val, probs)))
        print(f"  → Fold {fold_idx+1} AUROC={aurocs[-1]:.4f}  AUPRC={auprcs[-1]:.4f}")

    return aurocs, auprcs


def train_final(
    X_dev: np.ndarray, y_dev: np.ndarray, g_dev: np.ndarray,
    params: dict, epochs: int, seed: int,
) -> SimpleResNet1DModel:
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_dev, y_dev, g_dev))

    X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
    X_es, y_es = X_dev[es_idx], y_dev[es_idx]

    undersample = params["loss_fn"] != "weighted"
    X_tr, y_tr = balance_train(X_tr, y_tr, seed=seed, undersample=undersample)
    balance_label = "1:1, undersampled" if undersample else "unbalanced, weighted CE"
    logger.info(
        "Final training: %d train (%s, no aug) + %d early-stop (clean)",
        len(y_tr), balance_label, len(y_es),
    )
    model = SimpleResNet1DModel(**params, epochs=epochs)
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
        f"  {'Operating point':<18} {'τ':>7}  {'sens':>6}  {'spec':>6}  {'PPV':>5}  {'NPV':>5}  {'F1':>5}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}",
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
    model = SimpleResNet1D(**net_params).eval()
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
        text = f"{model}\n\nTotal params: {n_params:,}\n(install `torchinfo` for layer-wise table)"

    print("\n" + "=" * 60)
    print("  Model architecture")
    print("=" * 60)
    print(text)

    (run_dir / "architecture.txt").write_text(text, encoding="utf-8")

    onnx_path = run_dir / "simple_resnet1d.onnx"
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
    parser = argparse.ArgumentParser(description="SimpleResNet1D — slow training, undersample only")
    parser.add_argument("--data-dir",  default="data/seniordesign_upload")
    parser.add_argument("--n-folds",   type=int,   default=3)
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--lr",            type=float, default=PARAMS["lr"])
    parser.add_argument("--dropout",       type=float, default=PARAMS["dropout"])
    parser.add_argument("--block-dropout", type=float, default=PARAMS["block_dropout"])
    parser.add_argument("--weight-decay",  type=float, default=PARAMS["weight_decay"])
    parser.add_argument("--batch-size",    type=int,   default=PARAMS["batch_size"])
    parser.add_argument("--patience",      type=int,   default=PARAMS["patience"])
    parser.add_argument("--loss-fn",       choices=["cross_entropy", "weighted"],
                        default=PARAMS["loss_fn"],
                        help="weighted: class-weighted CE on full unbalanced data (no undersample)")
    parser.add_argument("--cosine-lr",     action="store_true",
                        help="Cosine-anneal LR over epochs")
    parser.add_argument("--size",          choices=["tiny", "small", "medium", "large"],
                        default="large",
                        help="Capacity preset. tiny=26K, small=95K, medium=375K, large=1.5M params")
    args = parser.parse_args()

    SIZE_PRESETS = {
        "tiny":   dict(stem_channels=8,  block_channels=(8,   8,  16,  32)),
        "small":  dict(stem_channels=8,  block_channels=(8,  16,  32,  64)),
        "medium": dict(stem_channels=16, block_channels=(16, 32,  64, 128)),
        "large":  dict(stem_channels=32, block_channels=(32, 64, 128, 256)),
    }

    params = dict(PARAMS)
    params.update(SIZE_PRESETS[args.size])
    params.update(
        lr            = args.lr,
        dropout       = args.dropout,
        block_dropout = args.block_dropout,
        weight_decay  = args.weight_decay,
        batch_size    = args.batch_size,
        patience      = args.patience,
        loss_fn       = args.loss_fn,
        use_cosine_lr = args.cosine_lr,
    )

    run_dir = Path("cv_results") / f"simple_resnet1d_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "results.log", encoding="utf-8")],
    )
    logger.info("Output: %s", run_dir)

    # 0. Render architecture
    net_params = {k: params[k] for k in
                  ("stem_channels", "block_channels", "block_strides",
                   "kernel_size", "stem_kernel", "stem_stride",
                   "dropout", "block_dropout")}
    render_architecture(net_params, run_dir)

    # 1. Load + QC
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

    # 3. Preprocess
    X_dev  = preprocess(X_dev)
    X_test = preprocess(X_test)

    # Save config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir":      args.data_dir,
            "n_folds":       args.n_folds,
            "epochs":        args.epochs,
            "seed":          SEED,
            "params":        {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
            "qc":            qc,
            "n_dev":         int(len(y_dev)),
            "n_test":        int(len(y_test)),
            "dev_pos_rate":  float(y_dev.mean()),
            "test_pos_rate": float(y_test.mean()),
        }, f, indent=2)

    # 4. Patient-grouped k-fold CV
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=SEED)

    print(f"\n{'#'*60}\n  SimpleResNet1D — {args.n_folds}-fold patient-grouped CV"
          f"\n  lr={params['lr']:.1e}  dropout={params['dropout']}  "
          f"block_dropout={params['block_dropout']}  wd={params['weight_decay']:.1e}"
          f"\n  loss={params['loss_fn']}  cosine_lr={params['use_cosine_lr']}  "
          f"epochs={args.epochs}  patience={params['patience']}\n{'#'*60}")
    cv_aurocs, cv_auprcs = run_cv(X_dev, y_dev, folds, params=params, epochs=args.epochs)

    # 5. Final retrain + test eval
    print(f"\n  Retraining on full dev set …")
    final_model = train_final(X_dev, y_dev, g_dev, params=params, epochs=args.epochs, seed=SEED)
    probs_test = final_model.predict_proba(X_test)
    torch.save(final_model.model.state_dict(), run_dir / "model_simple_resnet1d.pt")

    # 6. Test report
    test_text, test_metrics = format_test_report(y_test, probs_test, g_test)

    cv_arr_auroc = np.array(cv_aurocs)
    cv_arr_auprc = np.array(cv_auprcs)
    summary_lines = [
        f"\n{'='*60}",
        f"  SimpleResNet1D",
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
        fig.add_trace(go.Scatter(x=epochs_axis, y=final_model.history["val_auroc"],   name="Val AUROC"),  row=1, col=2)
        fig.update_layout(title="SimpleResNet1D — Training Curves",
                          xaxis_title="Epoch", xaxis2_title="Epoch")
        fig.write_html(str(run_dir / "training_curves.html"))
    except Exception as e:
        logger.warning("Could not save training curves: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
