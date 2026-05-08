"""Model evaluation report — computes and prints all standard metrics.

Usage:
    # From a results directory (no GPU / model needed):
    python -m src.evaluation.report crossval_results/repnet_crosslead_deeper_multiseed_pe/2026-05-04_21-06-34

    # Or from code:
    from src.evaluation.report import report_from_dir
    report_from_dir("crossval_results/...")
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)


def operating_point_metrics(y_true: np.ndarray, probs: np.ndarray, tau: float) -> dict:
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(
        threshold=float(tau),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(ppv),
        npv=float(npv),
        f1=float(f1),
        accuracy=float(acc),
    )


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def sensitivity_threshold(y_true: np.ndarray, probs: np.ndarray, target: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    valid = tpr >= target
    if not valid.any():
        return 0.0
    return float(thresholds[valid][np.argmax(thresholds[valid])])


def bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
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
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return point, lo, hi


def full_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    patient_ids: np.ndarray | None = None,
    label: str = "Model",
) -> dict:
    """Compute all metrics and print a formatted table.

    Returns a dict with all computed values.
    """
    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)

    auroc_ci = auprc_ci = None
    if patient_ids is not None:
        _, auroc_lo, auroc_hi = bootstrap_ci(y_true, probs, patient_ids, roc_auc_score)
        _, auprc_lo, auprc_hi = bootstrap_ci(y_true, probs, patient_ids, average_precision_score)
        auroc_ci = (auroc_lo, auroc_hi)
        auprc_ci = (auprc_lo, auprc_hi)

    tau_youden = youden_threshold(y_true, probs)
    tau_s90 = sensitivity_threshold(y_true, probs, 0.90)
    tau_s85 = sensitivity_threshold(y_true, probs, 0.85)

    ops = {
        "tau=0.50": operating_point_metrics(y_true, probs, 0.50),
        f"tau={tau_youden:.3f} (Youden)": operating_point_metrics(y_true, probs, tau_youden),
        f"tau={tau_s90:.3f} (sens>=.90)": operating_point_metrics(y_true, probs, tau_s90),
        f"tau={tau_s85:.3f} (sens>=.85)": operating_point_metrics(y_true, probs, tau_s85),
    }

    # --- Print ---
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    w = 72

    print("=" * w)
    print(f"  {label}")
    print(f"  Test set: N={len(y_true)}  pos={n_pos}  neg={n_neg}  prevalence={n_pos/len(y_true):.1%}")
    print("=" * w)

    ci_str = ""
    if auroc_ci:
        ci_str = f"  95% CI [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]"
    print(f"  AUROC      : {auroc:.4f}{ci_str}")

    ci_str = ""
    if auprc_ci:
        ci_str = f"  95% CI [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]"
    print(f"  AUPRC      : {auprc:.4f}{ci_str}")

    print(f"  Brier      : {brier:.4f}")
    print()

    hdr = (
        f"  {'Operating point':<25} {'Sens':>6} {'Spec':>6} {'Acc':>6} "
        f"{'Prec':>6} {'F1':>6} {'NPV':>6}  "
        f"{'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, m in ops.items():
        print(
            f"  {name:<25} {m['sensitivity']:>6.3f} {m['specificity']:>6.3f} "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['f1']:>6.3f} "
            f"{m['npv']:>6.3f}  {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {m['tn']:>3}"
        )
    print("=" * w)

    return dict(
        auroc=auroc, auroc_ci=auroc_ci,
        auprc=auprc, auprc_ci=auprc_ci,
        brier=brier,
        operating_points=ops,
        n_test=len(y_true), n_pos=n_pos, n_neg=n_neg,
    )


def report_from_dir(run_dir: str | Path) -> dict:
    """Load a multiseed results directory and print reports for best model and ensemble."""
    run_dir = Path(run_dir)

    npz = np.load(run_dir / "all_probs.npz", allow_pickle=True)
    y_true = npz["y_true"]
    all_probs = npz["probs"]       # (n_seeds, n_test)
    seeds = npz["seeds"]
    patient_ids = npz["patient_ids"]

    with open(run_dir / "results.json") as f:
        results = json.load(f)

    best_seed = results["best_seed"]
    best_idx = int(np.where(seeds == best_seed)[0][0])
    probs_best = all_probs[best_idx]
    probs_ens = all_probs.mean(axis=0)

    print()
    best_metrics = full_report(
        y_true, probs_best, patient_ids,
        label=f"Best single model (seed #{best_seed})",
    )

    print()
    ens_metrics = full_report(
        y_true, probs_ens, patient_ids,
        label=f"Ensemble — {len(seeds)} seeds (mean probabilities)",
    )

    # Multi-seed summary
    print()
    print("=" * 72)
    print(f"  Multi-seed summary (N={len(seeds)} retrains)")
    print("=" * 72)
    for key, stat_key in [("AUROC", "auroc_stats"), ("AUPRC", "auprc_stats")]:
        s = results[stat_key]
        print(
            f"  {key:<8}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
            f"95% CI [{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]  "
            f"range [{s['min']:.4f}, {s['max']:.4f}]"
        )
    s = results["sens_sp80_stats"]
    print(
        f"  {'Sens@80':<8}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
        f"range [{s['min']:.4f}, {s['max']:.4f}]"
    )
    print(f"  Brier    : mean={results['brier_mean']:.4f}  std={results['brier_std']:.4f}")
    print("=" * 72)

    return {"best": best_metrics, "ensemble": ens_metrics, "multiseed": results}


def main():
    parser = argparse.ArgumentParser(description="Model evaluation report")
    parser.add_argument("run_dir", type=Path, help="Path to results directory")
    args = parser.parse_args()
    report_from_dir(args.run_dir)


if __name__ == "__main__":
    main()
