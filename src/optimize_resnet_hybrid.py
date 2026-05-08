"""Optuna search for RepNetResNetHybrid on the user's preferred eval protocol.

Search space (6 dims, prioritized by expected lift):
  lr           : log-uniform [1e-4, 2e-3]
  dropout      : uniform     [0.05, 0.40]
  weight_decay : log-uniform [1e-5, 1e-2]
  f2           : {32, 64, 96}
  f3           : {64, 128, 192}      (f4 tied to f3, no expansion in last block)
  batch_size   : {32, 64}

Skipped (separate experiments / already explored):
  - kernels (EDA-validated: 7 first, 5/5/3 thereafter)
  - n_layers (fixed at 4 per user choice)
  - augmentation strength (orthogonal -- separate experiment)
  - loss function (weighted wins; focal explored, lost)
  - class sampling (over/undersampling explored, both worse)

Protocol:
  - data/seniordesign_upload (~85/15, N=2186 after NaN-patient drop)
  - patient-grouped 80/20 holdout + 5-fold CV
  - weighted CE loss
  - basic 3x augmentation, no over/undersampling

Outputs (saved to optuna_resnet_hybrid/<timestamp>/):
  - study.db          (Optuna SQLite, RESUME-ABLE -- rerun the same command to continue)
  - best_params.json
  - results.log
  - summary.txt
  - best_model.pt     (retrained on full dev with winning params)
  - {contour, parallel, history, importance, slice}_plot.html

Usage:
    python -m src.optimize_resnet_hybrid                      # 80 trials, ~5-7 hrs
    python -m src.optimize_resnet_hybrid --n-trials 60        # 4-5 hrs
    python -m src.optimize_resnet_hybrid --n-trials 100       # 7-10 hrs

Resume an interrupted study:
    Just rerun the same command in the same output directory -- TPE picks up
    where it left off thanks to the SQLite storage. To resume an earlier study,
    point --resume-dir at the previous run directory.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.data.dataset import (
    kfold_cv_indices_grouped,
    load_seniordesign,
    split_holdout_grouped,
)
from src.models.repnet_resnet_hybrid import RepNetResNetHybridModel
from src.preprocessing.augmentation import (
    AmplitudeScaling,
    GaussianNoise,
    RandomTimeShift,
)
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)

SEED = 42

# Architecture stays at 4-block (1 per-lead + 3 mixed). Kernels EDA-validated.
FIXED_PARAMS = dict(
    f1=32,                 # per-lead block output channels
    wide_kernel=7,
    narrow_kernel=5,
    narrow_kernel_2=3,
    loss_fn="weighted",
)


def preprocess(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


def augment_train(X: np.ndarray, y: np.ndarray,
                  seed: int = SEED, n_copies: int = 2) -> tuple[np.ndarray, np.ndarray]:
    parts_X, parts_y = [X], [y]
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


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def _save_trial_artifacts(trial_dir: Path, trial: optuna.Trial,
                          params: dict, fold_aurocs: list[float],
                          fold_histories: list[dict], pruned_at: int | None = None) -> None:
    """Persist everything needed to reconstruct a trial's training curves later."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_number": trial.number,
        "params": params,
        "fold_aurocs": [float(a) for a in fold_aurocs],
        "cv_mean": float(np.mean(fold_aurocs)) if fold_aurocs else None,
        "cv_std": float(np.std(fold_aurocs)) if fold_aurocs else None,
        "pruned_at_fold": pruned_at,
        # Training curves: per-fold dict of {train_loss: [...], val_auroc: [...]}
        # Reconstruct epoch axis as range(1, len+1).
        "fold_histories": fold_histories,
    }
    out = trial_dir / f"trial_{trial.number:03d}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def objective(trial: optuna.Trial, X_dev, y_dev, folds, epochs: int,
              trials_dir: Path) -> float:
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.40)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    f2 = trial.suggest_categorical("f2", [32, 64, 96])
    f3 = trial.suggest_categorical("f3", [64, 128, 192])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    params = {
        **FIXED_PARAMS,
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "f2": f2,
        "f3": f3,
        "f4": f3,           # tie f4 = f3 (no expansion in last block)
        "batch_size": batch_size,
    }

    fold_aurocs: list[float] = []
    fold_histories: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        X_tr, y_tr = augment_train(X_tr, y_tr, seed=SEED + fold_idx)

        model = RepNetResNetHybridModel(**params, epochs=epochs)
        model.fit(X_tr, y_tr, X_val, y_val)
        auroc = model.score(X_val, y_val)
        fold_aurocs.append(auroc)
        # Capture full per-epoch curves for this fold.
        fold_histories.append({
            "fold_idx": fold_idx,
            "train_loss": [float(x) for x in model.history["train_loss"]],
            "val_auroc": [float(x) for x in model.history["val_auroc"]],
            "best_val_auroc": float(auroc),
            "n_epochs_run": len(model.history["train_loss"]),
        })

        logger.info(
            "Trial %d | lr=%.5f drop=%.3f wd=%.5f f2=%d f3=%d bs=%d | fold %d/%d | AUROC=%.4f",
            trial.number, lr, dropout, weight_decay, f2, f3, batch_size,
            fold_idx + 1, len(folds), auroc,
        )

        # Optuna pruning at the fold level: if first 2 folds are clearly bad,
        # don't waste time on the remaining 3.
        trial.report(float(np.mean(fold_aurocs)), fold_idx)
        if trial.should_prune():
            logger.info("Trial %d pruned at fold %d", trial.number, fold_idx + 1)
            # Persist what we have before pruning so curves aren't lost.
            _save_trial_artifacts(trials_dir, trial, params, fold_aurocs,
                                   fold_histories, pruned_at=fold_idx + 1)
            raise optuna.TrialPruned()

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    logger.info(
        "Trial %d | DONE | CV AUROC=%.4f (+/- %.4f) | params=%s",
        trial.number, mean_auroc, std_auroc, trial.params,
    )
    _save_trial_artifacts(trials_dir, trial, params, fold_aurocs, fold_histories)
    return mean_auroc


def main():
    parser = argparse.ArgumentParser(
        description="Optuna search: lr/dropout/wd/f2/f3/batch for RepNetResNetHybrid",
    )
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Resume from an existing optuna_resnet_hybrid/<timestamp>/ directory")
    args = parser.parse_args()

    if args.resume_dir is not None:
        run_dir = Path(args.resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"resume-dir does not exist: {run_dir}")
    else:
        run_dir = Path("optuna_resnet_hybrid") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "results.log"),
        ],
    )
    logger.info("Output directory: %s", run_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Device: %s", torch.cuda.get_device_name(0))

    # Load + drop NaN-patient rows (5 in seniordesign_upload)
    X, y, groups = load_seniordesign(args.data_dir, return_patient_ids=True)
    valid = ~np.isnan(groups.astype(float))
    n_drop = int((~valid).sum())
    if n_drop:
        logger.info("Dropping %d rows with missing Pat_Obfus_MRN", n_drop)
        X, y, groups = X[valid], y[valid], groups[valid]
    logger.info(
        "Loaded %d samples (pos=%d, neg=%d, pos_rate=%.1f%%)",
        len(y), int((y == 1).sum()), int((y == 0).sum()), 100 * y.mean(),
    )
    X = preprocess(X)

    X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
        X, y, groups, test_size=0.20, seed=args.seed,
    )
    folds = kfold_cv_indices_grouped(y_dev, g_dev, n_folds=args.n_folds, seed=args.seed)
    logger.info(
        "Dev: %d (pos=%.1f%%)  Test: %d (pos=%.1f%%)  patient-grouped",
        len(y_dev), 100 * y_dev.mean(), len(y_test), 100 * y_test.mean(),
    )

    # SQLite-backed study so it resumes seamlessly if the machine restarts.
    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(
        study_name="resnet_hybrid_full",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        # Median pruner: kill trials whose intermediate fold mean is below
        # the median of completed trials at the same fold step. Saves time
        # on hopeless configs without rejecting promising-but-slow ones.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2,
        ),
        storage=storage,
        load_if_exists=True,
    )
    n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_done > 0:
        logger.info("Resuming study with %d completed trials, target=%d", n_done, args.n_trials)

    trials_dir = run_dir / "trials"
    trials_dir.mkdir(exist_ok=True)
    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev, folds,
                                epochs=args.epochs, trials_dir=trials_dir),
        n_trials=args.n_trials,
    )

    best_full = {
        **FIXED_PARAMS,
        **study.best_params,
        "f4": study.best_params["f3"],   # tied
    }
    with open(run_dir / "best_params.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_cv_auroc": study.best_value,
            "best_params": study.best_params,
            "fixed_params": FIXED_PARAMS,
            "full_params": best_full,
            "n_completed_trials": int(sum(1 for t in study.trials
                                          if t.state == optuna.trial.TrialState.COMPLETE)),
            "n_pruned_trials": int(sum(1 for t in study.trials
                                       if t.state == optuna.trial.TrialState.PRUNED)),
        }, f, indent=2, default=str)

    # Retrain best on full dev with stratified 90/10 early-stop split
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=args.seed,
    )
    X_tr, y_tr = augment_train(X_tr, y_tr, seed=args.seed)
    model = RepNetResNetHybridModel(**best_full, epochs=args.epochs)
    model.fit(X_tr, y_tr, X_es, y_es)
    torch.save(model.model.state_dict(), run_dir / "best_model.pt")

    # Persist the final retrain's training curve too -- otherwise you can't
    # reconstruct the loss/AUROC trajectory for the model whose weights you'll ship.
    with open(run_dir / "best_model_history.json", "w") as f:
        json.dump({
            "params": best_full,
            "epochs_run": len(model.history["train_loss"]),
            "train_loss": [float(x) for x in model.history["train_loss"]],
            "val_auroc": [float(x) for x in model.history["val_auroc"]],
        }, f, indent=2)

    proba = model.predict_proba(X_test)
    test_auroc = float(roc_auc_score(y_test, proba))
    thresh_j = youden_threshold(y_test, proba)

    # Structured holdout test metrics for downstream analysis without parsing log.
    with open(run_dir / "holdout_test_metrics.json", "w") as f:
        json.dump({
            "test_auroc": test_auroc,
            "threshold_youden": thresh_j,
            "n_test": int(len(y_test)),
            "n_test_pos": int((y_test == 1).sum()),
            "n_test_neg": int((y_test == 0).sum()),
            "report_05": classification_report(
                y_test, (proba >= 0.5).astype(int),
                target_names=["No PE", "PE"], output_dict=True, zero_division=0,
            ),
            "report_youden": classification_report(
                y_test, (proba >= thresh_j).astype(int),
                target_names=["No PE", "PE"], output_dict=True, zero_division=0,
            ),
            "probs": [float(p) for p in proba],
            "y_test": [int(v) for v in y_test],
        }, f, indent=2)

    # Pandas-friendly dump of every trial (params, value, state, datetimes,
    # intermediate fold values from trial.report). For quick analysis.
    try:
        df = study.trials_dataframe(
            attrs=("number", "value", "state", "datetime_start", "datetime_complete",
                   "duration", "params", "intermediate_values"),
        )
        df.to_csv(run_dir / "all_trials.csv", index=False)
        logger.info("Wrote all_trials.csv (%d rows)", len(df))
    except Exception as e:
        logger.warning("Could not write trials_dataframe: %s", e)

    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

    lines = [
        "=" * 66,
        "OPTIMIZATION COMPLETE - RepNetResNetHybrid",
        "=" * 66,
        f"Trials:         {n_complete} completed, {n_pruned} pruned, {len(study.trials)} total",
        f"Best trial:     #{study.best_trial.number}",
        f"Best CV AUROC:  {study.best_value:.4f}",
        f"Best lr:        {study.best_params['lr']:.6f}",
        f"Best dropout:   {study.best_params['dropout']:.4f}",
        f"Best wd:        {study.best_params['weight_decay']:.6f}",
        f"Best f2:        {study.best_params['f2']}",
        f"Best f3 (=f4):  {study.best_params['f3']}",
        f"Best batch:     {study.best_params['batch_size']}",
        "",
        "=" * 66,
        "HOLDOUT TEST",
        "=" * 66,
        f"  AUROC: {test_auroc:.4f}",
        "",
        f"  Threshold = 0.50:",
        classification_report(y_test, (proba >= 0.5).astype(int),
                              target_names=["No PE", "PE"], zero_division=0),
        f"  Threshold = {thresh_j:.3f} (Youden's J):",
        classification_report(y_test, (proba >= thresh_j).astype(int),
                              target_names=["No PE", "PE"], zero_division=0),
        "",
        "Top 10 trials:",
    ]
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    for t in completed[:10]:
        lines.append(
            f"  #{t.number:3d} | lr={t.params['lr']:.5f} drop={t.params['dropout']:.3f} "
            f"wd={t.params['weight_decay']:.5f} f2={t.params['f2']:>3d} f3={t.params['f3']:>3d} "
            f"bs={t.params['batch_size']:>2d} | AUROC={t.value:.4f}"
        )
    summary = "\n".join(lines)
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(summary)

    try:
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
        plot_contour(study, params=["lr", "dropout"]).write_html(
            str(run_dir / "contour_lr_dropout.html"))
        plot_contour(study, params=["lr", "weight_decay"]).write_html(
            str(run_dir / "contour_lr_wd.html"))
        plot_optimization_history(study).write_html(
            str(run_dir / "optimization_history.html"))
        plot_parallel_coordinate(study).write_html(
            str(run_dir / "parallel_coordinate.html"))
        plot_param_importances(study).write_html(
            str(run_dir / "param_importances.html"))
        plot_slice(study).write_html(
            str(run_dir / "slice_plot.html"))
        logger.info("Saved plots to %s", run_dir)
    except Exception as e:
        logger.warning("Could not generate plots: %s", e)

    print(f"\nResults saved to: {run_dir}/")


if __name__ == "__main__":
    main()
