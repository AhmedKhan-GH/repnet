"""Explorer V2 — Rich ECG augmentation + LightGBM ensemble for PE prediction.

Key changes vs prior experiments:
  1. On-the-fly augmentation (7 ECG-specific transforms vs 2 pre-computed)
  2. Mixup regularization at batch level
  3. AdamW + cosine annealing LR schedule
  4. Label smoothing (0.05)
  5. WeightedRandomSampler for class-balanced mini-batches (no data discarded)
  6. LightGBM on 656 hand-crafted ECG features
  7. Neural + GBM probability-averaging ensemble
  8. Combined data loader ensuring waveform/feature alignment

Uses the full dataset (~2,100+ ECGs) — no undersampling.

Usage:
    python -m src.train_explorer_v2
    python -m src.train_explorer_v2 --n-seeds 20 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import decimate as scipy_decimate
from scipy.signal import resample as scipy_resample
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from src.models.resnet1d_3stage import ResNet1D3Stage
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

SD_LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
SD_LABEL_POS = "Preeclampsia or Other Hypertensive Disorders of Pregnancy"
SD_N_SAMPLES = 2500
SD_N_SAMPLES_500HZ = 5000

_NON_FEATURE_COLS = {
    "ECGTestID", "Pat_Obfus_MRN", "Preg_Obfus_ID", "PatLabel",
    "WeekDifference", "WeeksPreg", "PregECGTrimester",
}

TEST_SIZE = 0.20

# ============================================================
# Hyperparameters
# ============================================================

NEURAL_PARAMS = dict(
    stage_filters=(48, 96, 192),
    kernels=(7, 5, 3),
    dropout=0.12,
    n_classes=2,
)

TRAIN_CFG = dict(
    lr=2e-3,
    weight_decay=5e-3,
    batch_size=64,
    epochs=100,
    patience=20,
    label_smoothing=0.05,
    mixup_alpha=0.2,
    grad_clip=1.0,
    n_inits=3,
    val_n_splits=8,     # 87.5/12.5 within dev
)

AUG_CFG = dict(
    p_noise=0.5,
    noise_sigma_range=(0.01, 0.05),
    p_amp_scale=0.5,
    amp_scale_range=0.15,
    p_time_shift=0.3,
    max_time_shift=150,
    p_lead_drop=0.10,
    lead_drop_p=0.12,
    p_cutout=0.25,
    cutout_len_range=(50, 200),
    p_wander=0.2,
    wander_amp=0.2,
    wander_freq_range=(0.05, 0.5),
    p_resample=0.10,
    resample_rate=0.05,
)

LGBM_CFG = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=7,
    min_child_samples=20,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    class_weight="balanced",
    n_jobs=-1,
    objective="binary",
    metric="auc",
    verbosity=-1,
)


# ============================================================
# Combined data loader (waveforms + features, aligned)
# ============================================================

def load_combined(
    data_dir: str | Path = "data/seniordesign_upload",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load waveforms and hand-crafted features, perfectly aligned.

    Returns:
        X_wave      (N, 12, 2500) float32
        X_feat      (N, n_features) float32 (NaN-filled for LightGBM)
        y           (N,) int64
        patient_ids (N,)
        feat_cols   list[str]
    """
    data_dir = Path(data_dir)
    meta_path = next(
        (data_dir / name for name in ("metadata.csv", "metadata_balanced.csv")
         if (data_dir / name).exists()),
        None,
    )
    if meta_path is None:
        raise FileNotFoundError(f"No metadata CSV in {data_dir}")
    meta = pd.read_csv(meta_path)
    ekg_dir = data_dir / "ekg_data"

    available = {int(f.stem) for f in ekg_dir.iterdir() if f.suffix == ".csv"}
    meta = meta[meta["ECGTestID"].apply(lambda x: int(x) in available)].copy()
    meta = meta[meta["Pat_Obfus_MRN"].notna()].reset_index(drop=True)

    X_wave_list, keep_rows = [], []
    n_decimated, n_skip = 0, 0

    for row_idx, row in meta.iterrows():
        path = ekg_dir / f"{int(row['ECGTestID'])}.csv"
        try:
            df = pd.read_csv(path, skipinitialspace=True, usecols=SD_LEAD_ORDER)
            arr = df[SD_LEAD_ORDER].values.T.astype(np.float32)
            if arr.shape[0] != 12:
                n_skip += 1
                continue
            if arr.shape[1] == SD_N_SAMPLES_500HZ:
                arr = scipy_decimate(arr, q=2, axis=1).astype(np.float32)
                n_decimated += 1
            elif arr.shape[1] != SD_N_SAMPLES:
                n_skip += 1
                continue
            if arr.shape != (12, SD_N_SAMPLES):
                n_skip += 1
                continue
            if not np.isfinite(arr).all():
                n_skip += 1
                continue
            if (arr.std(axis=1) < 1e-4).any():
                n_skip += 1
                continue
            X_wave_list.append(arr)
            keep_rows.append(row_idx)
        except Exception:
            n_skip += 1

    if n_decimated:
        logger.info("Decimated %d recordings from 500→250 Hz", n_decimated)
    if n_skip:
        logger.info("Skipped %d recordings (quality)", n_skip)

    X_wave = np.stack(X_wave_list)
    meta_kept = meta.loc[keep_rows].reset_index(drop=True)

    feat_cols = [c for c in meta_kept.columns if c not in _NON_FEATURE_COLS]
    X_feat_df = meta_kept[feat_cols].apply(pd.to_numeric, errors="coerce")
    nan_rate = X_feat_df.isna().mean()
    feat_cols = nan_rate[nan_rate <= 0.30].index.tolist()
    X_feat = X_feat_df[feat_cols].values.astype(np.float32)

    y = (meta_kept["PatLabel"] == SD_LABEL_POS).astype(np.int64).values
    patient_ids = meta_kept["Pat_Obfus_MRN"].values

    logger.info(
        "Combined load: N=%d  pos=%d (%.1f%%)  features=%d  patients=%d",
        len(y), int(y.sum()), 100 * y.mean(), len(feat_cols),
        len(np.unique(patient_ids)),
    )
    return X_wave, X_feat, y, patient_ids, feat_cols


# ============================================================
# Preprocessing
# ============================================================

def preprocess_waveforms(X: np.ndarray) -> np.ndarray:
    X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X.copy())
    X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
    X, _ = ZScoreNormalization(per_lead=True).transform(X)
    return X


# ============================================================
# ECG Augmentation — on-the-fly, per-sample
# ============================================================

def _add_noise(x, sigma_range):
    sigma = np.random.uniform(*sigma_range)
    return x + np.random.normal(0, sigma, x.shape).astype(x.dtype)


def _scale_amplitude(x, scale_range):
    scales = np.random.uniform(
        1 - scale_range, 1 + scale_range, size=(x.shape[0], 1)
    ).astype(x.dtype)
    return x * scales


def _time_shift(x, max_shift):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=1)


def _drop_leads(x, p):
    mask = (np.random.random(x.shape[0]) > p).astype(x.dtype)
    return x * mask[:, np.newaxis]


def _cutout(x, len_range):
    x = x.copy()
    T = x.shape[1]
    cut_len = np.random.randint(len_range[0], len_range[1] + 1)
    start = np.random.randint(0, max(1, T - cut_len))
    x[:, start : start + cut_len] = 0
    return x


def _wander(x, max_amp, freq_range, fs=250):
    T = x.shape[1]
    t = np.linspace(0, T / fs, T, dtype=x.dtype)
    freq = np.random.uniform(*freq_range)
    phase = np.random.uniform(0, 2 * np.pi)
    amp = np.random.uniform(0, max_amp)
    w = amp * np.sin(2 * np.pi * freq * t + phase)
    return x + w[np.newaxis, :]


def _speed_perturb(x, max_rate):
    rate = 1 + np.random.uniform(-max_rate, max_rate)
    T = x.shape[1]
    new_T = max(T // 2, int(T * rate))
    x_rs = scipy_resample(x, new_T, axis=1)
    if new_T > T:
        start = (new_T - T) // 2
        x_rs = x_rs[:, start : start + T]
    elif new_T < T:
        pad = T - new_T
        pad_l = pad // 2
        x_rs = np.pad(x_rs, ((0, 0), (pad_l, pad - pad_l)), mode="edge")
    return x_rs.astype(x.dtype)


def augment_ecg(x: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    if cfg is None:
        cfg = AUG_CFG
    if np.random.random() < cfg["p_noise"]:
        x = _add_noise(x, cfg["noise_sigma_range"])
    if np.random.random() < cfg["p_amp_scale"]:
        x = _scale_amplitude(x, cfg["amp_scale_range"])
    if np.random.random() < cfg["p_time_shift"]:
        x = _time_shift(x, cfg["max_time_shift"])
    if np.random.random() < cfg["p_lead_drop"]:
        x = _drop_leads(x, cfg["lead_drop_p"])
    if np.random.random() < cfg["p_cutout"]:
        x = _cutout(x, cfg["cutout_len_range"])
    if np.random.random() < cfg["p_wander"]:
        x = _wander(x, cfg["wander_amp"], cfg["wander_freq_range"])
    if np.random.random() < cfg["p_resample"]:
        x = _speed_perturb(x, cfg["resample_rate"])
    return x


# ============================================================
# PyTorch Dataset with on-the-fly augmentation
# ============================================================

class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False, aug_cfg=None):
        self.X = X
        self.y = y
        self.augment = augment
        self.aug_cfg = aug_cfg or AUG_CFG

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = augment_ecg(x, self.aug_cfg)
        return (
            torch.from_numpy(x).float(),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ============================================================
# Neural model training
# ============================================================

def train_neural(
    X_train, y_train, X_val, y_val, seed,
    net_params=None, train_cfg=None,
):
    net_params = net_params or NEURAL_PARAMS
    train_cfg = train_cfg or TRAIN_CFG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ResNet1D3Stage(**net_params).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg["label_smoothing"],
    )

    class_counts = np.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train]
    sampler = WeightedRandomSampler(
        sample_weights.tolist(), len(y_train), replacement=True,
    )
    train_ds = ECGDataset(X_train, y_train, augment=True)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=gen,
    )

    Xv_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    history = {"train_loss": [], "val_auroc": [], "val_auprc": [], "lr": []}
    best_auroc, best_state, no_improve = 0.0, None, 0
    mixup_alpha = train_cfg["mixup_alpha"]

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=device)
                xb_mix = lam * xb + (1 - lam) * xb[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb_mix)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(
                    logits, yb[idx]
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg["grad_clip"]
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()

        avg_loss = epoch_loss / max(n_batches, 1)
        val_auroc = float(roc_auc_score(y_val, val_probs))
        val_auprc = float(average_precision_score(y_val, val_probs))
        cur_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_loss)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)
        history["lr"].append(cur_lr)

        marker = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or marker:
            print(
                f"    Ep {epoch+1:3d}/{train_cfg['epochs']} | "
                f"loss={avg_loss:.4f} | val_AUROC={val_auroc:.4f}{marker} | "
                f"lr={cur_lr:.2e}"
            )

        if no_improve >= train_cfg["patience"]:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history, best_auroc


@torch.no_grad()
def predict_neural(model, X, batch_size=128):
    device = next(model.parameters()).device
    model.eval()
    dl = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    parts = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        parts.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(parts)


# ============================================================
# LightGBM training
# ============================================================

def train_lgbm(X_train, y_train, X_val, y_val, seed, params=None):
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except ImportError:
        logger.warning("lightgbm not available — skipping")
        return None, 0.0

    params = {**(params or LGBM_CFG), "random_state": seed}
    clf = LGBMClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_auroc = float(roc_auc_score(y_val, val_probs))
    return clf, val_auroc


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    spec = 1 - fpr
    valid = spec >= 0.80
    sens_sp80 = float(tpr[valid].max()) if valid.any() else 0.0
    return {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "sens_sp80": sens_sp80,
    }


def find_best_ensemble_alpha(
    neural_probs, lgbm_probs, y_true, steps=21,
):
    best_alpha, best_auroc = 0.5, 0.0
    for a in np.linspace(0, 1, steps):
        ens = a * neural_probs + (1 - a) * lgbm_probs
        auroc = roc_auc_score(y_true, ens)
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = float(a)
    return best_alpha, best_auroc


# ============================================================
# Patient-grouped holdout split
# ============================================================

def grouped_holdout(X_wave, X_feat, y, groups, test_size, seed):
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=groups))

    leak = set(groups[dev_idx]) & set(groups[test_idx])
    if leak:
        raise RuntimeError(f"Patient leakage: {len(leak)} shared IDs")

    return (
        X_wave[dev_idx],
        X_wave[test_idx],
        X_feat[dev_idx],
        X_feat[test_idx],
        y[dev_idx],
        y[test_idx],
        groups[dev_idx],
        groups[test_idx],
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explorer V2: rich augmentation + LightGBM ensemble"
    )
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    seeds = args.seeds or list(range(42, 42 + args.n_seeds))
    tcfg = {**TRAIN_CFG, "epochs": args.epochs, "patience": args.patience}

    run_dir = Path("cv_results") / (
        f"explorer_v2_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "results.log", encoding="utf-8"),
        ],
    )

    # ---- Load & preprocess ----
    print("Loading combined data (waveforms + features)...")
    X_wave, X_feat, y, patient_ids, feat_cols = load_combined(args.data_dir)
    print(f"  N={len(y)}  pos={int(y.sum())} ({100*y.mean():.1f}%)  "
          f"features={len(feat_cols)}  patients={len(np.unique(patient_ids))}")

    print("Preprocessing waveforms (BWF + Notch + Z-score)...")
    X_wave = preprocess_waveforms(X_wave)

    config = {
        "neural_params": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in NEURAL_PARAMS.items()
        },
        "train_cfg": tcfg,
        "aug_cfg": AUG_CFG,
        "lgbm_cfg": {k: v for k, v in LGBM_CFG.items() if k != "n_jobs"},
        "seeds": seeds,
        "n_total": int(len(y)),
        "n_pos": int(y.sum()),
        "n_features": len(feat_cols),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Explorer V2 — {len(seeds)} seeds")
    print(f"  Neural: ResNet1D-3Stage + 7-transform aug + mixup + AdamW+cosine")
    print(f"  GBM:    LightGBM on {len(feat_cols)} features")
    print(f"  Ensemble: tuned probability average")
    print(f"  Output: {run_dir}")
    print(f"{'='*65}\n")

    all_results = []
    all_histories = {}

    for i, master_seed in enumerate(seeds):
        ss = np.random.SeedSequence(master_seed)
        split_seed, init_seed = (int(s) for s in ss.generate_state(2))
        print(
            f"\n{'─'*55}\n"
            f"  Seed {i+1}/{len(seeds)}  master={master_seed}  "
            f"split={split_seed}  init={init_seed}\n"
            f"{'─'*55}"
        )

        # ---- Split (waveform + features together) ----
        (
            Xw_dev, Xw_test,
            Xf_dev, Xf_test,
            y_dev, y_test,
            g_dev, g_test,
        ) = grouped_holdout(
            X_wave, X_feat, y, patient_ids,
            test_size=TEST_SIZE, seed=split_seed,
        )

        # Further split dev → train + val
        val_n_splits = tcfg.get("val_n_splits", 5)
        sgkf = StratifiedGroupKFold(
            n_splits=val_n_splits, shuffle=True, random_state=init_seed,
        )
        tr_idx, val_idx = next(
            sgkf.split(np.zeros(len(y_dev)), y_dev, groups=g_dev)
        )
        Xw_tr, Xw_val = Xw_dev[tr_idx], Xw_dev[val_idx]
        Xf_tr, Xf_val = Xf_dev[tr_idx], Xf_dev[val_idx]
        y_tr, y_val = y_dev[tr_idx], y_dev[val_idx]

        print(
            f"  train={len(y_tr)} (pos={int(y_tr.sum())})  "
            f"val={len(y_val)} (pos={int(y_val.sum())})  "
            f"test={len(y_test)} (pos={int(y_test.sum())})"
        )

        # ---- Train neural (multi-init ensemble) ----
        n_inits = tcfg.get("n_inits", 1)
        neural_probs_test_all = []
        neural_probs_val_all = []
        best_val_auroc_n = 0.0

        for init_i in range(n_inits):
            sub_seed = init_seed + init_i * 1000
            print(f"  [Neural {init_i+1}/{n_inits}] Training ResNet1D-3Stage...")
            model, history, val_auroc_n = train_neural(
                Xw_tr, y_tr, Xw_val, y_val, sub_seed,
                net_params=NEURAL_PARAMS, train_cfg=tcfg,
            )
            neural_probs_test_all.append(predict_neural(model, Xw_test))
            neural_probs_val_all.append(predict_neural(model, Xw_val))
            best_val_auroc_n = max(best_val_auroc_n, val_auroc_n)
            del model
            torch.cuda.empty_cache()

        neural_probs_test = np.mean(neural_probs_test_all, axis=0)
        neural_probs_val = np.mean(neural_probs_val_all, axis=0)
        val_auroc_n = best_val_auroc_n
        neural_test = evaluate(y_test, neural_probs_test)
        print(
            f"  [Neural] test AUROC={neural_test['auroc']:.4f}  "
            f"AUPRC={neural_test['auprc']:.4f}  "
            f"sens@sp80={neural_test['sens_sp80']:.4f}"
        )

        # ---- Train LightGBM ----
        print("  [LightGBM] Training on features...")
        lgbm, val_auroc_g = train_lgbm(
            Xf_tr, y_tr, Xf_val, y_val, init_seed,
        )

        result = {
            "seed": int(master_seed),
            "split_seed": int(split_seed),
            "init_seed": int(init_seed),
            "n_test": int(len(y_test)),
            "n_pos_test": int(y_test.sum()),
            "neural": neural_test,
            "neural_val_auroc": float(val_auroc_n),
        }

        if lgbm is not None:
            lgbm_probs_test = lgbm.predict_proba(Xf_test)[:, 1]
            lgbm_probs_val = lgbm.predict_proba(Xf_val)[:, 1]
            lgbm_test = evaluate(y_test, lgbm_probs_test)
            result["lgbm"] = lgbm_test
            result["lgbm_val_auroc"] = float(val_auroc_g)
            print(
                f"  [LightGBM] test AUROC={lgbm_test['auroc']:.4f}  "
                f"AUPRC={lgbm_test['auprc']:.4f}  "
                f"sens@sp80={lgbm_test['sens_sp80']:.4f}"
            )

            # ---- Ensemble (tuned alpha on val) ----
            best_alpha, _ = find_best_ensemble_alpha(
                neural_probs_val, lgbm_probs_val, y_val,
            )
            ens_probs = (
                best_alpha * neural_probs_test
                + (1 - best_alpha) * lgbm_probs_test
            )
            ens_test = evaluate(y_test, ens_probs)
            result["ensemble"] = ens_test
            result["ensemble_alpha"] = best_alpha
            print(
                f"  [Ensemble] alpha={best_alpha:.2f}  "
                f"test AUROC={ens_test['auroc']:.4f}  "
                f"AUPRC={ens_test['auprc']:.4f}  "
                f"sens@sp80={ens_test['sens_sp80']:.4f}"
            )

        all_results.append(result)
        all_histories[int(master_seed)] = history

        # Save incremental
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        with open(run_dir / "histories.json", "w", encoding="utf-8") as f:
            json.dump(all_histories, f, indent=2)

    # ============================================================
    # Aggregate & report
    # ============================================================
    def _stats(arr):
        n = len(arr)
        sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "sem": sem,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "ci95_lo": float(arr.mean() - 1.96 * sem),
            "ci95_hi": float(arr.mean() + 1.96 * sem),
        }

    neural_aurocs = np.array([r["neural"]["auroc"] for r in all_results])
    neural_auprcs = np.array([r["neural"]["auprc"] for r in all_results])

    has_lgbm = all(r.get("lgbm") for r in all_results)
    has_ens = all(r.get("ensemble") for r in all_results)

    summary_lines = [
        "",
        "=" * 65,
        f"  Explorer V2 — {len(seeds)} seeds",
        "=" * 65,
        "",
        "  Neural (ResNet1D-3Stage + rich aug + mixup + AdamW+cosine):",
        f"    AUROC: {_stats(neural_aurocs)}",
        f"    AUPRC: {_stats(neural_auprcs)}",
    ]

    summary_data = {
        "neural_auroc": _stats(neural_aurocs),
        "neural_auprc": _stats(neural_auprcs),
    }

    if has_lgbm:
        lgbm_aurocs = np.array([r["lgbm"]["auroc"] for r in all_results])
        lgbm_auprcs = np.array([r["lgbm"]["auprc"] for r in all_results])
        summary_lines += [
            "",
            "  LightGBM (656 features):",
            f"    AUROC: {_stats(lgbm_aurocs)}",
            f"    AUPRC: {_stats(lgbm_auprcs)}",
        ]
        summary_data["lgbm_auroc"] = _stats(lgbm_aurocs)
        summary_data["lgbm_auprc"] = _stats(lgbm_auprcs)

    if has_ens:
        ens_aurocs = np.array([r["ensemble"]["auroc"] for r in all_results])
        ens_auprcs = np.array([r["ensemble"]["auprc"] for r in all_results])
        ens_alphas = np.array([r["ensemble_alpha"] for r in all_results])
        summary_lines += [
            "",
            "  Ensemble (neural + LightGBM, tuned alpha):",
            f"    AUROC: {_stats(ens_aurocs)}",
            f"    AUPRC: {_stats(ens_auprcs)}",
            f"    Alpha: mean={ens_alphas.mean():.2f}  std={ens_alphas.std():.2f}",
        ]
        summary_data["ensemble_auroc"] = _stats(ens_aurocs)
        summary_data["ensemble_auprc"] = _stats(ens_auprcs)

    summary_lines += ["", "=" * 65]
    summary = "\n".join(summary_lines)
    print(summary)

    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # ---- Pretty console summary ----
    n_auroc = _stats(neural_aurocs)
    print(f"\n  Neural AUROC:   {n_auroc['mean']:.4f} ± {n_auroc['std']:.4f}  "
          f"95% CI [{n_auroc['ci95_lo']:.4f}, {n_auroc['ci95_hi']:.4f}]")
    if has_lgbm:
        g_auroc = _stats(lgbm_aurocs)
        print(f"  LightGBM AUROC: {g_auroc['mean']:.4f} ± {g_auroc['std']:.4f}  "
              f"95% CI [{g_auroc['ci95_lo']:.4f}, {g_auroc['ci95_hi']:.4f}]")
    if has_ens:
        e_auroc = _stats(ens_aurocs)
        print(f"  Ensemble AUROC: {e_auroc['mean']:.4f} ± {e_auroc['std']:.4f}  "
              f"95% CI [{e_auroc['ci95_lo']:.4f}, {e_auroc['ci95_hi']:.4f}]")

    # ---- Plotly artifacts ----
    try:
        import plotly.graph_objects as go

        # AUROC distribution comparison
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=neural_aurocs, name="Neural", opacity=0.7, nbinsx=15,
        ))
        if has_lgbm:
            fig.add_trace(go.Histogram(
                x=lgbm_aurocs, name="LightGBM", opacity=0.7, nbinsx=15,
            ))
        if has_ens:
            fig.add_trace(go.Histogram(
                x=ens_aurocs, name="Ensemble", opacity=0.7, nbinsx=15,
            ))
        fig.update_layout(
            barmode="overlay",
            title=f"Test AUROC distribution across {len(seeds)} seeds",
            xaxis_title="Test AUROC",
            yaxis_title="Count",
            template="plotly_white",
            width=900,
            height=500,
        )
        fig.write_html(str(run_dir / "auroc_distribution.html"))

        # Per-seed comparison
        seed_labels = [str(r["seed"]) for r in all_results]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seed_labels, y=neural_aurocs, name="Neural", opacity=0.8,
        ))
        if has_lgbm:
            fig.add_trace(go.Bar(
                x=seed_labels, y=lgbm_aurocs, name="LightGBM", opacity=0.8,
            ))
        if has_ens:
            fig.add_trace(go.Bar(
                x=seed_labels, y=ens_aurocs, name="Ensemble", opacity=0.8,
            ))
        fig.update_layout(
            barmode="group",
            title="Per-seed test AUROC",
            xaxis_title="Master Seed",
            yaxis_title="Test AUROC",
            template="plotly_white",
            width=1200,
            height=500,
        )
        fig.write_html(str(run_dir / "per_seed_auroc.html"))

        logger.info("Saved plotly artifacts to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save plotly: %s", e)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
