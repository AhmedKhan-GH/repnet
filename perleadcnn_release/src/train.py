"""Train PerLeadCNN with the exact configuration behind the released results.

Runs 30 patient-grouped splits (StratifiedGroupKFold), trains one model per
split, and writes `summary.json`, `per_split.json`, `best_model.pt`, and
`median_model.pt` into `results/multisplit_<tag>/` — the same artifacts that
ship in `results/multisplit_dbb6f49/`.

Note on reproducibility: training is seeded per split, but exact bit-for-bit
reproduction of the released checkpoints is not guaranteed across different
hardware / BLAS / cuDNN versions. Retraining reproduces the *distribution*
(mean AUROC ~= 0.71 over 30 splits); the shipped checkpoints reproduce the
*exact* recorded metrics (see `evaluate.py`).

Run (from the package root):
    python -m src.train
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .data import FS, load_dataset, train_val_test_split
from .metrics import compute_metrics
from .model import PerLeadCNN, count_parameters


def resolve_device(name: str | None = None) -> torch.device:
    """Pick a device: explicit `name` / $REPNET_DEVICE, else cuda > mps > cpu."""
    name = name or os.environ.get("REPNET_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Human-readable device label (includes GPU model when available)."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"

# ---------------------------------------------------------------------------
# Hyperparameters — the released configuration ("stratified, focal-only")
# ---------------------------------------------------------------------------
FILTERS = (16, 32, 48)
KERNELS = (31, 21, 11)
DROPOUT = 0.15

LR = 1.2e-3
WEIGHT_DECAY = 5e-3
BATCH_SIZE = 64
ACCUM_STEPS = 2                # effective batch size 128
MAX_EPOCHS = 80
PATIENCE = 20
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
FOCAL_GAMMA = 1.0

N_SPLITS = 30
SEED = 42
TIME_BUDGET = float(os.environ.get("REPNET_TIME_BUDGET", "inf"))  # seconds/split

# 7 on-the-fly ECG augmentations (applied to the 250 Hz waveforms).
AUG_CFG = dict(
    p_noise=0.5, noise_sigma_range=(0.01, 0.05),
    p_amp_scale=0.5, amp_scale_range=0.15,
    p_time_shift=0.3, max_time_shift=150,
    p_lead_drop=0.10, lead_drop_p=0.12,
    p_cutout=0.25, cutout_len_range=(50, 200),
    p_wander=0.2, wander_amp=0.2, wander_freq_range=(0.05, 0.5),
    p_resample=0.10, resample_rate=0.05,
)


# ---------------------------------------------------------------------------
# Augmentation + dataset
# ---------------------------------------------------------------------------
def augment_ecg(x, cfg):
    """Stochastic augmentations for a single (12, T) waveform."""
    T = x.shape[1]
    if np.random.rand() < cfg["p_noise"]:
        sigma = np.random.uniform(*cfg["noise_sigma_range"])
        x = x + np.random.randn(*x.shape).astype(x.dtype) * sigma
    if np.random.rand() < cfg["p_amp_scale"]:
        scale = 1 + np.random.uniform(-cfg["amp_scale_range"],
                                      cfg["amp_scale_range"], (x.shape[0], 1))
        x = x * scale.astype(x.dtype)
    if np.random.rand() < cfg["p_time_shift"]:
        shift = np.random.randint(-cfg["max_time_shift"], cfg["max_time_shift"] + 1)
        x = np.roll(x, shift, axis=1)
    if np.random.rand() < cfg["p_lead_drop"]:
        mask = np.random.rand(x.shape[0]) > cfg["lead_drop_p"]
        x = x * mask[:, None].astype(x.dtype)
    if np.random.rand() < cfg["p_cutout"]:
        length = np.random.randint(*cfg["cutout_len_range"])
        start = np.random.randint(0, max(T - length, 1))
        x[:, start:start + length] = 0
    if np.random.rand() < cfg["p_wander"]:
        t = np.arange(T, dtype=np.float32)
        freq = np.random.uniform(*cfg["wander_freq_range"])
        amp = np.random.uniform(0, cfg["wander_amp"])
        x = x + (amp * np.sin(2 * np.pi * freq * t / FS))[None, :]
    if np.random.rand() < cfg["p_resample"]:
        from scipy.signal import resample as scipy_resample
        rate = 1 + np.random.uniform(-cfg["resample_rate"], cfg["resample_rate"])
        new_len = int(T * rate)
        x = scipy_resample(x, new_len, axis=1).astype(np.float32)
        if x.shape[1] > T:
            x = x[:, :T]
        elif x.shape[1] < T:
            x = np.concatenate([x, np.zeros((x.shape[0], T - x.shape[1]), x.dtype)], axis=1)
    return x


class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False, aug_cfg=None):
        self.X, self.y = X, y
        self.augment = augment
        self.aug_cfg = aug_cfg or AUG_CFG

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = augment_ecg(x, self.aug_cfg)
        return torch.from_numpy(x).float(), torch.tensor(self.y[idx], dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Single split: train + evaluate
# ---------------------------------------------------------------------------
def train_one_split(X_tr, y_tr, X_va, y_va, X_te, y_te, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PerLeadCNN(filters=FILTERS, kernels=KERNELS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    train_dl = DataLoader(ECGDataset(X_tr, y_tr, augment=True), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0,
                          pin_memory=torch.cuda.is_available())
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)

    best_val, best_state, no_improve, t0 = 0.0, None, 0, time.time()
    for _epoch in range(MAX_EPOCHS):
        if time.time() - t0 > TIME_BUDGET:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            if MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                idx = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[idx]
                logits = model(xb)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[idx])
            else:
                loss = criterion(model(xb), yb)
            (loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_dl):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(Xv), dim=1)[:, 1].cpu().numpy()
        val_auroc = roc_auc_score(y_va, val_probs)
        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = []
        Xt = torch.tensor(X_te, dtype=torch.float32)
        for i in range(0, len(Xt), BATCH_SIZE):
            probs.append(torch.softmax(model(Xt[i:i + BATCH_SIZE].to(device)),
                                       dim=1)[:, 1].cpu().numpy())
    metrics = compute_metrics(y_te, np.concatenate(probs))
    metrics["best_val_auroc"] = float(best_val)
    metrics["seconds"] = time.time() - t0
    return metrics, best_state


def _run_tag():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "retrain"


def main(n_splits: int = N_SPLITS, out_dir: str | None = None, write: bool = True):
    """Train PerLeadCNN over `n_splits` patient-grouped splits.

    Returns (summary, rows) — the aggregate dict and the per-split metrics —
    so callers (e.g. a notebook) can display them. With `write=True` the
    release artifacts (summary.json, per_split.json, best/median checkpoints)
    are saved into `out_dir` (default: results/multisplit_<git-tag>/).

    Defaults reproduce the released run, so `python -m src.train` is unchanged.
    """
    torch.set_float32_matmul_precision("high")
    device = resolve_device()

    print(f"Device: {device_name(device)}")
    print("Loading dataset...")
    X, y, groups = load_dataset()
    n_params = count_parameters(PerLeadCNN(filters=FILTERS, kernels=KERNELS))
    print(f"  {len(y)} recordings, {int(y.sum())} positive ({y.mean():.1%}), "
          f"{len(np.unique(groups))} patients")
    print(f"Model: PerLeadCNN filters={FILTERS} kernels={KERNELS} "
          f"params={n_params:,}")
    print(f"Running {n_splits} stratified patient-grouped splits...\n")

    rows, states = [], []
    t_start = time.time()
    for i in range(n_splits):
        tr, va, te = train_val_test_split(i, y, groups)
        m, state = train_one_split(X[tr], y[tr], X[va], y[va], X[te], y[te],
                                   device, seed=i * 7 + 1000 + 2)
        m = {"split": i, **m}
        rows.append(m)
        states.append(state)
        print(f"Split {i + 1:2d}/{n_splits} | AUROC={m['auroc']:.4f} "
              f"AUPRC={m['auprc']:.4f} | youden F1={m['youden_f1']:.3f} "
              f"| {m['seconds']:.0f}s")

    total_seconds = time.time() - t_start
    aurocs = np.array([r["auroc"] for r in rows])
    auprcs = np.array([r["auprc"] for r in rows])
    split_secs = np.array([r["seconds"] for r in rows])
    print(f"\nAUROC {aurocs.mean():.4f} +/- {aurocs.std():.4f} | "
          f"AUPRC {auprcs.mean():.4f} +/- {auprcs.std():.4f}")
    print(f"Training time: {total_seconds:.0f}s total "
          f"({split_secs.mean():.0f}s/split) on {device_name(device)}")

    summary = {"n_splits": n_splits, "num_params": n_params,
               "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
               "auprc_mean": float(auprcs.mean()), "auprc_std": float(auprcs.std()),
               "device": device_name(device),
               "total_seconds": float(total_seconds),
               "seconds_per_split_mean": float(split_secs.mean())}
    for thr in ("youden", "sens80"):
        for metric in ("sens", "spec", "prec", "acc", "f1", "npv", "threshold"):
            vals = [r[f"{thr}_{metric}"] for r in rows]
            summary[f"{thr}_{metric}_mean"] = float(np.mean(vals))
            summary[f"{thr}_{metric}_std"] = float(np.std(vals))

    if write:
        if out_dir is None:
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results", f"multisplit_{_run_tag()}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(out_dir, "per_split.json"), "w") as f:
            json.dump(rows, f, indent=2)
        best_i = int(aurocs.argmax())
        median_i = int(aurocs.argsort()[len(aurocs) // 2])
        if states[best_i] is not None:
            torch.save(states[best_i], os.path.join(out_dir, "best_model.pt"))
        if states[median_i] is not None:
            torch.save(states[median_i], os.path.join(out_dir, "median_model.pt"))
        print(f"Saved to {out_dir}/")

    return summary, rows


if __name__ == "__main__":
    main()
