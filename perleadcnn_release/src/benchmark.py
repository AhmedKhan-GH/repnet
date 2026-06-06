"""Benchmark PerLeadCNN training throughput on CPU, CUDA, or Apple-Silicon MPS.

Measures how fast one training epoch runs on the current device and projects
the time for a single split and the full 30-split run. Uses **synthetic data
by default**, so you can benchmark hardware without the (PHI) dataset; pass
``--real`` to time on the actual cohort instead.

Run (from the package root):
    python -m src.benchmark                  # auto device, synthetic data
    python -m src.benchmark --device cpu
    python -m src.benchmark --device mps     # Apple Silicon GPU
    python -m src.benchmark --device cuda --epochs 5
    python -m src.benchmark --real           # use the actual dataset

The reported numbers reuse the real training step (mixup + focal loss +
gradient accumulation) and, with augmentation on, the real DataLoader path,
so the per-epoch time reflects actual training cost.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import N_LEADS, SEQ_LEN_MODEL, load_dataset, train_val_test_split
from .model import PerLeadCNN, count_parameters
from .train import (ACCUM_STEPS, BATCH_SIZE, DROPOUT, FILTERS, FOCAL_GAMMA,
                    KERNELS, LABEL_SMOOTHING, LR, MIXUP_ALPHA, WEIGHT_DECAY,
                    ECGDataset, FocalLoss, device_name, resolve_device)

# Approx. number of training samples per split in the released setup
# (2178 records x 4/5 dev x 7/8 train). Used to scale measured per-epoch time.
REAL_TRAIN_N = 1524
# Typical number of epochs actually run per split (early stopping, patience 20).
TYPICAL_EPOCHS = 45


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _synthetic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_LEADS, SEQ_LEN_MODEL)).astype(np.float32)
    y = (rng.random(n) < 0.154).astype(np.int64)
    if y.sum() == 0:          # guarantee both classes for the val AUROC / loss
        y[0] = 1
    return X, y


def _run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, (xb, yb) in enumerate(loader):
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
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def run_benchmark(device=None, epochs=3, n_train=512, n_eval=256,
                  batch_size=BATCH_SIZE, augment=True, synthetic=True,
                  data_dir=None, seed=0) -> dict:
    """Time `epochs` training epochs + one eval pass; return a metrics dict."""
    dev = resolve_device(device) if not isinstance(device, torch.device) else device
    torch.manual_seed(seed)
    np.random.seed(seed)

    if synthetic:
        X, y = _synthetic(n_train + n_eval, seed)
    else:
        Xa, ya, groups = load_dataset(data_dir)
        tr, _va, te = train_val_test_split(0, ya, groups)
        X = np.concatenate([Xa[tr][:n_train], Xa[te][:n_eval]])
        y = np.concatenate([ya[tr][:n_train], ya[te][:n_eval]])
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_ev, y_ev = X[n_train:n_train + n_eval], y[n_train:n_train + n_eval]

    model = PerLeadCNN(filters=FILTERS, kernels=KERNELS, dropout=DROPOUT).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    loader = DataLoader(ECGDataset(X_tr, y_tr, augment=augment),
                        batch_size=batch_size, shuffle=True, num_workers=0)

    # Warm-up epoch (kernel compile / allocator) — not timed.
    _run_epoch(model, loader, optimizer, criterion, dev)
    _sync(dev)

    epoch_times = []
    for _ in range(epochs):
        t0 = time.time()
        _run_epoch(model, loader, optimizer, criterion, dev)
        _sync(dev)
        epoch_times.append(time.time() - t0)
    sec_epoch = float(np.mean(epoch_times))

    # Eval throughput
    model.eval()
    Xe = torch.tensor(X_ev, dtype=torch.float32).to(dev)
    with torch.no_grad():
        model(Xe[:batch_size]); _sync(dev)          # warm-up
        t0 = time.time()
        for i in range(0, len(Xe), batch_size):
            torch.softmax(model(Xe[i:i + batch_size]), dim=1)
        _sync(dev)
        eval_sec = time.time() - t0

    sec_epoch_real = sec_epoch * (REAL_TRAIN_N / n_train)
    return {
        "device": dev.type,
        "device_label": device_name(dev),
        "params": count_parameters(model),
        "augment": augment,
        "n_train": n_train,
        "batch_size": batch_size,
        "epochs_timed": epochs,
        "seconds_per_epoch": sec_epoch,
        "train_samples_per_sec": n_train / sec_epoch,
        "eval_samples_per_sec": n_eval / max(eval_sec, 1e-9),
        "seconds_per_epoch_real_size": sec_epoch_real,
        "est_seconds_per_split": sec_epoch_real * TYPICAL_EPOCHS,
        "est_seconds_full_run": sec_epoch_real * TYPICAL_EPOCHS * 30,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    ap.add_argument("--epochs", type=int, default=3, help="timed epochs (default 3)")
    ap.add_argument("--n-train", type=int, default=512, help="synthetic train size")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no-augment", action="store_true",
                    help="disable augmentation (pure model throughput)")
    ap.add_argument("--real", action="store_true",
                    help="use the real dataset instead of synthetic")
    args = ap.parse_args()

    print("Benchmarking PerLeadCNN training...")
    r = run_benchmark(device=args.device, epochs=args.epochs, n_train=args.n_train,
                      batch_size=args.batch_size, augment=not args.no_augment,
                      synthetic=not args.real)

    print("=" * 60)
    print(f"Device            : {r['device_label']}")
    print(f"Parameters        : {r['params']:,}")
    print(f"Data              : {'real' if args.real else 'synthetic'} | "
          f"augment={r['augment']} | batch={r['batch_size']}")
    print(f"Train throughput  : {r['train_samples_per_sec']:.0f} samples/s "
          f"({r['seconds_per_epoch']:.2f}s/epoch @ {r['n_train']} samples)")
    print(f"Eval throughput   : {r['eval_samples_per_sec']:.0f} samples/s")
    print("-" * 60)
    print(f"Estimated for the released 30-split run "
          f"(~{TYPICAL_EPOCHS} epochs/split, ~{REAL_TRAIN_N} train samples):")
    print(f"  per split       : ~{r['est_seconds_per_split']:.0f}s "
          f"({r['est_seconds_per_split']/60:.1f} min)")
    print(f"  full 30 splits  : ~{r['est_seconds_full_run']/60:.1f} min "
          f"({r['est_seconds_full_run']/3600:.1f} h)")
    print("=" * 60)
    print("Note: estimates scale the measured per-epoch time to the real train "
          "size; actual epochs vary with early stopping.")


if __name__ == "__main__":
    main()
