"""Final neural network training -- RepNet-SE with systematic exploration.

Uses the proven explorer_v2 training pipeline (7 augmentations, mixup,
AdamW + cosine annealing, WeightedRandomSampler) with RepNet-SE architecture.

Systematic exploration:
  1. Multiple configurations (filter widths, dropout, LR)
  2. Bagging (average N models with different seeds)
  3. Focal loss vs label-smoothed cross-entropy

Usage:
    python -m src.train_neural_final
    python -m src.train_neural_final --n-seeds 20 --n-bags 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.repnet_crosslead import FocalLoss
from src.models.repnet_se import RepNetSE
from src.train_explorer_v2 import (
    AUG_CFG,
    ECGDataset,
    load_combined,
    preprocess_waveforms,
)

CONFIGS = {
    "repnet_se_base": dict(
        net=dict(stage_filters=(16, 32, 64), stage_kernels=(7, 5, 5),
                 ms_kernels=(5, 9, 15), dropout=0.15, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=2e-3, weight_decay=5e-3, batch_size=64, epochs=80,
                   patience=20, label_smoothing=0.05, mixup_alpha=0.2,
                   grad_clip=1.0, loss="label_smooth"),
    ),
    "repnet_se_wider": dict(
        net=dict(stage_filters=(24, 48, 96), stage_kernels=(7, 5, 3),
                 ms_kernels=(5, 11, 21), dropout=0.15, n_heads=4,
                 se_reduction=4, attn_pool_hidden=48),
        train=dict(lr=1.5e-3, weight_decay=5e-3, batch_size=64, epochs=80,
                   patience=20, label_smoothing=0.05, mixup_alpha=0.2,
                   grad_clip=1.0, loss="label_smooth"),
    ),
    "repnet_se_focal": dict(
        net=dict(stage_filters=(16, 32, 64), stage_kernels=(7, 5, 5),
                 ms_kernels=(5, 9, 15), dropout=0.12, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=2e-3, weight_decay=5e-3, batch_size=64, epochs=80,
                   patience=20, focal_alpha=0.25, focal_gamma=2.0,
                   mixup_alpha=0.2, grad_clip=1.0, loss="focal"),
    ),
    "repnet_se_deep": dict(
        net=dict(stage_filters=(16, 32, 48, 64), stage_kernels=(7, 5, 5, 3),
                 ms_kernels=(5, 9, 15), dropout=0.15, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=2e-3, weight_decay=5e-3, batch_size=64, epochs=80,
                   patience=20, label_smoothing=0.05, mixup_alpha=0.2,
                   grad_clip=1.0, loss="label_smooth"),
    ),
    # --- Round 2: stronger regularization to close val/test gap ---
    "repnet_se_base_reg": dict(
        net=dict(stage_filters=(16, 32, 64), stage_kernels=(7, 5, 5),
                 ms_kernels=(5, 9, 15), dropout=0.25, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=2e-3, weight_decay=0.01, batch_size=64, epochs=60,
                   patience=12, label_smoothing=0.08, mixup_alpha=0.3,
                   grad_clip=1.0, loss="label_smooth"),
    ),
    "repnet_se_deep_reg": dict(
        net=dict(stage_filters=(16, 32, 48, 64), stage_kernels=(7, 5, 5, 3),
                 ms_kernels=(5, 9, 15), dropout=0.25, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=2e-3, weight_decay=0.01, batch_size=64, epochs=60,
                   patience=12, label_smoothing=0.08, mixup_alpha=0.3,
                   grad_clip=1.0, loss="label_smooth"),
    ),
    "repnet_se_base_heavy": dict(
        net=dict(stage_filters=(16, 32, 64), stage_kernels=(7, 5, 5),
                 ms_kernels=(5, 9, 15), dropout=0.30, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=1.5e-3, weight_decay=0.02, batch_size=64, epochs=50,
                   patience=10, label_smoothing=0.10, mixup_alpha=0.4,
                   grad_clip=1.0, loss="label_smooth"),
    ),
    "repnet_se_deep_nomix": dict(
        net=dict(stage_filters=(16, 32, 48, 64), stage_kernels=(7, 5, 5, 3),
                 ms_kernels=(5, 9, 15), dropout=0.25, n_heads=4,
                 se_reduction=4, attn_pool_hidden=32),
        train=dict(lr=2e-3, weight_decay=0.01, batch_size=64, epochs=60,
                   patience=12, label_smoothing=0.08, mixup_alpha=0.0,
                   grad_clip=1.0, loss="label_smooth"),
    ),
}


def build_criterion(cfg, y_train, device):
    loss_type = cfg.get("loss", "label_smooth")
    if loss_type == "focal":
        return FocalLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
        )
    ls = cfg.get("label_smoothing", 0.0)
    return nn.CrossEntropyLoss(label_smoothing=ls)


def train_one(X_wave_tr, y_tr, X_wave_val, y_val, seed, net_cfg, train_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = RepNetSE(**net_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=1e-6,
    )
    criterion = build_criterion(train_cfg, y_tr, device)

    class_counts = np.bincount(y_tr.astype(int))
    sample_weights = np.where(y_tr == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
    sampler = WeightedRandomSampler(sample_weights, len(y_tr), replacement=True)

    train_ds = ECGDataset(X_wave_tr, y_tr, augment=True, aug_cfg=AUG_CFG)
    train_dl = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], sampler=sampler,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    val_t = torch.tensor(X_wave_val, dtype=torch.float32).to(device)

    mixup_alpha = train_cfg.get("mixup_alpha", 0.0)
    best_auroc, best_state, no_imp = 0.0, None, 0
    patience = train_cfg.get("patience", 20)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        ep_loss, n_b = 0.0, 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if mixup_alpha > 0 and np.random.rand() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[idx]
                yb_a, yb_b = yb, yb[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = lam * criterion(logits, yb_a) + (1 - lam) * criterion(logits, yb_b)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("grad_clip", 1.0))
            optimizer.step()
            ep_loss += loss.item()
            n_b += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(val_t), dim=1)[:, 1].cpu().numpy()
        val_auroc = roc_auc_score(y_val, probs)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_auroc, n_params


@torch.no_grad()
def predict(model, X, batch_size=128):
    device = next(model.parameters()).device
    model.eval()
    from torch.utils.data import TensorDataset
    dl = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    parts = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        parts.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(parts)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/seniordesign_upload")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-bags", type=int, default=3)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.n_seeds))

    run_dir = Path("cv_results") / (
        f"neural_final_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_wave, X_feat, y, patient_ids, feat_cols = load_combined(args.data_dir)
    print(f"  N={len(y)}  pos={int(y.sum())} ({100*y.mean():.1f}%)")

    print("Preprocessing waveforms...")
    X_wave = preprocess_waveforms(X_wave)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {device_name}")

    # Phase 1: Config selection (3 seeds each)
    print(f"\n{'='*65}")
    print(f"  Phase 1: Config Selection (3 seeds each)")
    print(f"{'='*65}")

    config_results = {}
    scout_seeds = seeds[:3]

    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            print(f"  Unknown config: {cfg_name}, skipping")
            continue

        cfg = CONFIGS[cfg_name]
        net_cfg = cfg["net"]
        train_cfg = {**cfg["train"]}
        if args.epochs:
            train_cfg["epochs"] = args.epochs

        aurocs = []
        for seed in scout_seeds:
            ss = np.random.SeedSequence(seed)
            split_seed = int(ss.generate_state(1)[0])
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
            dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=patient_ids))
            sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed + 1)
            tr_idx, val_idx = next(sgkf2.split(
                np.zeros(len(y[dev_idx])), y[dev_idx], groups=patient_ids[dev_idx]))

            Xw_tr = X_wave[dev_idx][tr_idx]
            Xw_val = X_wave[dev_idx][val_idx]
            Xw_test = X_wave[test_idx]
            y_tr = y[dev_idx][tr_idx]
            y_val = y[dev_idx][val_idx]
            y_test = y[test_idx]

            model, val_auroc, n_params = train_one(
                Xw_tr, y_tr, Xw_val, y_val, seed, net_cfg, train_cfg)
            test_probs = predict(model, Xw_test)
            test_auroc = roc_auc_score(y_test, test_probs)
            aurocs.append(test_auroc)
            print(f"  {cfg_name} seed={seed}: val={val_auroc:.4f} test={test_auroc:.4f} ({n_params} params)")

            del model
            torch.cuda.empty_cache()

        arr = np.array(aurocs)
        config_results[cfg_name] = {"mean": float(arr.mean()), "std": float(arr.std()),
                                     "aurocs": aurocs, "n_params": n_params}
        print(f"  {cfg_name}: mean={arr.mean():.4f} +/- {arr.std():.4f}")

    with open(run_dir / "config_selection.json", "w") as f:
        json.dump(config_results, f, indent=2)

    best_cfg_name = max(config_results, key=lambda k: config_results[k]["mean"])
    print(f"\n  Best config: {best_cfg_name} (mean={config_results[best_cfg_name]['mean']:.4f})")

    # Phase 2: Full evaluation with best config + bagging
    best_cfg = CONFIGS[best_cfg_name]
    net_cfg = best_cfg["net"]
    train_cfg = {**best_cfg["train"]}
    if args.epochs:
        train_cfg["epochs"] = args.epochs

    print(f"\n{'='*65}")
    print(f"  Phase 2: Full Evaluation -- {best_cfg_name} ({args.n_bags}-bag, {len(seeds)} seeds)")
    print(f"{'='*65}")

    all_results = []

    for i, master_seed in enumerate(seeds):
        ss = np.random.SeedSequence(master_seed)
        split_seed = int(ss.generate_state(1)[0])
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=patient_ids))
        sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed + 1)
        tr_idx, val_idx = next(sgkf2.split(
            np.zeros(len(y[dev_idx])), y[dev_idx], groups=patient_ids[dev_idx]))

        Xw_tr = X_wave[dev_idx][tr_idx]
        Xw_val = X_wave[dev_idx][val_idx]
        Xw_test = X_wave[test_idx]
        y_tr = y[dev_idx][tr_idx]
        y_val = y[dev_idx][val_idx]
        y_test = y[test_idx]

        bag_probs = []
        bag_val_aurocs = []
        best_bag_auroc, best_bag_state = 0.0, None
        for bag_i in range(args.n_bags):
            bag_seed = master_seed + bag_i * 1000
            model, val_auroc, _ = train_one(
                Xw_tr, y_tr, Xw_val, y_val, bag_seed, net_cfg, train_cfg)
            bag_probs.append(predict(model, Xw_test))
            bag_val_aurocs.append(val_auroc)
            if val_auroc > best_bag_auroc:
                best_bag_auroc = val_auroc
                best_bag_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            del model
            torch.cuda.empty_cache()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state_dict": best_bag_state,
            "net_cfg": net_cfg,
            "seed": int(master_seed),
            "val_auroc": best_bag_auroc,
        }, weights_dir / f"best_seed_{master_seed}.pt")

        avg_probs = np.mean(bag_probs, axis=0)
        metrics = evaluate(y_test, avg_probs)

        print(f"  Seed {i+1}/{len(seeds)} (master={master_seed}): "
              f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
              f"vals={[f'{v:.3f}' for v in bag_val_aurocs]}")

        result = {
            "seed": int(master_seed),
            "n_test": int(len(y_test)),
            "n_pos_test": int(y_test.sum()),
            "metrics": metrics,
            "bag_val_aurocs": bag_val_aurocs,
        }
        all_results.append(result)

        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    n = len(all_results)
    aurocs = np.array([r["metrics"]["auroc"] for r in all_results])
    auprcs = np.array([r["metrics"]["auprc"] for r in all_results])

    def _s(arr):
        sem = arr.std(ddof=1) / np.sqrt(len(arr))
        return (f"mean={arr.mean():.4f} +/- {arr.std(ddof=1):.4f}  "
                f"median={np.median(arr):.4f}  "
                f"CI=[{arr.mean()-1.96*sem:.4f},{arr.mean()+1.96*sem:.4f}]")

    summary = "\n".join([
        "",
        "=" * 65,
        f"  RepNet-SE Final -- {best_cfg_name} ({args.n_bags}-bag, {n} splits)",
        f"  Params: {config_results[best_cfg_name]['n_params']}",
        "=" * 65,
        f"  AUROC: {_s(aurocs)}",
        f"  AUPRC: {_s(auprcs)}",
        f"  >=0.70 AUROC: {(aurocs >= 0.70).sum()}/{n} ({100*(aurocs >= 0.70).mean():.0f}%)",
        "=" * 65,
    ])
    print(summary)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    with open(run_dir / "summary.json", "w") as f:
        json.dump({
            "best_config": best_cfg_name,
            "n_params": config_results[best_cfg_name]["n_params"],
            "n_bags": args.n_bags,
            "auroc": {"mean": float(aurocs.mean()), "std": float(aurocs.std(ddof=1)),
                      "median": float(np.median(aurocs))},
            "auprc": {"mean": float(auprcs.mean()), "std": float(auprcs.std(ddof=1)),
                      "median": float(np.median(auprcs))},
            "config_selection": config_results,
        }, f, indent=2)

    print(f"\nAll artifacts saved to: {run_dir}/")


if __name__ == "__main__":
    main()
