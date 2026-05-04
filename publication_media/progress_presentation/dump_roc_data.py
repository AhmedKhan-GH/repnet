"""Dump ROC curve, PR curve, histogram, and threshold data for the best model.
Uses MPS (Apple Silicon GPU) for fast inference.
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeper
from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

DATA_DIR = '../../data/seniordesign_upload'
MODEL_PATH = '../../optuna_results/optuna_crosslead_3stage_filters/2026-04-27_10-29-59/best_model.pt'
SEED = 42

device = torch.device('mps')
print(f'Device: {device}')

# --- Load & preprocess ---
X, y, pids = load_seniordesign(DATA_DIR, return_patient_ids=True)
print(f'Loaded {len(y)} samples')

flat_mask = (X.std(axis=2) < 1e-4).any(axis=1)
try:
    nan_mask = np.isnan(pids.astype(float))
except (ValueError, TypeError):
    nan_mask = np.array([str(p).strip() in ('', 'nan', 'None') for p in pids])
keep = ~flat_mask & ~nan_mask
X, y, pids = X[keep], y[keep], pids[keep]
print(f'After QC: {len(y)} samples')

X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
X, _ = ZScoreNormalization(per_lead=True).transform(X)
print('Preprocessing done')

X_dev, X_test, y_dev, y_test, _, _ = split_holdout_grouped(
    X, y, pids, test_size=0.20, seed=SEED,
)
print(f'Dev: {len(y_dev)}  Test: {len(y_test)}')

# --- Model ---
net = RepNetCrossLeadDeeper(
    stage_filters=(48, 96, 192), kernels=(7, 5, 3),
    dropout=0.0546, n_heads=4,
).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
net.eval()
print('Model loaded')


def infer(net, X_np, device, batch_size=64):
    Xt = torch.tensor(X_np, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt), batch_size=batch_size, num_workers=0)
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            logits = net(xb.to(device))
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


probs_dev = infer(net, X_dev, device)
probs_test = infer(net, X_test, device)
print('Inference done\n')

# --- ROC (test) ---
fpr, tpr, _ = roc_curve(y_test, probs_test)
auroc = roc_auc_score(y_test, probs_test)
idx = np.unique(np.linspace(0, len(fpr) - 1, 40).astype(int))
print('=== ROC_TEST ===')
print(f'AUROC={auroc:.4f}')
for i in idx:
    print(f'({fpr[i]:.4f},{tpr[i]:.4f})')

# --- ROC (dev) ---
fpr_d, tpr_d, _ = roc_curve(y_dev, probs_dev)
auroc_d = roc_auc_score(y_dev, probs_dev)
idx2 = np.unique(np.linspace(0, len(fpr_d) - 1, 40).astype(int))
print('\n=== ROC_DEV ===')
print(f'AUROC={auroc_d:.4f}')
for i in idx2:
    print(f'({fpr_d[i]:.4f},{tpr_d[i]:.4f})')

# --- PR (test) ---
prec, rec, _ = precision_recall_curve(y_test, probs_test)
auprc = average_precision_score(y_test, probs_test)
idx3 = np.unique(np.linspace(0, len(prec) - 1, 40).astype(int))
print('\n=== PR_TEST ===')
print(f'AUPRC={auprc:.4f}')
for i in idx3:
    print(f'({rec[i]:.4f},{prec[i]:.4f})')

# --- Histogram (test) ---
bins = np.linspace(0, 1, 21)
centers = 0.5 * (bins[:-1] + bins[1:])
h_norm, _ = np.histogram(probs_test[y_test == 0], bins=bins)
h_pe, _ = np.histogram(probs_test[y_test == 1], bins=bins)
print('\n=== HIST_TEST ===')
for c, hn, hp in zip(centers, h_norm, h_pe):
    print(f'{c:.3f} {int(hn)} {int(hp)}')

# --- Thresholds ---
fpr_t, tpr_t, thr_t = roc_curve(y_test, probs_test)
tau_youden = float(thr_t[np.argmax(tpr_t - fpr_t)])
print(f'\n=== THRESHOLDS ===')
print(f'Youden={tau_youden:.4f}')
for tau in [0.50, tau_youden]:
    pred = (probs_test >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
    print(f'tau={tau:.4f} TP={tp} FP={fp} FN={fn} TN={tn} '
          f'sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f} npv={npv:.3f} f1={f1:.3f}')
