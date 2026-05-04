"""Generate Frobenius norm difference heatmap."""
import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

DATA_DIR = '../../data/seniordesign_upload'
SEED = 42
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

X, y, pids = load_seniordesign(DATA_DIR, return_patient_ids=True)
flat_mask = (X.std(axis=2) < 1e-4).any(axis=1)
try:
    nan_mask = np.isnan(pids.astype(float))
except (ValueError, TypeError):
    nan_mask = np.array([str(p).strip() in ('', 'nan', 'None') for p in pids])
keep = ~flat_mask & ~nan_mask
X, y, pids = X[keep], y[keep], pids[keep]
X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
X, _ = ZScoreNormalization(per_lead=True).transform(X)

# Compute per-class correlation matrices
# Flatten each sample's 12 leads across time, then correlate leads
def lead_corr(X_subset):
    # X_subset: (N, 12, T) -> stack all time across samples -> (12, N*T)
    flat = X_subset.reshape(X_subset.shape[0], 12, -1)  # (N, 12, T)
    cat = np.concatenate(flat, axis=1)  # (12, N*T)
    return np.corrcoef(cat)  # (12, 12)

C_norm = lead_corr(X[y == 0])
C_pe = lead_corr(X[y == 1])
C_diff = C_pe - C_norm
frob = np.linalg.norm(C_diff, 'fro')

print(f'Normal samples: {(y==0).sum()}, PreE samples: {(y==1).sum()}')
print(f'||C_PreE - C_Normal||_F = {frob:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                         gridspec_kw={'wspace': 0.3, 'left': 0.05, 'right': 0.95})

im0 = axes[0].imshow(C_norm, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
axes[0].set_xticks(range(12)); axes[0].set_xticklabels(LEADS, fontsize=7, rotation=45, ha='right')
axes[0].set_yticks(range(12)); axes[0].set_yticklabels(LEADS, fontsize=7)
axes[0].set_title(f'Normal (N={int((y==0).sum())})', fontsize=10, fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(C_pe, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
axes[1].set_xticks(range(12)); axes[1].set_xticklabels(LEADS, fontsize=7, rotation=45, ha='right')
axes[1].set_yticks(range(12)); axes[1].set_yticklabels(LEADS, fontsize=7)
axes[1].set_title(f'PreE (N={int((y==1).sum())})', fontsize=10, fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

vmax_diff = max(abs(C_diff.min()), abs(C_diff.max()))
im2 = axes[2].imshow(C_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
axes[2].set_xticks(range(12)); axes[2].set_xticklabels(LEADS, fontsize=7, rotation=45, ha='right')
axes[2].set_yticks(range(12)); axes[2].set_yticklabels(LEADS, fontsize=7)
axes[2].set_title(f'Difference (PreE $-$ Normal)', fontsize=10, fontweight='bold')
axes[2].text(0.98, 0.02, f'$\\|\\Delta C\\|_F = {frob:.3f}$',
             transform=axes[2].transAxes, fontsize=11, fontweight='bold',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle('Inter-lead correlation: Normal vs PreE', fontsize=12, fontweight='bold')
plt.savefig('frobenius_diff.png', dpi=200, bbox_inches='tight')
print('Saved frobenius_diff.png')
