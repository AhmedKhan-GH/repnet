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

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                         gridspec_kw={'wspace': 0.08, 'left': 0.04, 'right': 0.88})

for ax, mat, title in zip(axes,
                           [C_norm, C_pe, C_diff],
                           [f'Normal (N={int((y==0).sum())})',
                            f'PreE (N={int((y==1).sum())})',
                            'Difference (PreE $-$ Normal)']):
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(12)); ax.set_xticklabels(LEADS, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(12)); ax.set_yticklabels(LEADS, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold')

axes[2].text(0.98, 0.02, f'$\\|\\Delta C\\|_F = {frob:.3f}$',
             transform=axes[2].transAxes, fontsize=11, fontweight='bold',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Pearson $r$')

fig.suptitle('Inter-lead correlation: Normal vs PreE', fontsize=12, fontweight='bold')
plt.savefig('frobenius_diff.png', dpi=200, bbox_inches='tight')
print('Saved frobenius_diff.png')
