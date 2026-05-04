"""Generate attention heatmap, lead importance, and saliency/Grad-CAM figures.
Uses MPS for inference. Outputs PDFs for LaTeX inclusion.
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeper
from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

DATA_DIR = '../../data/seniordesign_upload'
MODEL_PATH = '../../optuna_results/optuna_crosslead_3stage_filters/2026-04-27_10-29-59/best_model.pt'
SEED = 42
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

device = torch.device('mps')

# --- Colors matching presentation ---
AGGIE_BLUE = '#022851'
AGGIE_GOLD = '#FFBF00'
COOL_GRAY = '#4D4F53'

# --- Load data ---
print('Loading data...')
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
X_dev, X_test, y_dev, y_test, _, _ = split_holdout_grouped(X, y, pids, test_size=0.20, seed=SEED)
print(f'Test: {len(y_test)} samples ({int(y_test.sum())} PreE)')

# --- Model ---
net = RepNetCrossLeadDeeper(stage_filters=(48, 96, 192), kernels=(7, 5, 3),
                            dropout=0.0546, n_heads=4).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
net.eval()
print('Model loaded')

# --- Inference + probabilities ---
def infer(net, X_np):
    Xt = torch.tensor(X_np, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt), batch_size=64, num_workers=0)
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            out.append(torch.softmax(net(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)

probs_test = infer(net, X_test)
print('Inference done')

# ================================================================
# FIGURE 1: Attention heatmaps + lead importance
# ================================================================
print('Computing attention maps...')
attn_storage = {f'stage{i+1}': [] for i in range(3)}

def make_hook(name):
    def hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attn_storage[name].append(output[1].detach().cpu().numpy())
    return hook

handles = []
for i, stage in enumerate(net.stages):
    handles.append(stage['attn'].attn.register_forward_hook(make_hook(f'stage{i+1}')))

with torch.no_grad():
    Xt = torch.tensor(X_test, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt), batch_size=64)
    for (xb,) in dl:
        net(xb.to(device))

for h in handles:
    h.remove()

attn_by_stage = {k: np.concatenate(v, axis=0) for k, v in attn_storage.items()}

# Compute lead importance: mean attention received (column-mean) per stage per class
lead_importance_norm = {}
lead_importance_pe = {}
for k in ['stage1', 'stage2', 'stage3']:
    A = attn_by_stage[k]
    lead_importance_norm[k] = A[y_test == 0].mean(axis=(0, 1))  # mean across samples and heads -> (12,12)
    lead_importance_pe[k] = A[y_test == 1].mean(axis=(0, 1))

fig = plt.figure(figsize=(11, 5.5))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.35, hspace=0.45,
                       left=0.06, right=0.95, top=0.90, bottom=0.08)

# Row 1: Normal attention heatmaps (stages 1-3)
for col, stage_name in enumerate(['stage1', 'stage2', 'stage3']):
    ax = fig.add_subplot(gs[0, col])
    A = attn_by_stage[stage_name][y_test == 0].mean(axis=0)  # (n_heads, 12, 12) -> mean over heads
    if A.ndim == 3:
        A = A.mean(axis=0)
    im = ax.imshow(A, cmap='Blues', aspect='equal')
    ax.set_xticks(range(12))
    ax.set_xticklabels(LEADS, fontsize=4, rotation=45, ha='right')
    ax.set_yticks(range(12))
    ax.set_yticklabels(LEADS, fontsize=4)
    ax.set_title(f'Stage {col+1} — Normal', fontsize=7, fontweight='bold')
    if col == 0:
        ax.set_ylabel('Query lead', fontsize=6)

# Row 2: PE attention heatmaps (stages 1-3)
for col, stage_name in enumerate(['stage1', 'stage2', 'stage3']):
    ax = fig.add_subplot(gs[1, col])
    A = attn_by_stage[stage_name][y_test == 1].mean(axis=0)
    if A.ndim == 3:
        A = A.mean(axis=0)
    im = ax.imshow(A, cmap='Reds', aspect='equal')
    ax.set_xticks(range(12))
    ax.set_xticklabels(LEADS, fontsize=4, rotation=45, ha='right')
    ax.set_yticks(range(12))
    ax.set_yticklabels(LEADS, fontsize=4)
    ax.set_title(f'Stage {col+1} — PreE', fontsize=7, fontweight='bold')
    if col == 0:
        ax.set_ylabel('Query lead', fontsize=6)

# Right column: lead importance bar chart (stage 3 difference)
ax_bar = fig.add_subplot(gs[:, 3])
A_norm_s3 = attn_by_stage['stage3'][y_test == 0].mean(axis=0)
A_pe_s3 = attn_by_stage['stage3'][y_test == 1].mean(axis=0)
if A_norm_s3.ndim == 3:
    A_norm_s3 = A_norm_s3.mean(axis=0)
    A_pe_s3 = A_pe_s3.mean(axis=0)

# Attention received = column mean (how much each lead is attended to)
importance_norm = A_norm_s3.mean(axis=0)
importance_pe = A_pe_s3.mean(axis=0)
diff = importance_pe - importance_norm

order = np.argsort(diff)
colors = [AGGIE_GOLD if d > 0 else AGGIE_BLUE for d in diff[order]]
ax_bar.barh(range(12), diff[order], color=colors, edgecolor='white', linewidth=0.5)
ax_bar.set_yticks(range(12))
ax_bar.set_yticklabels([LEADS[i] for i in order], fontsize=6)
ax_bar.set_xlabel('$\\Delta$ attention (PreE $-$ Normal)', fontsize=7)
ax_bar.set_title('Stage 3: Lead importance\n(attention received)', fontsize=7, fontweight='bold')
ax_bar.axvline(0, color='black', linewidth=0.5)
ax_bar.tick_params(labelsize=5)

plt.savefig('fig_attention.pdf', dpi=300, bbox_inches='tight')
print('Saved fig_attention.pdf')
plt.close()

# ================================================================
# FIGURE 2: Saliency (IG) + Grad-CAM for 2 example cases
# ================================================================
print('Computing saliency...')

def integrated_gradients(net, x_np, target_class=1, n_steps=32, smooth_sigma=5.0):
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    base = torch.zeros_like(x)
    alphas = torch.linspace(0.5/n_steps, 1.0-0.5/n_steps, n_steps, device=device).view(-1, 1, 1)
    interp = base.unsqueeze(0) + alphas * (x - base).unsqueeze(0)
    interp.requires_grad_(True)
    net.zero_grad()
    logits = net(interp)
    grads = torch.autograd.grad(logits[:, target_class].sum(), interp)[0]
    avg_grad = grads.mean(dim=0)
    attr = ((x - base) * avg_grad).cpu().numpy()
    if smooth_sigma > 0:
        attr = gaussian_filter1d(attr, sigma=smooth_sigma, axis=1)
    return attr

def gradcam_per_lead(net, x_np, target_class=1, smooth_sigma=2.0):
    x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
    activations = {}
    def fwd_hook(_m, _i, o):
        activations['feat'] = o
        o.retain_grad()
    h = net.stages[-1]['conv'].register_forward_hook(fwd_hook)
    net.zero_grad()
    logits = net(x)
    logits[0, target_class].backward()
    h.remove()
    feat = activations['feat']
    grad = feat.grad
    weights = grad.mean(dim=(0, 3))
    cam = (weights.unsqueeze(-1) * feat[0]).sum(dim=1)
    T = x_np.shape[1]
    cam_full = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(12, T),
                             mode='bilinear', align_corners=False)
    cam_full = cam_full.squeeze().detach().cpu().numpy()
    if smooth_sigma > 0:
        cam_full = gaussian_filter1d(cam_full, sigma=smooth_sigma, axis=1)
    return cam_full

# Pick confident TP
pred = (probs_test >= 0.5).astype(int)
tp = np.where((y_test == 1) & (pred == 1))[0]
confident_tp = tp[np.argmax(probs_test[tp])]
idx = confident_tp
print(f'  Confident TP (idx={idx}, P(PreE)={probs_test[idx]:.3f})')

ig_attr = integrated_gradients(net, X_test[idx])
gc_attr = gradcam_per_lead(net, X_test[idx])

# Combine IG and Grad-CAM by averaging their normalized versions
ig_norm = ig_attr / (np.abs(ig_attr).max() or 1.0)
gc_norm = gc_attr / (np.abs(gc_attr).max() or 1.0)
attr = 0.5 * (ig_norm + gc_norm)

# Log-scale the attribution magnitude to prevent dark areas from washing out
# sign(attr) * log(1 + |attr|/eps) / log(1 + 1/eps)
def log_scale_attr(a, eps=0.02):
    sign = np.sign(a)
    mag = np.abs(a)
    log_mag = np.log1p(mag / eps) / np.log1p(1.0 / eps)
    return sign * log_mag

attr_log = log_scale_attr(attr)

# Plot 12 leads split into two figures of 6
ecg = X_test[idx]
fs = 250.0
t = np.arange(ecg.shape[1]) / fs
prob = probs_test[idx]

cmax = 1.0
norm = TwoSlopeNorm(vmin=-cmax, vcenter=0, vmax=cmax)

lead_strength = np.abs(attr).sum(axis=1)
lead_order = np.argsort(-lead_strength)

for page, (start, end, suffix) in enumerate([(0, 6, '1'), (6, 12, '2')]):
    page_leads = lead_order[start:end]
    fig, axes = plt.subplots(6, 1, figsize=(11, 5.0),
                             gridspec_kw={'hspace': 0.15, 'top': 0.91, 'bottom': 0.08,
                                          'left': 0.06, 'right': 0.92})

    for row, L in enumerate(page_leads):
        ax = axes[row]
        a = attr_log[L]
        ax.imshow(a[np.newaxis, :], aspect='auto', cmap='RdBu_r',
                  norm=norm, extent=[0, 10, -1.5, 1.5], alpha=0.7, interpolation='bilinear')
        ax.plot(t, ecg[L], 'k-', linewidth=0.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(-2.0, 2.0)
        ax.set_yticks([])
        ax.set_ylabel(LEADS[L], fontsize=7, fontweight='bold', rotation=0, labelpad=14, va='center')
        if row < 5:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.tick_params(labelsize=5)

    rank_label = f'Leads ranked {start+1}–{end} by attribution' if page == 0 else f'Leads ranked {start+1}–{end} by attribution'
    fig.suptitle(f'Saliency — Confident PreE (P(PreE)={prob:.3f})  |  '
                 f'{rank_label}  |  '
                 f'Red → PreE, Blue → Normal',
                 fontsize=8, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.015, pad=0.015)
    cbar.set_label('Log-scaled attribution', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    plt.savefig(f'fig_saliency_{suffix}.pdf', dpi=300, bbox_inches='tight')
    print(f'Saved fig_saliency_{suffix}.pdf')
    plt.close()

print('Done.')
