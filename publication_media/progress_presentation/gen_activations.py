"""Generate layer-by-layer activation heatmaps for the confident TP case.
Shows how the ECG signal propagates through conv stages → fusion → GAP → vote.
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeper
from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization

DATA_DIR = '../../data/seniordesign_upload'
MODEL_PATH = '../../optuna_results/optuna_crosslead_3stage_filters/2026-04-27_10-29-59/best_model.pt'
SEED = 42
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

device = torch.device('mps')

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
_, X_test, _, y_test, _, _ = split_holdout_grouped(X, y, pids, test_size=0.20, seed=SEED)
print(f'Test: {len(y_test)} samples')

# --- Model ---
net = RepNetCrossLeadDeeper(stage_filters=(48, 96, 192), kernels=(7, 5, 3),
                            dropout=0.0546, n_heads=4).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
net.eval()

# --- Get probs to find confident TP ---
def infer(net, X_np):
    Xt = torch.tensor(X_np, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt), batch_size=64, num_workers=0)
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            out.append(torch.softmax(net(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)

probs_test = infer(net, X_test)
pred = (probs_test >= 0.5).astype(int)
tp = np.where((y_test == 1) & (pred == 1))[0]
idx = tp[np.argmax(probs_test[tp])]
print(f'Confident TP: idx={idx}, P(PreE)={probs_test[idx]:.3f}')

# --- Hook activations at every stage ---
acts = {}

def cap(name):
    def hook(_m, _i, output):
        if isinstance(output, tuple):
            output = output[0]
        acts[name] = output.detach().cpu()
    return hook

handles = [
    net.stages[0]['conv'].register_forward_hook(cap('s1_conv')),
    net.stages[0]['attn'].register_forward_hook(cap('s1_attn')),
    net.stages[1]['conv'].register_forward_hook(cap('s2_conv')),
    net.stages[1]['attn'].register_forward_hook(cap('s2_attn')),
    net.stages[2]['conv'].register_forward_hook(cap('s3_conv')),
    net.stages[2]['attn'].register_forward_hook(cap('s3_attn')),
    net.fuse.register_forward_hook(cap('fuse')),
    net.gap.register_forward_hook(cap('gap')),
    net.fc.register_forward_hook(cap('fc')),
]

with torch.no_grad():
    x = torch.tensor(X_test[idx:idx+1], dtype=torch.float32, device=device)
    logits = net(x)
    probs = torch.softmax(logits, dim=1)

for h in handles:
    h.remove()

print(f'Logits: {logits.cpu().numpy().ravel()}')
print(f'Softmax: {probs.cpu().numpy().ravel()} -> P(PreE) = {probs[0,1].item():.4f}')

# --- Figures: Per-lead activation through stages, split into 2 pages ---
TRACK_LEAD = 1
ecg_lead = X_test[idx, TRACK_LEAD, :]
fs = 250.0

pages = [
    (1, [('Stage 1 PerLeadConv (48 ch)', 's1_conv'),
         ('Stage 1 CrossLeadAttn', 's1_attn')]),
    (2, [('Stage 2 PerLeadConv (96 ch)', 's2_conv'),
         ('Stage 2 CrossLeadAttn', 's2_attn')]),
    (3, [('Stage 3 PerLeadConv (192 ch)', 's3_conv'),
         ('Stage 3 CrossLeadAttn', 's3_attn')]),
]

for stage_num, stages_list in pages:
    n_rows = 1 + len(stages_list)
    fig = plt.figure(figsize=(11, 5.5))
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=[0.7, 1.0, 1.0],
                           hspace=0.45, top=0.90, bottom=0.08, left=0.08, right=0.95)

    # Input ECG row
    ax0 = fig.add_subplot(gs[0])
    t_in = np.arange(2500) / fs
    ax0.plot(t_in, ecg_lead, 'k-', linewidth=0.6)
    ax0.set_xlim(0, 10)
    ax0.set_ylabel('mV', fontsize=7)
    ax0.set_title(f'Input ECG — Lead {LEADS[TRACK_LEAD]} (2500 samples @ 250 Hz)',
                  fontsize=8, fontweight='bold')
    ax0.tick_params(labelsize=6)
    ax0.set_xticks([])

    # Activation rows
    for r, (name, key) in enumerate(stages_list):
        ax = fig.add_subplot(gs[1 + r])
        a = acts[key][0, TRACK_LEAD].numpy()
        C, T_stage = a.shape
        cmax = float(np.abs(a).max()) or 1.0
        im = ax.imshow(a, aspect='auto', cmap='RdBu_r', vmin=-cmax, vmax=cmax,
                       extent=[0, 10, C-0.5, -0.5], interpolation='nearest')
        ax.set_xlim(0, 10)
        ax.set_ylabel('Channel', fontsize=7)
        ax.set_title(f'{name} — Lead {LEADS[TRACK_LEAD]} — ({C} channels × {T_stage} timesteps)',
                     fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        if r < len(stages_list) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (s)', fontsize=8)

    fig.suptitle(f'Activation Propagation — Stage {stage_num}  |  '
                 f'Confident PreE (P(PreE)={probs_test[idx]:.3f})',
                 fontsize=10, fontweight='bold')

    plt.savefig(f'fig_activations_{stage_num}.pdf', dpi=300, bbox_inches='tight')
    print(f'Saved fig_activations_{stage_num}.pdf')
    plt.close()

# --- Figure 2: Fusion → GAP → Vote ---
fuse_out = acts['fuse'][0].numpy()   # (192, T)
gap_out = acts['gap'][0, :, 0].numpy()  # (192,)
fc_out = acts['fc'][0].numpy()       # (2,)
softmax_out = probs[0].cpu().numpy()  # (2,)

C_fuse, T_fuse = fuse_out.shape

fig2 = plt.figure(figsize=(11, 5.5))
gs2 = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1.2, 1],
                        wspace=0.3, hspace=0.45, top=0.90, bottom=0.10,
                        left=0.07, right=0.95)

# Top-left: Fusion heatmap (192 ch x T)
ax_fuse = fig2.add_subplot(gs2[0, 0])
cmax_f = float(np.abs(fuse_out).max()) or 1.0
ax_fuse.imshow(fuse_out, aspect='auto', cmap='RdBu_r', vmin=-cmax_f, vmax=cmax_f,
               extent=[0, 10, C_fuse-0.5, -0.5], interpolation='nearest')
ax_fuse.set_xlabel('Time (s)', fontsize=7)
ax_fuse.set_ylabel('Feature channel', fontsize=7)
ax_fuse.set_title(f'Fusion output ({C_fuse} ch × {T_fuse} t) — all leads merged',
                  fontsize=8, fontweight='bold')
ax_fuse.tick_params(labelsize=5)

# Top-right: GAP bar chart (192-dim vector)
ax_gap = fig2.add_subplot(gs2[0, 1])
colors_gap = ['#C62828' if v > 0 else '#1565C0' for v in gap_out]
ax_gap.barh(np.arange(len(gap_out)), gap_out, color=colors_gap, edgecolor='none', height=1.0)
ax_gap.set_xlabel('Avg activation', fontsize=6)
ax_gap.set_ylabel('Channel', fontsize=6)
ax_gap.set_title(f'GAP → 192-dim vector', fontsize=8, fontweight='bold')
ax_gap.axvline(0, color='black', linewidth=0.5)
ax_gap.set_ylim(-0.5, len(gap_out)-0.5)
ax_gap.invert_yaxis()
ax_gap.tick_params(labelsize=4)

# Bottom: Vote visualization
ax_vote = fig2.add_subplot(gs2[1, :])

# Show each GAP channel as a "voter" with its contribution to PE vs Normal
# FC weights: fc.weight is (2, 192), fc.bias is (2,)
fc_w = net.fc.weight.detach().cpu().numpy()  # (2, 192)
fc_b = net.fc.bias.detach().cpu().numpy()    # (2,)

# Per-channel contribution to PE logit
contrib_pe = gap_out * fc_w[1, :]  # (192,)
contrib_norm = gap_out * fc_w[0, :]  # (192,)
contrib_diff = contrib_pe - contrib_norm  # positive = votes PE

order = np.argsort(contrib_diff)
colors_vote = ['#C62828' if c > 0 else '#1565C0' for c in contrib_diff[order]]
ax_vote.bar(np.arange(192), contrib_diff[order], color=colors_vote, edgecolor='none', width=1.0)
ax_vote.axhline(0, color='black', linewidth=0.5)
ax_vote.set_xlabel('Feature channels (sorted by vote)', fontsize=7)
ax_vote.set_ylabel('Vote strength\n(PreE − Normal)', fontsize=7)
ax_vote.set_title(f'Channel votes: {(contrib_diff > 0).sum()} channels vote PreE, '
                  f'{(contrib_diff <= 0).sum()} vote Normal  →  '
                  f'P(PreE) = {softmax_out[1]:.3f}',
                  fontsize=8, fontweight='bold')
ax_vote.tick_params(labelsize=5)
ax_vote.set_xlim(-1, 192)

fig2.suptitle(f'Fusion → Global Average Pooling → Classification vote  |  '
              f'Confident PreE (P(PreE)={probs_test[idx]:.3f})',
              fontsize=9, fontweight='bold')

plt.savefig('fig_activations_vote.pdf', dpi=300, bbox_inches='tight')
print('Saved fig_activations_vote.pdf')
plt.close()

print('Done.')
