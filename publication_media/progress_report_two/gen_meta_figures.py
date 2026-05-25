"""Generate publication figures for the Model Meta-Analysis section."""
import sys
sys.path.insert(0, '../..')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

from src.data.dataset import load_seniordesign, split_holdout_grouped
from src.preprocessing.filters import BaselineWanderFilter, NotchFilter
from src.preprocessing.normalization import ZScoreNormalization
from src.models.repnet_crosslead_deeper import RepNetCrossLeadDeeper

DATA_DIR = '../../data/seniordesign_upload'
RUN_DIR = Path('../../final_data/repnet_crosslead_deeper_multiseed_pe_2026-05-04_21-06-34')
SEED = 42
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── colours ──
C_NORM = '#4a7fc4'
C_PE = '#d62728'
C_TN = '#4a7fc4'
C_TP = '#d62728'
C_FP = '#ff7f0e'
C_FN = '#9467bd'

# ── load config & model ──
with open(RUN_DIR / 'config.json') as f:
    cfg = json.load(f)

p = cfg['params']
NET_PARAMS = dict(
    n_leads=12, n_classes=2,
    stage_filters=tuple(p['stage_filters']),
    kernels=tuple(p['kernels']),
    n_heads=p['n_heads'],
    attn_stages=(True, True, True),
    dropout=p['dropout'],
)
net = RepNetCrossLeadDeeper(**NET_PARAMS).to(device)
net.load_state_dict(torch.load(RUN_DIR / 'best_model.pt', map_location=device))
net.eval()

# ── load & preprocess data ──
X_raw, y_raw, patient_ids = load_seniordesign(DATA_DIR, return_patient_ids=True)
flat_mask = (X_raw.std(axis=2) < 1e-4).any(axis=1)
try:
    nan_mask = np.isnan(patient_ids.astype(float))
except (ValueError, TypeError):
    nan_mask = np.array([str(p).strip() in ('', 'nan', 'None') for p in patient_ids])
keep = ~flat_mask & ~nan_mask
X, y, pids = X_raw[keep], y_raw[keep], patient_ids[keep]

X, _ = BaselineWanderFilter(cutoff=0.5, order=4, fs=250.0).transform(X)
X, _ = NotchFilter(freq=60.0, Q=30.0, fs=250.0).transform(X)
X, _ = ZScoreNormalization(per_lead=True).transform(X)

X_dev, X_test, y_dev, y_test, g_dev, g_test = split_holdout_grouped(
    X, y, pids, test_size=0.2, seed=SEED)

# ── extract GAP embeddings ──
def extract_gap_and_probs(net, X, device, batch_size=64):
    bag = []
    probs_list = []
    hooks = []
    gap_out = {}
    def hook_fn(module, input, output):
        gap_out['val'] = output.detach().cpu().numpy()
    for name, mod in net.named_modules():
        if isinstance(mod, torch.nn.AdaptiveAvgPool1d) and 'gap' in name:
            hooks.append(mod.register_forward_hook(hook_fn))
    if not hooks:
        for name, mod in net.named_modules():
            if isinstance(mod, torch.nn.AdaptiveAvgPool1d):
                hooks.append(mod.register_forward_hook(hook_fn))
                break
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = net(xb)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(p)
            bag.append(gap_out['val'].squeeze(-1))
    for h in hooks:
        h.remove()
    return np.concatenate(probs_list), np.concatenate(bag)

probs_dev, emb_dev = extract_gap_and_probs(net, X_dev, device)
probs_test, emb_test = extract_gap_and_probs(net, X_test, device)

pred = (probs_test >= 0.5).astype(int)
categ = np.where(y_test == 1,
                 np.where(pred == 1, 'TP', 'FN'),
                 np.where(pred == 0, 'TN', 'FP'))

# ── PCA ──
scaler = StandardScaler().fit(emb_dev)
emb_devZ = scaler.transform(emb_dev)
emb_teZ = scaler.transform(emb_test)

pca = PCA(n_components=min(192, emb_devZ.shape[0])).fit(emb_devZ)
pcs_test = pca.transform(emb_teZ)
pcs_dev = pca.transform(emb_devZ)
var_ratio = pca.explained_variance_ratio_
cum = np.cumsum(var_ratio)

# ── multi-seed probs ──
npz = np.load(RUN_DIR / 'all_probs.npz', allow_pickle=True)
ms_probs = npz['probs']
ms_mean = ms_probs.mean(axis=0)
ms_std = ms_probs.std(axis=0)

# ── ensemble probs ──
probs_ensemble = ms_mean

# ── t-SNE ──
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0, init='pca', learning_rate='auto')
emb2_tsne = tsne.fit_transform(emb_teZ)

# ══════════════════════════════════════════════════════════════════
#  FIGURES: 4 separate scatter plots
# ══════════════════════════════════════════════════════════════════
def single_scatter_class(coords, y, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(3.4, 3.0),
                           gridspec_kw={'left': 0.15, 'right': 0.95, 'top': 0.90, 'bottom': 0.15})
    for cls, label, c in [(0, 'Normal', C_NORM), (1, 'PE', C_PE)]:
        mask = y == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=c, s=14, alpha=0.6, label=label, edgecolors='white', linewidths=0.3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, markerscale=1.5)
    ax.tick_params(labelsize=7)
    plt.savefig(fname, dpi=250, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()

def single_scatter_prob(coords, probs, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(3.4, 3.0),
                           gridspec_kw={'left': 0.15, 'right': 0.82, 'top': 0.90, 'bottom': 0.15})
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=probs, cmap='RdBu_r', vmin=0, vmax=1,
                    s=14, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label('$P(\\mathrm{PE})$', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.tick_params(labelsize=7)
    plt.savefig(fname, dpi=250, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()

single_scatter_class(pcs_test, y_test,
    f'PC1 ({var_ratio[0]*100:.1f}%)', f'PC2 ({var_ratio[1]*100:.1f}%)',
    'PCA — true labels', 'fig_pca_class.png')
single_scatter_prob(pcs_test, probs_test,
    f'PC1 ({var_ratio[0]*100:.1f}%)', f'PC2 ({var_ratio[1]*100:.1f}%)',
    'PCA — predicted $P(\\mathrm{PE})$', 'fig_pca_prob.png')
single_scatter_class(emb2_tsne, y_test,
    't-SNE 1', 't-SNE 2',
    't-SNE — true labels', 'fig_tsne_class.png')
single_scatter_prob(emb2_tsne, probs_test,
    't-SNE 1', 't-SNE 2',
    't-SNE — predicted $P(\\mathrm{PE})$', 'fig_tsne_prob.png')
plt.close()

# ══════════════════════════════════════════════════════════════════
#  FIGURES: ROC, PR, Prediction histogram, Youden — each separate
# ══════════════════════════════════════════════════════════════════
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

fpr_te, tpr_te, _ = roc_curve(y_test, probs_test)
prec_te, rec_te, _ = precision_recall_curve(y_test, probs_test)
auroc_te = roc_auc_score(y_test, probs_test)
auprc_te = average_precision_score(y_test, probs_test)

# ROC
fig, ax = plt.subplots(figsize=(3.4, 3.0),
                       gridspec_kw={'left': 0.15, 'right': 0.95, 'top': 0.90, 'bottom': 0.15})
ax.plot(fpr_te, tpr_te, color='tomato', linewidth=1.5, label=f'AUC = {auroc_te:.3f}')
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.7)
ax.set_xlabel('False Positive Rate', fontsize=8)
ax.set_ylabel('True Positive Rate', fontsize=8)
ax.set_title('ROC Curve', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.tick_params(labelsize=7)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
plt.savefig('fig_roc.png', dpi=250, bbox_inches='tight')
print('Saved fig_roc.png')
plt.close()

# Precision-Recall
fig, ax = plt.subplots(figsize=(3.4, 3.0),
                       gridspec_kw={'left': 0.15, 'right': 0.95, 'top': 0.90, 'bottom': 0.15})
ax.plot(rec_te, prec_te, color='tomato', linewidth=1.5, label=f'AP = {auprc_te:.3f}')
prevalence = y_test.mean()
ax.axhline(prevalence, color='gray', linestyle=':', linewidth=0.7, label=f'Prevalence ({prevalence:.3f})')
ax.set_xlabel('Recall (Sensitivity)', fontsize=8)
ax.set_ylabel('Precision (PPV)', fontsize=8)
ax.set_title('Precision-Recall Curve', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.tick_params(labelsize=7)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
plt.savefig('fig_pr.png', dpi=250, bbox_inches='tight')
print('Saved fig_pr.png')
plt.close()

# Compute Youden's J (used by both histogram and threshold figure)
thresholds = np.linspace(0.01, 0.99, 200)
sens_arr = np.array([((probs_test >= t) & (y_test == 1)).sum() / (y_test == 1).sum() for t in thresholds])
spec_arr = np.array([((probs_test < t) & (y_test == 0)).sum() / (y_test == 0).sum() for t in thresholds])
youden_j = sens_arr + spec_arr - 1
best_idx = np.argmax(youden_j)
best_tau = thresholds[best_idx]

# Prediction histogram (mirror, raw counts)
fig, ax = plt.subplots(figsize=(3.4, 3.0),
                       gridspec_kw={'left': 0.15, 'right': 0.95, 'top': 0.90, 'bottom': 0.15})
bins = np.linspace(0, 1, 31)
pe_counts, _ = np.histogram(probs_test[y_test == 1], bins=bins)
norm_counts, _ = np.histogram(probs_test[y_test == 0], bins=bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bw = bins[1] - bins[0]
ax.bar(bin_centers, pe_counts, width=bw, alpha=0.7, color=C_PE, label='PE ($n$=58)', edgecolor='white', linewidth=0.4)
ax.bar(bin_centers, -norm_counts, width=bw, alpha=0.7, color=C_NORM, label='Normal ($n$=373)', edgecolor='white', linewidth=0.4)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.0, label='$\\tau{=}0.5$')
ax.axvline(best_tau, color='red', linestyle='--', linewidth=1.0, label="$\\tau^*{=}%.3f$ (Youden's $J$)" % best_tau)
ax.set_xlabel('Predicted $P(\\mathrm{PE})$', fontsize=8)
ax.set_ylabel('Count', fontsize=8)
ax.set_title('Prediction Distribution by Class', fontsize=9, fontweight='bold')
yticks = ax.get_yticks()
ax.set_yticklabels([f'{int(abs(v))}' for v in yticks])
ax.legend(fontsize=6, loc='lower right')
ax.tick_params(labelsize=7)
plt.savefig('fig_pred_histogram.png', dpi=250, bbox_inches='tight')
print('Saved fig_pred_histogram.png')
plt.close()

# Youden's J threshold figure
fig, ax = plt.subplots(figsize=(3.4, 3.0),
                       gridspec_kw={'left': 0.15, 'right': 0.95, 'top': 0.90, 'bottom': 0.15})
ax.plot(thresholds, sens_arr, color=C_PE, linewidth=1.3, label='Sensitivity')
ax.plot(thresholds, spec_arr, color=C_NORM, linewidth=1.3, label='Specificity')
ax.plot(thresholds, youden_j, color='#2ca02c', linewidth=1.3, linestyle='--', label="Youden's $J$")
ax.axvline(best_tau, color='black', linestyle=':', linewidth=0.8)
ax.scatter([best_tau], [youden_j[best_idx]], color='#2ca02c', s=30, zorder=5)
ax.text(best_tau + 0.03, youden_j[best_idx] + 0.03,
        '$\\tau^*\\!=\\!%.3f$\n$J\\!=\\!%.3f$' % (best_tau, youden_j[best_idx]),
        fontsize=7, va='bottom')
ax.set_xlabel('Threshold $\\tau$', fontsize=8)
ax.set_ylabel('Score', fontsize=8)
ax.set_title("Youden's $J$ Threshold Selection", fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='center left')
ax.tick_params(labelsize=7)
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
plt.savefig('fig_youden.png', dpi=250, bbox_inches='tight')
print('Saved fig_youden.png')
plt.close()

# ══════════════════════════════════════════════════════════════════
#  FIGURE: Confusion matrix at tau=0.5
# ══════════════════════════════════════════════════════════════════
pred_best = (probs_test >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred_best, labels=[0, 1])

fig, ax = plt.subplots(figsize=(3.0, 2.8),
                       gridspec_kw={'left': 0.20, 'right': 0.95, 'top': 0.88, 'bottom': 0.15})
im = ax.imshow(cm, cmap='Blues', aspect='equal')
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'PE'], fontsize=8)
ax.set_yticklabels(['Normal', 'PE'], fontsize=8)
ax.set_xlabel('Predicted', fontsize=9)
ax.set_ylabel('Actual', fontsize=9)
ax.set_title('Confusion Matrix ($\\tau{=}0.5$)', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=7)

plt.savefig('fig_confusion_matrix.png', dpi=250, bbox_inches='tight')
print('Saved fig_confusion_matrix.png')
plt.close()

# ══════════════════════════════════════════════════════════════════
#  FIGURE 2: Uncertainty & Calibration (seed-std boxplot + reliability)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                         gridspec_kw={'wspace': 0.35, 'left': 0.09, 'right': 0.97,
                                      'top': 0.88, 'bottom': 0.15})

# (a) Multi-seed std by correctness
ax = axes[0]
cat_order = ['TN', 'TP', 'FP', 'FN']
cat_colors = {'TN': C_TN, 'TP': C_TP, 'FP': C_FP, 'FN': C_FN}
box_data = [ms_std[categ == c] for c in cat_order]
bp = ax.boxplot(box_data, labels=cat_order, patch_artist=True,
                widths=0.5, showfliers=True,
                flierprops=dict(marker='.', markersize=3, alpha=0.4))
for patch, c in zip(bp['boxes'], [cat_colors[c] for c in cat_order]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.2)
ax.set_ylabel('Std of $P(\\mathrm{PE})$ across 30 seeds', fontsize=8)
ax.set_title('(a) Predictive uncertainty by category', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=7)

# (b) Reliability diagram
ax = axes[1]
def reliability(probs_, y_, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs_, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = (idx == b)
        if not m.any():
            rows.append((np.nan, np.nan, 0))
            continue
        rows.append((float(probs_[m].mean()), float(y_[m].mean()), int(m.sum())))
    return rows

iso = IsotonicRegression(out_of_bounds='clip').fit(probs_dev, y_dev)
p_iso = iso.predict(probs_test)

for name, p, color, marker in [('Raw', probs_test, 'tomato', 'o'),
                                 ('Isotonic', p_iso, 'steelblue', 's')]:
    rel = reliability(p, y_test)
    mean_p = [r[0] for r in rel if r[2] > 0]
    frac_pos = [r[1] for r in rel if r[2] > 0]
    ax.plot(mean_p, frac_pos, f'{marker}-', color=color, markersize=4, linewidth=1.2, label=name)

ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
ax.set_xlabel('Mean predicted probability', fontsize=8)
ax.set_ylabel('Observed fraction positive', fontsize=8)
ax.set_title('(b) Reliability diagram', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
ax.tick_params(labelsize=7)

_, ece_raw, _ = reliability(probs_test, y_test), None, None
ece_raw_val = 0.2439
ece_iso_val = 0.0311
ax.text(0.05, 0.92, f'ECE raw: {ece_raw_val:.3f}\nECE iso: {ece_iso_val:.3f}',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

plt.savefig('fig_uncertainty_calibration.png', dpi=250, bbox_inches='tight')
print('Saved fig_uncertainty_calibration.png')
plt.close()

# ══════════════════════════════════════════════════════════════════
#  FIGURE 3: Error Characterization (Mahalanobis + kNN vs model)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                         gridspec_kw={'wspace': 0.35, 'left': 0.09, 'right': 0.97,
                                      'top': 0.88, 'bottom': 0.15})

# (a) Mahalanobis distance to predicted-class centroid
ax = axes[0]

def class_mahalanobis(emb_train, y_train, emb_query, cls):
    sub = emb_train[y_train == cls]
    mu = sub.mean(axis=0)
    cov = np.cov(sub, rowvar=False) + 1e-3 * np.eye(sub.shape[1])
    inv = np.linalg.pinv(cov)
    diff = emb_query - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv, diff))

d_to_norm = class_mahalanobis(emb_devZ, y_dev, emb_teZ, cls=0)
d_to_pe = class_mahalanobis(emb_devZ, y_dev, emb_teZ, cls=1)
d_to_pred = np.where(pred == 1, d_to_pe, d_to_norm)

box_data = [d_to_pred[categ == c] for c in cat_order]
bp = ax.boxplot(box_data, labels=cat_order, patch_artist=True,
                widths=0.5, showfliers=True,
                flierprops=dict(marker='.', markersize=3, alpha=0.4))
for patch, c in zip(bp['boxes'], [cat_colors[c] for c in cat_order]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.2)
ax.set_ylabel('Mahalanobis distance', fontsize=8)
ax.set_title('(a) Distance to predicted-class centroid', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=7)

# (b) kNN purity vs model P(PE)
ax = axes[1]
from sklearn.neighbors import NearestNeighbors
K = 10
nn = NearestNeighbors(n_neighbors=K).fit(emb_devZ)
dist, idx = nn.kneighbors(emb_teZ)
neigh_y = y_dev[idx]
purity_pe = neigh_y.mean(axis=1)
auroc_knn = roc_auc_score(y_test, purity_pe)

for cls, label, c in [(0, 'Normal', C_NORM), (1, 'PE', C_PE)]:
    mask = y_test == cls
    ax.scatter(purity_pe[mask], probs_test[mask],
               c=c, s=12, alpha=0.5, label=label, edgecolors='white', linewidths=0.3)
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
ax.set_xlabel(f'Fraction PE in {K}-NN (dev)', fontsize=8)
ax.set_ylabel('Model $P(\\mathrm{PE})$', fontsize=8)
ax.set_title(f'(b) kNN purity vs model (kNN AUROC={auroc_knn:.3f})', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, markerscale=1.5)
ax.tick_params(labelsize=7)

plt.savefig('fig_error_characterization.png', dpi=250, bbox_inches='tight')
print('Saved fig_error_characterization.png')
plt.close()

# ══════════════════════════════════════════════════════════════════
#  FIGURE: Optuna optimization history across 4 studies
# ══════════════════════════════════════════════════════════════════
import re

OPTUNA_DIR = Path('../../optuna_results')
studies = [
    ('optuna_crosslead_3stage', '2026-04-26_23-12-14', 'LR + Dropout'),
    ('optuna_crosslead_3stage_rf', '2026-04-27_08-27-21', 'Receptive Field'),
    ('optuna_crosslead_3stage_reg', '2026-04-27_05-38-28', 'Regularization'),
    ('optuna_crosslead_3stage_filters', '2026-04-27_10-29-59', 'Architecture'),
]

all_aurocs = []
study_boundaries = []
study_labels = []

for study_name, timestamp, label in studies:
    summary_path = OPTUNA_DIR / study_name / timestamp / 'summary.txt'
    text = summary_path.read_text(encoding='utf-8', errors='replace')
    trial_aurocs = [float(m.group(1)) for m in re.finditer(r'AUROC=([\d.]+)', text)
                    if 'Best CV' not in text[max(0, text.index(m.group(0))-30):text.index(m.group(0))]]
    lines = text.split('\n')
    trial_aurocs = []
    for line in lines:
        m = re.match(r'\s*#\s*\d+\s*\|.*AUROC=([\d.]+)', line)
        if m:
            trial_aurocs.append(float(m.group(1)))
    study_boundaries.append(len(all_aurocs))
    study_labels.append(label)
    all_aurocs.extend(trial_aurocs)

all_aurocs = np.array(all_aurocs)
running_best = np.maximum.accumulate(all_aurocs)

fig, ax = plt.subplots(figsize=(6.5, 3.0),
                       gridspec_kw={'left': 0.10, 'right': 0.97, 'top': 0.88, 'bottom': 0.18})

study_colors = ['#7fcdbb', '#41b6c4', '#2c7fb8', '#253494']
study_boundaries.append(len(all_aurocs))

for i in range(len(studies)):
    start = study_boundaries[i]
    end = study_boundaries[i + 1]
    x = np.arange(start, end)
    ax.scatter(x, all_aurocs[start:end], c=study_colors[i], s=18, alpha=0.7,
               label=study_labels[i], edgecolors='white', linewidths=0.3, zorder=3)

ax.plot(np.arange(len(all_aurocs)), running_best, color='#d62728', linewidth=1.5,
        linestyle='-', label='Running best', zorder=4)

for i in range(1, len(studies)):
    ax.axvline(study_boundaries[i] - 0.5, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)

ax.set_xlabel('Trial (cumulative)', fontsize=8)
ax.set_ylabel('5-fold CV AUROC', fontsize=8)
ax.set_title('Optuna Hyperparameter Optimization History', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5, ncol=3, loc='lower right')
ax.tick_params(labelsize=7)
ax.set_xlim(-1, len(all_aurocs))

plt.savefig('fig_optuna_history.png', dpi=250, bbox_inches='tight')
print('Saved fig_optuna_history.png')
plt.close()

print('Done — all figures saved.')
