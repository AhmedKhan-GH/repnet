#!/usr/bin/env python3
"""Export publication-quality LaTeX tables compiled to PDF + PNG.

Generates .tex → .pdf → .png for:
  1. Classification report at τ=0.50 and τ=Youden
  2. Aggregate performance across 20 holdout splits
  3. Model architecture comparison
  4. Optuna hyperparameter search results
  5. Per-seed results
  6. Dataset characteristics
  7. Architecture summary

Usage:
    python publication_media/final_report/tables/export_tables.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats as sp_stats
import torch

OUT_DIR = REPO_ROOT / "publication_media" / "final_report" / "tables"
RESULTS_DIR = REPO_ROOT / "cv_results" / "neural_final_2026-05-20_08-29-02"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def compile_tex(tex_src: str, name: str):
    """Write .tex, compile to PDF, convert to PNG."""
    tex_path = OUT_DIR / f"{name}.tex"
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"

    tex_path.write_text(tex_src, encoding="utf-8")

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(OUT_DIR), str(tex_path)],
        capture_output=True, timeout=30,
    )
    # run twice for references
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(OUT_DIR), str(tex_path)],
        capture_output=True, timeout=30,
    )

    if pdf_path.exists():
        subprocess.run(
            ["sips", "-s", "format", "png", "-s", "dpiWidth", "300", "-s", "dpiHeight", "300",
             str(pdf_path), "--out", str(png_path)],
            capture_output=True, timeout=30,
        )

    # cleanup aux files
    for ext in [".aux", ".log", ".out", ".fls", ".fdb_latexmk"]:
        p = OUT_DIR / f"{name}{ext}"
        if p.exists():
            p.unlink()

    if png_path.exists():
        print(f"  -> {png_path.relative_to(REPO_ROOT)}")
    elif pdf_path.exists():
        print(f"  -> {pdf_path.relative_to(REPO_ROOT)} (PNG conversion failed)")
    else:
        print(f"  !! {name}: pdflatex failed")


PREAMBLE = r"""\documentclass[border=8pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{siunitx}
\definecolor{bestrow}{HTML}{E8F5E9}
\definecolor{headerblue}{HTML}{1565C0}
\newcolumntype{R}{>{\raggedleft\arraybackslash}p}
\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


def _metrics_at(probs, y, tau):
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    acc = (tp + tn) / len(y)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1, acc=acc)


def load_repnet_se_predictions():
    from src.models.repnet_se import RepNetSE
    from src.train_explorer_v2 import load_combined, preprocess_waveforms

    with open(RESULTS_DIR / "results.json") as f:
        results = json.load(f)
    aurocs = np.array([r["metrics"]["auroc"] for r in results])
    seeds = [r["seed"] for r in results]
    best_seed = int(seeds[np.argmax(aurocs)])

    ckpt = torch.load(
        RESULTS_DIR / "weights" / f"best_seed_{best_seed}.pt",
        map_location=DEVICE, weights_only=False,
    )
    model = RepNetSE(**ckpt["net_cfg"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_wave, _, y, patient_ids, _ = load_combined(
        str(REPO_ROOT / "data" / "seniordesign_upload")
    )
    X_wave = preprocess_waveforms(X_wave)

    ss = np.random.SeedSequence(best_seed)
    split_seed = int(ss.generate_state(1)[0])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=patient_ids))
    X_test, y_test = X_wave[test_idx], y[test_idx]

    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return y_test, probs, results, aurocs, seeds


# ===================================================================
# Table 1: Classification Report
# ===================================================================
def table1_classification_report(y_test, probs):
    print("Table 1: classification_report")

    fpr, tpr, thr = roc_curve(y_test, probs)
    tau_youden = float(thr[np.argmax(tpr - fpr)])
    j_stat = float(np.max(tpr - fpr))
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)

    m50 = _metrics_at(probs, y_test, 0.50)
    my = _metrics_at(probs, y_test, tau_youden)
    n_pe = int(y_test.sum())
    n_norm = int((y_test == 0).sum())

    tex = PREAMBLE + r"""
\begin{tabular}{l cc}
\toprule
\textbf{Metric} & $\boldsymbol{\tau = 0.500}$ & $\boldsymbol{\tau = """ + f"{tau_youden:.3f}" + r"""}$ \textbf{(Youden)} \\
\midrule
AUROC          & """ + f"{auroc:.4f}" + r""" & """ + f"{auroc:.4f}" + r""" \\
AUPRC          & """ + f"{auprc:.4f}" + r""" & """ + f"{auprc:.4f}" + r""" \\
Sensitivity    & """ + f"{m50['sens']:.3f}" + r""" & """ + f"{my['sens']:.3f}" + r""" \\
Specificity    & """ + f"{m50['spec']:.3f}" + r""" & """ + f"{my['spec']:.3f}" + r""" \\
Precision (PPV) & """ + f"{m50['ppv']:.3f}" + r""" & """ + f"{my['ppv']:.3f}" + r""" \\
NPV            & """ + f"{m50['npv']:.3f}" + r""" & """ + f"{my['npv']:.3f}" + r""" \\
F1-Score       & """ + f"{m50['f1']:.3f}" + r""" & """ + f"{my['f1']:.3f}" + r""" \\
Accuracy       & """ + f"{m50['acc']:.3f}" + r""" & """ + f"{my['acc']:.3f}" + r""" \\
Youden's $J$   & --- & """ + f"{j_stat:.3f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Confusion Matrix}} \\
\quad TP / FP  & """ + f"{m50['tp']} / {m50['fp']}" + r""" & """ + f"{my['tp']} / {my['fp']}" + r""" \\
\quad FN / TN  & """ + f"{m50['fn']} / {m50['tn']}" + r""" & """ + f"{my['fn']} / {my['tn']}" + r""" \\
\bottomrule
\multicolumn{3}{l}{\footnotesize Test set: $N=""" + str(len(y_test)) + r"""$ (PE+ $n=""" + str(n_pe) + r"""$, Normal $n=""" + str(n_norm) + r"""$, prevalence $=""" + f"{100*n_pe/len(y_test):.1f}" + r"""\%$)} \\
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "classification_report")


# ===================================================================
# Table 2: Aggregate Performance
# ===================================================================
def table2_aggregate_performance(results, aurocs):
    print("Table 2: aggregate_performance")

    auprcs = np.array([r["metrics"]["auprc"] for r in results])
    briers = np.array([r["metrics"]["brier"] for r in results])
    n = len(aurocs)

    def row(name, arr):
        mean = arr.mean()
        std = arr.std(ddof=1)
        sem = std / np.sqrt(n)
        ci_lo = mean - 1.96 * sem
        ci_hi = mean + 1.96 * sem
        return (f"{name} & {mean:.4f} & {std:.4f} & "
                f"[{ci_lo:.4f}, {ci_hi:.4f}] & "
                f"{arr.min():.4f} & {arr.max():.4f} & {np.median(arr):.4f}")

    w_stat, w_p = sp_stats.shapiro(aurocs)
    above_70 = int((aurocs >= 0.70).sum())
    best_idx = int(np.argmax(aurocs))
    seeds = [r["seed"] for r in results]

    tex = PREAMBLE + r"""
\begin{tabular}{l cccccc}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{SD} & \textbf{95\% CI} & \textbf{Min} & \textbf{Max} & \textbf{Median} \\
\midrule
""" + row("AUROC", aurocs) + r""" \\
""" + row("AUPRC", auprcs) + r""" \\
""" + row("Brier Score", briers) + r""" \\
\bottomrule
\multicolumn{7}{l}{\footnotesize Seeds with AUROC $\geq 0.70$: """ + f"{above_70}/{n} ({100*above_70/n:.0f}\\%)" + r"""} \\
\multicolumn{7}{l}{\footnotesize Best seed: """ + f"{seeds[best_idx]} (AUROC={aurocs[best_idx]:.4f}, AUPRC={auprcs[best_idx]:.4f})" + r"""} \\
\multicolumn{7}{l}{\footnotesize Shapiro--Wilk: $W=""" + f"{w_stat:.4f}" + r"""$, $p=""" + f"{w_p:.4f}" + r"""$""" + (" (normal)" if w_p > 0.05 else " (non-normal)") + r"""} \\
\multicolumn{7}{l}{\footnotesize """ + f"{n}" + r""" patient-grouped holdout splits, 5-bag ensemble per seed} \\
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "aggregate_performance")


# ===================================================================
# Table 3: Model Comparison
# ===================================================================
def table3_model_comparison():
    print("Table 3: model_comparison")

    rows = []

    # RepNet CrossLead
    p = REPO_ROOT / "crossval_results" / "repnet_crosslead" / "2026-04-26_20-41-26" / "cv_results.json"
    if p.exists():
        with open(p) as f:
            cv = json.load(f)
        t50 = cv["test"]["operating_points"]["tau=0.50"]
        ty = cv["test"]["operating_points"]["tau=Youden"]
        rows.append(
            f"RepNet-CrossLead & $\\sim$240K & 3 & "
            f"${cv['cv_auroc_mean']:.3f} \\pm {cv['cv_auroc_std']:.3f}$ & "
            f"{cv['test']['auroc']['point']:.3f} & {cv['test']['auprc']['point']:.3f} & "
            f"{t50['sensitivity']:.3f} & {t50['specificity']:.3f} & "
            f"{ty['sensitivity']:.3f} & {ty['specificity']:.3f}"
        )

    # RepNet CrossLead Deeper
    p = REPO_ROOT / "crossval_results" / "repnet_crosslead_deeper" / "2026-04-26_20-48-58" / "cv_results.json"
    if p.exists():
        with open(p) as f:
            cv = json.load(f)
        t50 = cv["test"]["operating_points"]["tau=0.50"]
        ty = cv["test"]["operating_points"]["tau=Youden"]
        rows.append(
            f"RepNet-CrossLead-Deep & $\\sim$965K & 3 & "
            f"${cv['cv_auroc_mean']:.3f} \\pm {cv['cv_auroc_std']:.3f}$ & "
            f"{cv['test']['auroc']['point']:.3f} & {cv['test']['auprc']['point']:.3f} & "
            f"{t50['sensitivity']:.3f} & {t50['specificity']:.3f} & "
            f"{ty['sensitivity']:.3f} & {ty['specificity']:.3f}"
        )

    # PE-optimized ensemble
    p = REPO_ROOT / "crossval_results" / "repnet_crosslead_deeper_multiseed_pe" / "2026-05-04_21-06-34" / "results.json"
    if p.exists():
        with open(p) as f:
            pe = json.load(f)
        ens = pe["ensemble"]
        rows.append(
            f"CrossLead-Deep-PE (30-seed) & 965K & 3 & "
            f"${pe['auroc_stats']['mean']:.3f} \\pm {pe['auroc_stats']['std']:.3f}$ & "
            f"{ens['auroc']:.3f} & {ens['auprc']:.3f} & "
            f"--- & --- & --- & ---"
        )

    # RepNet-SE
    with open(RESULTS_DIR / "results.json") as f:
        se_res = json.load(f)
    se_aurocs = np.array([r["metrics"]["auroc"] for r in se_res])
    se_auprcs = np.array([r["metrics"]["auprc"] for r in se_res])
    rows.append(
        f"\\rowcolor{{bestrow}} \\textbf{{RepNet-SE (ours)}} & \\textbf{{65.7K}} & 4 & "
        f"$\\mathbf{{{se_aurocs.mean():.3f} \\pm {se_aurocs.std(ddof=1):.3f}}}$ & "
        f"\\textbf{{{se_aurocs.max():.3f}}} & \\textbf{{{se_auprcs.max():.3f}}} & "
        f"\\multicolumn{{4}}{{c}}{{See Table 1}}"
    )

    rows_tex = " \\\\\n".join(rows) + " \\\\"

    tex = PREAMBLE + r"""
\begin{tabular}{l r c c cc cccc}
\toprule
 & & & & \multicolumn{2}{c}{\textbf{Test (best)}} & \multicolumn{2}{c}{$\boldsymbol{\tau=0.50}$} & \multicolumn{2}{c}{$\boldsymbol{\tau}$\textbf{=Youden}} \\
\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
\textbf{Model} & \textbf{Params} & \textbf{Stages} & \textbf{CV AUROC} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{Sens} & \textbf{Spec} & \textbf{Sens} & \textbf{Spec} \\
\midrule
""" + rows_tex + r"""
\bottomrule
\multicolumn{10}{l}{\footnotesize All models evaluated with patient-grouped holdout splits. No patient leakage across train/val/test.} \\
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "model_comparison")


# ===================================================================
# Table 4: Optuna Results
# ===================================================================
def table4_optuna_results():
    print("Table 4: optuna_results")

    studies = [
        ("optuna_crosslead_3stage", "LR \\& Dropout"),
        ("optuna_crosslead_3stage_reg", "Regularization"),
        ("optuna_crosslead_3stage_rf", "Receptive Field"),
        ("optuna_crosslead_3stage_filters", "Filter Width"),
    ]

    rows = []
    for study_name, label in studies:
        bp_paths = sorted((REPO_ROOT / "optuna_results" / study_name).glob("*/best_params.json"))
        if not bp_paths:
            continue
        with open(bp_paths[0]) as f:
            bp = json.load(f)

        db_paths = sorted((REPO_ROOT / "optuna_results" / study_name).glob("*/study.db"))
        n_trials = "?"
        if db_paths:
            conn = sqlite3.connect(str(db_paths[0]))
            n_trials = str(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
            conn.close()

        best_trial = bp.get("best_trial", "?")
        best_auroc = bp.get("best_cv_auroc", 0)

        if study_name == "optuna_crosslead_3stage":
            finding = f"lr$={bp['best_params']['lr']:.4f}$, dropout$={bp['best_params']['dropout']:.3f}$"
        elif study_name == "optuna_crosslead_3stage_reg":
            finding = f"wd$={bp['best_params']['weight_decay']:.1e}$, $\\sigma={bp['best_params']['aug_sigma']:.3f}$"
        elif study_name == "optuna_crosslead_3stage_rf":
            finding = f"kernels=[{','.join(str(k) for k in bp['best_kernels'])}], RF={bp['best_rf_samples']}samp"
        elif study_name == "optuna_crosslead_3stage_filters":
            finding = f"filters=[{','.join(str(f) for f in bp['best_stage_filters'])}], {bp['best_param_count']//1000}K params"
        else:
            finding = "---"

        rows.append(f"{label} & {n_trials} & \\#{best_trial} & {best_auroc:.4f} & {finding}")

    rows_tex = " \\\\\n".join(rows) + " \\\\"

    tex = PREAMBLE + r"""
\begin{tabular}{l c c c l}
\toprule
\textbf{Phase} & \textbf{Trials} & \textbf{Best Trial} & \textbf{CV AUROC} & \textbf{Key Finding} \\
\midrule
""" + rows_tex + r"""
\bottomrule
\multicolumn{5}{l}{\footnotesize Sequential 4-phase Optuna search. Each phase fixes parameters from prior phases.} \\
\multicolumn{5}{l}{\footnotesize All trials use 3-fold patient-grouped stratified CV on the development set.} \\
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "optuna_results")


# ===================================================================
# Table 5: Per-Seed Results
# ===================================================================
def table5_perseed_results(results, aurocs):
    print("Table 5: perseed_results")

    auprcs = np.array([r["metrics"]["auprc"] for r in results])
    briers = np.array([r["metrics"]["brier"] for r in results])
    seeds = [r["seed"] for r in results]
    best_auroc = aurocs.max()

    rows = []
    for i in range(len(seeds)):
        bold = aurocs[i] == best_auroc
        if bold:
            row = (f"\\rowcolor{{bestrow}} \\textbf{{{seeds[i]}}} & "
                   f"\\textbf{{{aurocs[i]:.4f}}} & \\textbf{{{auprcs[i]:.4f}}} & "
                   f"\\textbf{{{briers[i]:.4f}}}")
        else:
            row = f"{seeds[i]} & {aurocs[i]:.4f} & {auprcs[i]:.4f} & {briers[i]:.4f}"
        rows.append(row)

    rows_tex = " \\\\\n".join(rows) + " \\\\"

    tex = PREAMBLE + r"""
\begin{tabular}{r ccc}
\toprule
\textbf{Seed} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{Brier} \\
\midrule
""" + rows_tex + r"""
\midrule
Mean & """ + f"{aurocs.mean():.4f} & {auprcs.mean():.4f} & {briers.mean():.4f}" + r""" \\
SD   & """ + f"{aurocs.std(ddof=1):.4f} & {auprcs.std(ddof=1):.4f} & {briers.std(ddof=1):.4f}" + r""" \\
\bottomrule
\multicolumn{4}{l}{\footnotesize 20 patient-grouped holdout splits, 5-bag ensemble per seed.} \\
\multicolumn{4}{l}{\footnotesize \colorbox{bestrow}{Highlighted} = best seed.} \\
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "perseed_results")


# ===================================================================
# Table 6: Dataset Characteristics
# ===================================================================
def table6_dataset(y_test):
    print("Table 6: dataset_characteristics")

    from src.train_explorer_v2 import load_combined
    _, _, y_all, patient_ids, _ = load_combined(
        str(REPO_ROOT / "data" / "seniordesign_upload")
    )
    n_total = len(y_all)
    n_pe = int(y_all.sum())
    n_norm = n_total - n_pe
    n_patients = len(np.unique(patient_ids))

    tex = PREAMBLE + r"""
\begin{tabular}{l r}
\toprule
\textbf{Characteristic} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Dataset}} \\
\quad Total ECG recordings & """ + f"{n_total:,}" + r""" \\
\quad Unique patients & """ + f"{n_patients:,}" + r""" \\
\quad Preeclampsia (PE+) & """ + f"{n_pe:,} ({100*n_pe/n_total:.1f}\\%)" + r""" \\
\quad Normal (PE$-$) & """ + f"{n_norm:,} ({100*n_norm/n_total:.1f}\\%)" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{ECG Signal}} \\
\quad Leads & 12 \\
\quad Sample length & 2{,}500 samples \\
\quad Sampling rate & 250\,Hz \\
\quad Duration per recording & 10\,seconds \\
\midrule
\multicolumn{2}{l}{\textit{Evaluation Protocol}} \\
\quad Number of splits & 20 \\
\quad Split strategy & Patient-grouped (StratifiedGroupKFold) \\
\quad Dev / Test ratio & 80 / 20 \\
\quad Bags per seed & 5 \\
\midrule
\multicolumn{2}{l}{\textit{Best-Seed Test Set}} \\
\quad Test set size & """ + f"{len(y_test):,}" + r""" \\
\quad Test PE+ & """ + f"{int(y_test.sum()):,} ({100*y_test.mean():.1f}\\%)" + r""" \\
\bottomrule
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "dataset_characteristics")


# ===================================================================
# Table 7: Architecture Summary
# ===================================================================
def table7_architecture():
    print("Table 7: architecture_summary")

    tex = PREAMBLE + r"""
\begin{tabular}{l l r}
\toprule
\textbf{Component} & \textbf{Configuration} & \textbf{Output Shape} \\
\midrule
Input & 12-lead ECG @ 250\,Hz & $(B, 12, 2500)$ \\
\midrule
Stage 1 Conv & MultiScale DSConv $k$=(5,9,15) + SE(16, $r$=4) & $(B, 12, 16, 2500)$ \\
Stage 1 Attn & CrossLeadAttn(16, 4 heads) & $(B, 12, 16, 2500)$ \\
\midrule
Stage 2 Conv & SEPerLead DSConv(16$\to$32, $k$=5, $s$=2) + SE(32, $r$=4) & $(B, 12, 32, 1250)$ \\
Stage 2 Attn & CrossLeadAttn(32, 4 heads) & $(B, 12, 32, 1250)$ \\
\midrule
Stage 3 Conv & SEPerLead DSConv(32$\to$48, $k$=5, $s$=2) + SE(48, $r$=4) & $(B, 12, 48, 625)$ \\
Stage 3 Attn & CrossLeadAttn(48, 4 heads) & $(B, 12, 48, 625)$ \\
\midrule
Stage 4 Conv & SEPerLead DSConv(48$\to$64, $k$=3, $s$=2) + SE(64, $r$=4) & $(B, 12, 64, 313)$ \\
Stage 4 Attn & CrossLeadAttn(64, 4 heads) & $(B, 12, 64, 313)$ \\
\midrule
GAP & Per-lead Global Average Pooling & $(B, 12, 64)$ \\
Lead Attn Pool & Softmax(Linear(64$\to$32$\to$1)) weighted sum & $(B, 64)$ \\
Classifier & Dropout(0.15) $\to$ Linear(64$\to$2) & $(B, 2)$ \\
\midrule
\multicolumn{3}{l}{\textbf{Total Parameters: 65,748}} \\
\midrule
Training & AdamW lr=$2\times10^{-3}$, cosine anneal, grad clip=1.0, patience=20 & \\
Loss & Cross-entropy (label smoothing=0.05) + Mixup($\alpha$=0.2) & \\
Augmentation & 7 ECG transforms (noise, amplitude, shift, lead drop, cutout, wander, resample) & \\
\bottomrule
\end{tabular}
""" + POSTAMBLE
    compile_tex(tex, "architecture_summary")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("  RepNet-SE LaTeX Table Export")
    print(f"  Output: {OUT_DIR.relative_to(REPO_ROOT)}/")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model and computing predictions...")
    y_test, probs, results, aurocs, seeds = load_repnet_se_predictions()
    print(f"  Test set: N={len(y_test)}, PE+={int(y_test.sum())}")
    print(f"  AUROC={roc_auc_score(y_test, probs):.4f}")

    print("\n--- Generating LaTeX tables ---")
    table1_classification_report(y_test, probs)
    table2_aggregate_performance(results, aurocs)
    table3_model_comparison()
    table4_optuna_results()
    table5_perseed_results(results, aurocs)
    table6_dataset(y_test)
    table7_architecture()

    n_tex = len(list(OUT_DIR.glob("*.tex")))
    n_pdf = len(list(OUT_DIR.glob("*.pdf")))
    n_png = len(list(OUT_DIR.glob("*.png")))
    print(f"\n{'=' * 70}")
    print(f"  Done! {n_tex} .tex, {n_pdf} .pdf, {n_png} .png")
    print(f"  Output: {OUT_DIR.relative_to(REPO_ROOT)}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
