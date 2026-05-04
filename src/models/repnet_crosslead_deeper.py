"""RepNet CrossLead Deeper — 3-stage variant of RepNetCrossLead.

Extrapolates the original (wide_kernel=7, narrow_kernel=5) two-stage pattern
with a third stage at kernel=3:

  Stage 1: PerLeadConvBlock(1  → 32,  k=7) → CrossLeadAttention(32)
  Stage 2: PerLeadConvBlock(32 → 64,  k=5) → CrossLeadAttention(64)
  Stage 3: PerLeadConvBlock(64 → 128, k=3) → CrossLeadAttention(128)
  Fuse:    concat(12 leads, 128) → Conv1d(12*128 → 128, k=1) → GAP → Dropout → Linear(2)

Receptive-field motivation (per-lead temporal RF, samples @ 250 Hz):
  Original 2-stage:  RF =  32 samples = 128 ms — barely covers QRS (80-120 ms)
  New 3-stage:       RF =  49 samples = 196 ms — covers QRS + ST + early T-wave

PE/HDP's main ECG signature is mild repolarization abnormalities (ST/T-wave
morphology), which lives in the 200-300 ms window after R. A 128 ms RF clips
that window short. 196 ms reaches into ST and the T-wave onset, while the new
attention level lets leads compare each other at three abstraction depths:
QRS-level → ST-level → morphology-level.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_crosslead import (
    CrossLeadAttention,
    FocalLoss,
    PerLeadConvBlock,
)

logger = logging.getLogger(__name__)


class RepNetCrossLeadDeeper(nn.Module):
    """3-stage CrossLead — adds a kernel-3, 128-channel stage on top of the original.

    `attn_stages` selectively enables CrossLeadAttention per stage. If None, all
    stages get attention (back-compat). Disabling early-stage attention saves
    parameters and removes a noise source when low-level features aren't yet
    class-discriminative.
    """

    def __init__(
        self,
        n_leads:        int = 12,
        stage_filters:  tuple[int, ...] = (32, 64, 128),
        kernels:        tuple[int, ...] = (7, 5, 3),
        dropout:        float = 0.1,
        n_heads:        int = 4,
        attn_stages:    tuple[bool, ...] | None = None,
        attn_tokens:    tuple[int, ...] | int = 1,
        n_classes:      int = 2,
    ):
        super().__init__()
        assert len(stage_filters) == len(kernels), \
            "stage_filters and kernels must have the same length"
        if attn_stages is None:
            attn_stages = tuple([True] * len(stage_filters))
        assert len(attn_stages) == len(stage_filters), \
            "attn_stages must have the same length as stage_filters"
        self.attn_stages = tuple(bool(x) for x in attn_stages)
        # `attn_tokens` controls how many temporal tokens each lead is pooled to
        # before cross-lead attention. Scalar broadcasts to all stages; tuple is
        # per-stage (must match stage_filters length).
        if isinstance(attn_tokens, int):
            attn_tokens = tuple([attn_tokens] * len(stage_filters))
        assert len(attn_tokens) == len(stage_filters), \
            "attn_tokens must have the same length as stage_filters"
        self.attn_tokens = tuple(int(t) for t in attn_tokens)

        in_c = 1
        stages = []
        for f, k, use_attn, n_tok in zip(stage_filters, kernels,
                                          self.attn_stages, self.attn_tokens):
            stage = nn.ModuleDict({
                "conv": PerLeadConvBlock(in_c, f, k, dropout),
            })
            if use_attn:
                stage["attn"] = CrossLeadAttention(f, n_heads, dropout, n_tokens=n_tok)
            stages.append(stage)
            in_c = f
        self.stages = nn.ModuleList(stages)

        f_last = stage_filters[-1]
        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f_last, f_last, kernel_size=1),
            nn.BatchNorm1d(f_last),
            nn.ReLU(inplace=True),
        )
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.head_drop  = nn.Dropout(dropout)
        self.fc         = nn.Linear(f_last, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        x = x.unsqueeze(2)                # (B, 12, 1, T)
        for stage in self.stages:
            x = stage["conv"](x)
            if "attn" in stage:
                x = stage["attn"](x)
        # x: (B, 12, F_last, T')
        B, L, F, T_out = x.shape
        x = x.reshape(B, L * F, T_out)
        x = self.fuse(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_crosslead_deeper")
class RepNetCrossLeadDeeperModel(BaseModel):
    """Optuna-compatible wrapper around the 3-stage RepNetCrossLeadDeeper."""

    def __init__(
        self,
        stage_filters:  tuple[int, ...] = (32, 64, 128),
        kernels:        tuple[int, ...] = (7, 5, 3),
        dropout:        float = 0.0636,
        n_heads:        int = 4,
        attn_stages:    tuple[bool, ...] | None = None,
        attn_tokens:    tuple[int, ...] | int = 1,
        lr:             float = 8.76e-4,
        batch_size:     int = 64,
        epochs:         int = 50,
        loss_fn:        str = "cross_entropy",
        focal_gamma:    float = 2.0,
        focal_alpha:    float = 0.25,
        weight_decay:   float = 1e-4,
        **kwargs,
    ):
        self.net_params = dict(
            stage_filters=tuple(stage_filters),
            kernels=tuple(kernels),
            dropout=dropout,
            n_heads=n_heads,
            attn_stages=attn_stages,
            attn_tokens=attn_tokens,
        )
        self.lr           = lr
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.loss_fn      = loss_fn
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha
        self.weight_decay = weight_decay
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.history: dict | None = None
        logger.info("RepNetCrossLeadDeeper using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        loss_fn = trial.suggest_categorical(
            "deeper_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": trial.suggest_categorical(
                "deeper_stage_filters", [(32, 64, 128), (16, 32, 64)],
            ),
            "kernels":       trial.suggest_categorical("deeper_kernels", [(7, 5, 3), (9, 5, 3)]),
            "dropout":       trial.suggest_float("deeper_dropout", 0.05, 0.4),
            "n_heads":       trial.suggest_categorical("deeper_n_heads", [2, 4]),
            "lr":            trial.suggest_float("deeper_lr", 1e-4, 5e-3, log=True),
            "batch_size":    trial.suggest_categorical("deeper_batch_size", [32, 64]),
            "epochs":        trial.suggest_int("deeper_epochs", 20, 60),
            "loss_fn":       loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("deeper_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("deeper_focal_alpha", 0.1, 0.9)
        return params

    def _build_criterion(self, y_train: np.ndarray):
        if self.loss_fn == "weighted":
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            weight = torch.tensor(
                [1.0, n_neg / max(n_pos, 1)], dtype=torch.float32
            ).to(self.device)
            return nn.CrossEntropyLoss(weight=weight)
        if self.loss_fn == "focal":
            return FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        return nn.CrossEntropyLoss()

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = RepNetCrossLeadDeeper(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        train_dl = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if torch.cuda.is_available() else False,
        )
        Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.history = {"train_loss": [], "val_auroc": []}
        best_val_auc, best_state = 0.0, None
        patience_counter, patience = 0, 10

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in train_dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            val_auc = self._score_device(Xv, y_val)
            self.history["train_loss"].append(avg_loss)
            self.history["val_auroc"].append(val_auc)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            print(f"  Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"val_AUROC={val_auc:.4f}{marker}")

            if patience_counter >= patience:
                print("  Early stop")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    @torch.no_grad()
    def _score_device(self, Xv: torch.Tensor, y_val: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        probs = torch.softmax(self.model(Xv), dim=1)[:, 1].cpu().numpy()
        return roc_auc_score(y_val, probs)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32)
        dl = DataLoader(
            TensorDataset(Xt),
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        probs = []
        for (xb,) in dl:
            xb = xb.to(self.device, non_blocking=True)
            probs.append(torch.softmax(self.model(xb), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)
