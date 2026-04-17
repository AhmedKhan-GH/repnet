"""RepNet Hybrid — CNN-Transformer with per-lead convolutions and cross-lead attention.

Architecture:
  1. Per-lead CNN: 12 independent Conv1D streams preserve lead identity
  2. Cross-lead attention: between conv stages, leads attend to each other
  3. Fusion: concatenate lead representations → final conv → GAP → classifier

  Stage 1: 12x Conv1D(1→F1, k=7) → CrossLeadAttention(12 tokens, dim=F1)
  Stage 2: 12x Conv1D(F1→F2, k=5) → CrossLeadAttention(12 tokens, dim=F2)
  Fuse:    Concatenate 12 leads → Conv1D(12*F2→F_out, k=1) → GAP → Dropout → Linear(2)

EDA justification:
  - Lead discriminability CV=2.236 (Step 9): leads contribute very unequally → attention
    should learn per-patient lead weighting
  - Cross-lead correlation Frobenius norm=1.028 (Step 11): modest cross-lead differences
    exist — attention can capture these without 2D conv
  - PSD peak at 2 Hz (Step 8): wide kernels in per-lead conv streams
  - Temporal width ~full strip (Step 10): MaxPool stacking builds receptive field
"""

import logging
import math

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced binary classification.

    Lin et al. (2017): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

logger = logging.getLogger(__name__)


class PerLeadConvBlock(nn.Module):
    """Apply the same Conv1D block independently to each of 12 leads.

    Input:  (batch, n_leads, C_in, T)
    Output: (batch, n_leads, C_out, T//2)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=p)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=p)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # x: (batch, n_leads, C, T)
        B, L, C, T = x.shape
        # Merge batch and leads for efficient conv
        x_flat = x.reshape(B * L, C, T)
        residual = self.skip(x_flat)
        out = self.act(self.bn1(self.conv1(x_flat)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        out = self.dropout(self.pool(out))
        # Unflatten
        _, C_out, T_out = out.shape
        return out.reshape(B, L, C_out, T_out)


class CrossLeadAttention(nn.Module):
    """Multi-head self-attention across 12 lead representations.

    Each lead's feature map is pooled to a single vector, then leads
    attend to each other. The attention output modulates the full
    feature maps via a gating mechanism.

    Input:  (batch, n_leads, C, T)
    Output: (batch, n_leads, C, T) — same shape, attention-weighted
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B*L, C, T) → (B*L, C, 1)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_leads, C, T)
        B, L, C, T = x.shape

        # Pool each lead to a vector: (batch, n_leads, C)
        tokens = self.pool(x.reshape(B * L, C, T)).squeeze(-1).reshape(B, L, C)

        # Self-attention across leads
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = self.norm(tokens + attn_out)  # residual + layer norm

        # Gate: convert attention output to per-lead weights
        gate_weights = self.gate(attn_out)  # (B, L, C)

        # Apply gate to full feature maps
        return x * gate_weights.unsqueeze(-1)  # broadcast over T


class RepNetHybrid(nn.Module):
    """CNN-Transformer hybrid with per-lead convolutions and cross-lead attention.

    Preserves lead identity throughout — attention operates on actual leads,
    not mixed feature channels.
    """

    def __init__(
        self,
        n_leads: int = 12,
        stage_filters: tuple[int, int] = (32, 64),
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        dropout: float = 0.1,
        n_heads: int = 4,
        n_classes: int = 2,
    ):
        super().__init__()
        f1, f2 = stage_filters

        # Stage 1: per-lead conv + cross-lead attention
        self.conv1 = PerLeadConvBlock(1, f1, wide_kernel, dropout)
        self.attn1 = CrossLeadAttention(f1, n_heads, dropout)

        # Stage 2: per-lead conv + cross-lead attention
        self.conv2 = PerLeadConvBlock(f1, f2, narrow_kernel, dropout)
        self.attn2 = CrossLeadAttention(f2, n_heads, dropout)

        # Fusion: concatenate leads → pointwise conv → GAP
        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f2, f2, kernel_size=1),
            nn.BatchNorm1d(f2),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f2, n_classes)

    def forward(self, x):
        # x: (batch, 12, 2500) — channels-first ECG
        B, L, T = x.shape

        # Reshape to (batch, n_leads, 1, T) — each lead is a single-channel signal
        x = x.unsqueeze(2)  # (B, 12, 1, T)

        # Stage 1
        x = self.conv1(x)    # (B, 12, F1, T//2)
        x = self.attn1(x)    # (B, 12, F1, T//2)

        # Stage 2
        x = self.conv2(x)    # (B, 12, F2, T//4)
        x = self.attn2(x)    # (B, 12, F2, T//4)

        # Fuse: reshape (B, 12, F2, T') → (B, 12*F2, T') → conv → GAP
        B, L, F, T_out = x.shape
        x = x.reshape(B, L * F, T_out)
        x = self.fuse(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_hybrid")
class RepNetHybridModel(BaseModel):
    """Optuna-compatible wrapper around RepNet Hybrid."""

    def __init__(self, stage_filters=(32, 64), wide_kernel=7, narrow_kernel=5,
                 dropout=0.1, n_heads=4,
                 lr=1e-3, batch_size=32, epochs=50,
                 loss_fn="weighted", focal_gamma=2.0, focal_alpha=0.25,
                 **kwargs):
        self.net_params = dict(
            stage_filters=stage_filters,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            dropout=dropout,
            n_heads=n_heads,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        logger.info("RepNetHybrid using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        f1 = trial.suggest_categorical("hybrid_stage1_filters", [16, 32])
        f2 = trial.suggest_categorical("hybrid_stage2_filters", [32, 64])
        loss_fn = trial.suggest_categorical(
            "hybrid_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": (f1, f2),
            "wide_kernel": trial.suggest_categorical("hybrid_wide_kernel", [5, 7, 9]),
            "narrow_kernel": trial.suggest_categorical("hybrid_narrow_kernel", [3, 5]),
            "dropout": trial.suggest_float("hybrid_dropout", 0.05, 0.4),
            "n_heads": trial.suggest_categorical("hybrid_n_heads", [2, 4]),
            "lr": trial.suggest_float("hybrid_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("hybrid_batch_size", [32, 64]),
            "epochs": trial.suggest_int("hybrid_epochs", 10, 50),
            "loss_fn": loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("hybrid_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("hybrid_focal_alpha", 0.1, 0.9)
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
        self.model = RepNetHybrid(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=1e-4,
        )
        criterion = self._build_criterion(y_train)

        # Keep data on CPU, use pinned memory for faster H2D transfer
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
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
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

            print(f"  Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f} | val_AUROC={val_auc:.4f}{marker}")

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
