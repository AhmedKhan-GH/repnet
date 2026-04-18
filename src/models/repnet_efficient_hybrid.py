"""RepNet Efficient Hybrid — Depthwise separable CNN + cross-lead attention.

Combines the parameter efficiency of depthwise separable convolutions with
the cross-lead attention mechanism from the hybrid model.

Architecture:
  Stage 1: 12× DepthwiseSeparable(1→F1, k=7) → CrossLeadAttention(12 tokens, dim=F1)
  Stage 2: 12× DepthwiseSeparable(F1→F2, k=5) → CrossLeadAttention(12 tokens, dim=F2)
  Fuse:    Concatenate 12 leads → Conv1d(12*F2→F2, k=1) → GAP → Dropout → Linear(2)

Key idea: depthwise conv extracts per-lead features with ~6x fewer params than
standard conv, then cross-lead attention redistributes information across leads.
Gets the hybrid's lead-level attention with the efficient model's parameter budget.
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_baseline import FocalLoss

logger = logging.getLogger(__name__)


class PerLeadDSBlock(nn.Module):
    """Depthwise separable conv block applied independently to each lead.

    Input:  (batch, n_leads, C_in, T)
    Output: (batch, n_leads, C_out, T//2)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        p = kernel_size // 2

        # First depthwise separable
        self.dw1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                             padding=p, groups=in_channels)
        self.dw1_bn = nn.BatchNorm1d(in_channels)
        self.pw1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pw1_bn = nn.BatchNorm1d(out_channels)

        # Second depthwise separable
        self.dw2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                             padding=p, groups=out_channels)
        self.dw2_bn = nn.BatchNorm1d(out_channels)
        self.pw2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.pw2_bn = nn.BatchNorm1d(out_channels)

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
        x_flat = x.reshape(B * L, C, T)
        residual = self.skip(x_flat)

        out = self.act(self.dw1_bn(self.dw1(x_flat)))
        out = self.act(self.pw1_bn(self.pw1(out)))
        out = self.act(self.dw2_bn(self.dw2(out)))
        out = self.pw2_bn(self.pw2(out))

        out = self.act(out + residual)
        out = self.dropout(self.pool(out))

        _, C_out, T_out = out.shape
        return out.reshape(B, L, C_out, T_out)


class CrossLeadAttention(nn.Module):
    """Multi-head self-attention across 12 lead representations.

    Each lead's feature map is pooled to a single vector, then leads
    attend to each other. The attention output modulates the full
    feature maps via a gating mechanism.

    Input:  (batch, n_leads, C, T)
    Output: (batch, n_leads, C, T)
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
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
        B, L, C, T = x.shape
        tokens = self.pool(x.reshape(B * L, C, T)).squeeze(-1).reshape(B, L, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = self.norm(tokens + attn_out)
        gate_weights = self.gate(attn_out)
        return x * gate_weights.unsqueeze(-1)


class RepNetEfficientHybrid(nn.Module):
    """Depthwise separable CNN + cross-lead attention."""

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

        self.conv1 = PerLeadDSBlock(1, f1, wide_kernel, dropout)
        self.attn1 = CrossLeadAttention(f1, n_heads, dropout)

        self.conv2 = PerLeadDSBlock(f1, f2, narrow_kernel, dropout)
        self.attn2 = CrossLeadAttention(f2, n_heads, dropout)

        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f2, f2, kernel_size=1),
            nn.BatchNorm1d(f2),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f2, n_classes)

    def forward(self, x):
        B, L, T = x.shape
        x = x.unsqueeze(2)  # (B, 12, 1, T)

        x = self.conv1(x)
        x = self.attn1(x)

        x = self.conv2(x)
        x = self.attn2(x)

        B, L, F, T_out = x.shape
        x = x.reshape(B, L * F, T_out)
        x = self.fuse(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_efficient_hybrid")
class RepNetEfficientHybridModel(BaseModel):
    """Optuna-compatible wrapper around RepNet Efficient Hybrid."""

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
        logger.info("RepNetEfficientHybrid using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        f1 = trial.suggest_categorical("eff_hybrid_stage1_filters", [16, 32])
        f2 = trial.suggest_categorical("eff_hybrid_stage2_filters", [32, 64])
        loss_fn = trial.suggest_categorical(
            "eff_hybrid_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": (f1, f2),
            "wide_kernel": trial.suggest_categorical("eff_hybrid_wide_kernel", [5, 7, 9]),
            "narrow_kernel": trial.suggest_categorical("eff_hybrid_narrow_kernel", [3, 5]),
            "dropout": trial.suggest_float("eff_hybrid_dropout", 0.05, 0.4),
            "n_heads": trial.suggest_categorical("eff_hybrid_n_heads", [2, 4]),
            "lr": trial.suggest_float("eff_hybrid_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("eff_hybrid_batch_size", [32, 64]),
            "epochs": trial.suggest_int("eff_hybrid_epochs", 10, 50),
            "loss_fn": loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("eff_hybrid_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("eff_hybrid_focal_alpha", 0.1, 0.9)
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
        self.model = RepNetEfficientHybrid(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=1e-4,
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y_train, dtype=torch.long).to(self.device)
        train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
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
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        dl = DataLoader(TensorDataset(Xt), batch_size=self.batch_size)
        probs = []
        for (xb,) in dl:
            probs.append(torch.softmax(self.model(xb), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)
