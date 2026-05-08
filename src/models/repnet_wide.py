"""RepNet Wide — wide-RF, low-param crosslead model. NO dilation.

Param-efficiency tricks (instead of dilation):
  - Per-lead shared weights (~12x saving on conv trunk)
  - Depthwise-separable convs (~k x saving per conv block)
  - Lead-attention pool fusion (~20x saving vs concat-fuse 1x1 conv)
  - Slim channel ladder: (16, 32, 64, 128) — half the deeper variant's width

RF expansion (without dilation):
  - 4 stages instead of 3
  - Kernels (7, 5, 3, 3) — keeps coarse-to-fine pattern
  - Per-lead temporal RF: 92 samples = 368 ms @ 250 Hz
    (vs 128 ms for 2-stage, 208 ms for 3-stage deeper)

Architecture:
  Stage 1: DSConv(  1 ->  16, k=7) → CrossLeadAttn( 16)
  Stage 2: DSConv( 16 ->  32, k=5) → CrossLeadAttn( 32)
  Stage 3: DSConv( 32 ->  64, k=3) → CrossLeadAttn( 64)
  Stage 4: DSConv( 64 -> 128, k=3) → CrossLeadAttn(128)
  Fusion:  per-lead GAP → LeadAttentionPool → Linear(128 -> 2)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_crosslead import CrossLeadAttention, FocalLoss

logger = logging.getLogger(__name__)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise (groups=in_ch) + 1x1 pointwise. Same I/O shape as Conv1d.

    Param count vs standard Conv1d(in_ch -> out_ch, k):
        standard:           in_ch * out_ch * k
        depthwise-separable: in_ch * (k + out_ch)
        ratio:              k * out_ch / (k + out_ch)  ≈ k when out_ch >> k
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        p = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=p, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class WidePerLeadBlock(nn.Module):
    """Per-lead block (weights shared across 12 leads).

    Two depthwise-separable convs + skip + MaxPool(2) + Dropout.
    Input/output: (B, n_leads, C, T).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.dsconv1 = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size)
        self.bn1     = nn.BatchNorm1d(out_ch)
        self.dsconv2 = DepthwiseSeparableConv1d(out_ch, out_ch, kernel_size)
        self.bn2     = nn.BatchNorm1d(out_ch)
        self.act     = nn.ReLU(inplace=True)
        self.pool    = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, T)
        B, L, C, T = x.shape
        x_flat = x.reshape(B * L, C, T)
        residual = self.skip(x_flat)
        out = self.act(self.bn1(self.dsconv1(x_flat)))
        out = self.bn2(self.dsconv2(out))
        out = self.act(out + residual)
        out = self.dropout(self.pool(out))
        _, C_out, T_out = out.shape
        return out.reshape(B, L, C_out, T_out)


class LeadAttentionPool(nn.Module):
    """Softmax-weighted pool across n_leads.

    Input:  (B, L, C)         per-lead embeddings (e.g., after GAP)
    Output: (B, C)            single weighted-sum vector

    Way cheaper than the concat-fuse 1x1 conv (~20x fewer params at the head).
    """

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores  = self.score(x).squeeze(-1)         # (B, L)
        weights = torch.softmax(scores, dim=1)      # (B, L)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * x).sum(dim=1)   # (B, C)


class RepNetWide(nn.Module):
    """4-stage param-efficient cross-lead net (no dilation)."""

    def __init__(
        self,
        n_leads:        int = 12,
        stage_filters:  tuple[int, ...] = (16, 32, 64, 128),
        kernels:        tuple[int, ...] = (7, 5, 3, 3),
        dropout:        float = 0.1,
        n_heads:        int = 4,
        attn_pool_hidden: int = 64,
        n_classes:      int = 2,
    ):
        super().__init__()
        assert len(stage_filters) == len(kernels), \
            "stage_filters and kernels must have the same length"

        in_c = 1
        stages = []
        for f, k in zip(stage_filters, kernels):
            stages.append(nn.ModuleDict({
                "conv": WidePerLeadBlock(in_c, f, k, dropout=dropout),
                "attn": CrossLeadAttention(f, n_heads, dropout),
            }))
            in_c = f
        self.stages = nn.ModuleList(stages)

        f_last = stage_filters[-1]
        self.lead_gap  = nn.AdaptiveAvgPool1d(1)
        self.lead_pool = LeadAttentionPool(f_last, hidden=attn_pool_hidden)
        self.head_drop = nn.Dropout(dropout)
        self.fc        = nn.Linear(f_last, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        x = x.unsqueeze(2)                 # (B, 12, 1, T)
        for stage in self.stages:
            x = stage["conv"](x)
            x = stage["attn"](x)
        # x: (B, 12, F_last, T_out)
        B, L, F, T_out = x.shape
        # Per-lead temporal GAP -> (B*L, F)
        x = self.lead_gap(x.reshape(B * L, F, T_out)).squeeze(-1)
        # Reshape -> (B, L, F)
        x = x.reshape(B, L, F)
        # Lead-attention pool across the 12 leads -> (B, F)
        x = self.lead_pool(x)
        return self.fc(self.head_drop(x))


@register_model("repnet_wide")
class RepNetWideModel(BaseModel):
    """Optuna-compatible wrapper around RepNetWide."""

    def __init__(
        self,
        stage_filters:  tuple[int, ...] = (16, 32, 64, 128),
        kernels:        tuple[int, ...] = (7, 5, 3, 3),
        dropout:        float = 0.1,
        n_heads:        int = 4,
        attn_pool_hidden: int = 64,
        lr:             float = 1e-3,
        batch_size:     int = 64,
        epochs:         int = 50,
        loss_fn:        str = "cross_entropy",
        focal_gamma:    float = 2.0,
        focal_alpha:    float = 0.25,
        **kwargs,
    ):
        self.net_params = dict(
            stage_filters=tuple(stage_filters),
            kernels=tuple(kernels),
            dropout=dropout,
            n_heads=n_heads,
            attn_pool_hidden=attn_pool_hidden,
        )
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.loss_fn     = loss_fn
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.history: dict | None = None
        logger.info("RepNetWide using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        loss_fn = trial.suggest_categorical(
            "wide_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": trial.suggest_categorical(
                "wide_stage_filters",
                [(16, 32, 64, 128), (8, 16, 32, 64), (16, 32, 64, 64)],
            ),
            "dropout":     trial.suggest_float("wide_dropout", 0.05, 0.4),
            "n_heads":     trial.suggest_categorical("wide_n_heads", [2, 4]),
            "lr":          trial.suggest_float("wide_lr", 1e-4, 5e-3, log=True),
            "batch_size":  trial.suggest_categorical("wide_batch_size", [32, 64]),
            "epochs":      trial.suggest_int("wide_epochs", 20, 60),
            "loss_fn":     loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("wide_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("wide_focal_alpha", 0.1, 0.9)
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
        self.model = RepNetWide(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=1e-4,
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
            val_auc  = self._score_device(Xv, y_val)
            self.history["train_loss"].append(avg_loss)
            self.history["val_auroc"].append(val_auc)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
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
