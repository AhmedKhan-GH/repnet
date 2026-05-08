"""RepNet CrossLead Stem — 2-stage cross-lead model with a per-lead conv stem.

Extends RepNetCrossLead by adding one per-lead conv block BEFORE the first
attention layer. The stem processes each lead independently to build up local
temporal features (QRS, P-wave shapes) before leads start attending to each other.

Architecture:
  Stem:    PerLeadConvBlock(  1 -> 16, k=9)  — per-lead only, no mixing
  Stage 1: PerLeadConvBlock( 16 -> 32, k=7) → CrossLeadAttention(32)
  Stage 2: PerLeadConvBlock( 32 -> 64, k=5) → CrossLeadAttention(64)
  Fusion:  concat 12 leads → Conv1d(12*64→64, k=1) → GAP → Linear(64→2)

Receptive field: ~292 ms @ 250 Hz (vs 128 ms for 2-stage without stem)
Parameters:    ~130K  (vs ~117K for 2-stage)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_crosslead import CrossLeadAttention, FocalLoss, PerLeadConvBlock

logger = logging.getLogger(__name__)


class RepNetCrossLeadStem(nn.Module):
    """2-stage cross-lead model with a per-lead convolutional stem."""

    def __init__(
        self,
        n_leads:       int = 12,
        stem_filters:  int = 16,
        stem_kernel:   int = 9,
        stage_filters: tuple[int, int] = (32, 64),
        wide_kernel:   int = 7,
        narrow_kernel: int = 5,
        dropout:       float = 0.1,
        n_heads:       int = 4,
        n_classes:     int = 2,
    ):
        super().__init__()
        f1, f2 = stage_filters

        self.stem  = PerLeadConvBlock(1,   stem_filters, stem_kernel,   dropout)

        self.conv1 = PerLeadConvBlock(stem_filters, f1, wide_kernel,   dropout)
        self.attn1 = CrossLeadAttention(f1, n_heads, dropout)

        self.conv2 = PerLeadConvBlock(f1,  f2,          narrow_kernel, dropout)
        self.attn2 = CrossLeadAttention(f2, n_heads, dropout)

        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f2, f2, kernel_size=1),
            nn.BatchNorm1d(f2),
            nn.ReLU(inplace=True),
        )
        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc        = nn.Linear(f2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        x = x.unsqueeze(2)   # (B, 12, 1, T)

        x = self.stem(x)     # (B, 12, stem_filters, T//2)

        x = self.conv1(x)    # (B, 12, F1, T//4)
        x = self.attn1(x)

        x = self.conv2(x)    # (B, 12, F2, T//8)
        x = self.attn2(x)

        B, L, F, T_out = x.shape
        x = x.reshape(B, L * F, T_out)
        x = self.fuse(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_crosslead_stem")
class RepNetCrossLeadStemModel(BaseModel):
    """Optuna-compatible wrapper around RepNetCrossLeadStem."""

    def __init__(
        self,
        stem_filters:  int = 16,
        stem_kernel:   int = 9,
        stage_filters: tuple[int, int] = (32, 64),
        wide_kernel:   int = 7,
        narrow_kernel: int = 5,
        dropout:       float = 0.1,
        n_heads:       int = 4,
        lr:            float = 1e-3,
        batch_size:    int = 64,
        epochs:        int = 50,
        loss_fn:       str = "cross_entropy",
        focal_gamma:   float = 2.0,
        focal_alpha:   float = 0.25,
        **kwargs,
    ):
        self.net_params = dict(
            stem_filters=stem_filters,
            stem_kernel=stem_kernel,
            stage_filters=tuple(stage_filters),
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            dropout=dropout,
            n_heads=n_heads,
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
        logger.info("RepNetCrossLeadStem using device: %s", self.device)

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
        self.model = RepNetCrossLeadStem(**self.net_params).to(self.device)

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
            persistent_workers=torch.cuda.is_available(),
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
                n_batches  += 1

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

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, self.predict_proba(X))
