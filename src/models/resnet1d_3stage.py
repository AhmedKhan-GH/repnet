"""3-stage 1D ResNet with the same filter widths and kernels as RepNet CrossLead
Deeper, but no per-lead processing and no cross-lead attention.

Architecture:
  Input (B, 12, 2500)
    Stage 1: ResBlock1D(12  → 48,  k=7) → MaxPool/2 → (B,  48, 1250)
    Stage 2: ResBlock1D(48  → 96,  k=5) → MaxPool/2 → (B,  96,  625)
    Stage 3: ResBlock1D(96  → 192, k=3) → MaxPool/2 → (B, 192,  312)
    GAP → Dropout → Linear(192 → 2)

Each ResBlock1D has two convs of the same kernel size with a 1x1 skip projection
when channel count changes.

This model is the natural ablation target for CrossLead Deeper: same depth (3
stages), same filter widths (48, 96, 192), same kernel triple (7, 5, 3), same
temporal receptive field (~180 ms at 250 Hz). The only differences are (a) the
input layer is a vanilla Conv1d(12 → 48) that mixes leads from the first conv,
and (b) there is no cross-lead attention.

Total parameters at default config: ~280K (vs 965K for CrossLead Deeper).
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model

logger = logging.getLogger(__name__)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float = 0.0):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=p, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.drop(out)


class ResNet1D3Stage(nn.Module):
    def __init__(
        self,
        n_leads:       int = 12,
        stage_filters: tuple[int, ...] = (48, 96, 192),
        kernels:       tuple[int, ...] = (7, 5, 3),
        dropout:       float = 0.0546,
        n_classes:     int = 2,
    ):
        super().__init__()
        assert len(stage_filters) == len(kernels)

        in_c = n_leads
        stages = []
        for f, k in zip(stage_filters, kernels):
            stages.append(ResBlock1D(in_c, f, kernel_size=k, dropout=0.0))
            stages.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_c = f
        self.stages = nn.Sequential(*stages)

        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc        = nn.Linear(in_c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("resnet1d_3stage")
class ResNet1D3StageModel(BaseModel):
    """Optuna-compatible wrapper for the 3-stage 1D ResNet ablation."""

    def __init__(
        self,
        stage_filters: tuple[int, ...] = (48, 96, 192),
        kernels:       tuple[int, ...] = (7, 5, 3),
        dropout:       float = 0.0546,
        lr:            float = 2.465e-3,
        batch_size:    int   = 64,
        epochs:        int   = 50,
        weight_decay:  float = 1.67e-4,
        loss_fn:       str   = "cross_entropy",
        **kwargs,
    ):
        self.net_params = dict(
            stage_filters=tuple(stage_filters),
            kernels=tuple(kernels),
            dropout=dropout,
        )
        self.lr           = lr
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.weight_decay = weight_decay
        self.loss_fn      = loss_fn
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.history: dict | None = None
        logger.info("ResNet1D3Stage using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "kernels":      trial.suggest_categorical("resnet3_kernels", [(7, 5, 3), (9, 5, 3), (5, 5, 3)]),
            "dropout":      trial.suggest_float("resnet3_dropout", 0.05, 0.4),
            "lr":           trial.suggest_float("resnet3_lr", 1e-4, 5e-3, log=True),
            "batch_size":   trial.suggest_categorical("resnet3_batch_size", [32, 64]),
            "weight_decay": trial.suggest_float("resnet3_wd", 1e-5, 1e-2, log=True),
        }

    def _build_criterion(self, y_train: np.ndarray):
        if self.loss_fn == "weighted":
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            weight = torch.tensor(
                [1.0, n_neg / max(n_pos, 1)], dtype=torch.float32
            ).to(self.device)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = ResNet1D3Stage(**self.net_params).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("ResNet1D3Stage parameters: %d", n_params)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        train_dl = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True, num_workers=2,
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
        self.model.eval()
        probs = torch.softmax(self.model(Xv), dim=1)[:, 1].cpu().numpy()
        return roc_auc_score(y_val, probs)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32)
        dl = DataLoader(
            TensorDataset(Xt),
            batch_size=self.batch_size, num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        probs = []
        for (xb,) in dl:
            xb = xb.to(self.device, non_blocking=True)
            probs.append(torch.softmax(self.model(xb), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)
