"""4-block 1D ResNet over the raw 12-lead input (no per-lead share, no attention).

Architecture:
  Input (B, 12, 2500)
    Stem:    Conv1d(12 → 32, k=11, stride=1) → BN → ReLU
    Block 1: ResBlock1D(32  → 32,  k=11, stride=1)
    Block 2: ResBlock1D(32  → 64,  k=11, stride=2)
    Block 3: ResBlock1D(64  → 128, k=11, stride=2)
    Block 4: ResBlock1D(128 → 256, k=11, stride=1)
    GlobalAvgPool → (B, 256)
    Dropout → Linear(256 → 2)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model

logger = logging.getLogger(__name__)


class ResBlock1D(nn.Module):
    """Standard 1D residual block (conv-BN-ReLU-conv-BN + skip)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 11,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=p, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
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


class SimpleResNet1D(nn.Module):
    def __init__(
        self,
        n_leads:        int = 12,
        stem_channels:  int = 32,
        block_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        block_strides:  tuple[int, int, int, int] = (1, 2, 2, 1),
        kernel_size:    int = 11,
        stem_kernel:    int = 11,
        stem_stride:    int = 1,
        dropout:        float = 0.1,
        block_dropout:  float = 0.0,
        n_classes:      int = 2,
    ):
        super().__init__()
        assert len(block_channels) == len(block_strides) == 4

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, stem_channels, kernel_size=stem_kernel,
                      stride=stem_stride, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        in_c = stem_channels
        blocks = []
        for out_c, s in zip(block_channels, block_strides):
            blocks.append(ResBlock1D(in_c, out_c, kernel_size=kernel_size,
                                     stride=s, dropout=block_dropout))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc        = nn.Linear(in_c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)            # (B, C)
        return self.fc(self.head_drop(x))      # logits (B, n_classes)


@register_model("simple_resnet1d")
class SimpleResNet1DModel(BaseModel):
    """Optuna-compatible wrapper for the 4-block SimpleResNet1D."""

    def __init__(
        self,
        stem_channels:   int = 32,
        block_channels:  tuple[int, int, int, int] = (32, 64, 128, 256),
        block_strides:   tuple[int, int, int, int] = (1, 2, 2, 1),
        kernel_size:     int = 11,
        stem_kernel:     int = 11,
        stem_stride:     int = 1,
        dropout:         float = 0.1,
        block_dropout:   float = 0.0,
        lr:              float = 5e-5,
        weight_decay:    float = 1e-4,
        batch_size:      int = 32,
        epochs:          int = 300,
        patience:        int = 50,
        loss_fn:         str = "cross_entropy",   # "cross_entropy" | "weighted"
        use_cosine_lr:   bool = False,
        **kwargs,
    ):
        self.net_params = dict(
            stem_channels=stem_channels,
            block_channels=tuple(block_channels),
            block_strides=tuple(block_strides),
            kernel_size=kernel_size,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            dropout=dropout,
            block_dropout=block_dropout,
        )
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.patience      = patience
        self.loss_fn       = loss_fn
        self.use_cosine_lr = use_cosine_lr

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.history: dict | None = None
        logger.info("SimpleResNet1D using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "kernel_size":  trial.suggest_categorical("simple_resnet1d_kernel_size", [7, 9, 11]),
            "dropout":      trial.suggest_float("simple_resnet1d_dropout", 0.05, 0.3),
            "lr":           trial.suggest_float("simple_resnet1d_lr", 1e-6, 5e-4, log=True),
            "weight_decay": trial.suggest_float("simple_resnet1d_wd", 1e-5, 1e-3, log=True),
            "batch_size":   trial.suggest_categorical("simple_resnet1d_batch_size", [32, 64]),
            "epochs":       trial.suggest_int("simple_resnet1d_epochs", 100, 400),
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
        self.model = SimpleResNet1D(**self.net_params).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            if self.use_cosine_lr else None
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y_train, dtype=torch.long).to(self.device)
        train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.history = {"train_loss": [], "val_auroc": []}
        best_val_auc, best_state = 0.0, None
        patience_counter = 0

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

            if scheduler is not None:
                scheduler.step()

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

            print(f"  Epoch {epoch+1:4d}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"val_AUROC={val_auc:.4f}{marker}")

            if patience_counter >= self.patience:
                print(f"  Early stop @ epoch {epoch+1} (best val AUROC={best_val_auc:.4f})")
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
