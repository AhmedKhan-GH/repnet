"""4-layer 1D ResNet for 12-lead ECG. First layer channel-independent.

Diagnosis from the previous hybrid run: train loss collapsed (0.66 -> 0.08)
while val AUROC degraded after epoch 2. ~370k params + cross-lead attn +
temporal attn + lr=8.76e-4 + dropout=0.06 was too rich for ~1700 dev samples.

This model strips back to:
  - 1 per-lead block (weight-shared across 12 leads): low-level lead-specific
    morphology with strong regularization from sharing.
  - 3 channel-mixed 1D ResNet blocks: cross-lead correlations learned by
    ordinary 1D convs once the lead axis is collapsed.
  - No attention. GAP -> dropout -> linear head.
  - Stronger regularization defaults (dropout=0.2, weight_decay=1e-3).

Forward pass (input x: (B, 12, 2500)):

  Regime 1 (per-lead):
    unsqueeze(2)                                  -> (B, 12, 1, 2500)
    block1: PerLeadConvBlock(1 -> F1, k=7)        -> (B, 12, F1, 1250)

  Pivot:
    reshape (B, 12, F1, 1250) -> (B, 12*F1, 1250)
    fuse: Conv1d(12*F1 -> F2, k=1) + BN + ReLU    -> (B, F2, 1250)

  Regime 2 (channel-mixed ResNet stack):
    block2: ChannelMixedConvBlock(F2 -> F2, k=5)  -> (B, F2, 625)
    block3: ChannelMixedConvBlock(F2 -> F3, k=5)  -> (B, F3, 312)
    block4: ChannelMixedConvBlock(F3 -> F4, k=3)  -> (B, F4, 156)

  Head:
    GAP                                            -> (B, F4)
    Dropout -> Linear(F4, 2)                       -> (B, 2)

Default filter progression keeps F4 capped at 128 (no 256 stage) to limit
capacity. Dropout 0.2 + weight_decay 1e-3 are the regularization knobs.
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_baseline import FocalLoss
from .repnet_baseline_large import PerLeadConvBlock
from .repnet_crosslead_hybrid import ChannelMixedConvBlock

logger = logging.getLogger(__name__)


class RepNetResNetHybrid(nn.Module):
    """4-block 1D ResNet: 1 per-lead + 3 channel-mixed."""

    def __init__(
        self,
        n_leads: int = 12,
        f1: int = 32,
        f2: int = 64,
        f3: int = 128,
        f4: int = 128,
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        narrow_kernel_2: int = 3,
        dropout: float = 0.2,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_leads = n_leads

        # Block 1: per-lead, weight-shared across 12 leads
        self.block1 = PerLeadConvBlock(1, f1, wide_kernel, dropout)

        # Pivot: 1x1 projection collapses lead axis into channels
        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f1, f2, kernel_size=1),
            nn.BatchNorm1d(f2),
            nn.ReLU(inplace=True),
        )

        # Blocks 2-4: channel-mixed ResNet
        self.block2 = ChannelMixedConvBlock(f2, f2, narrow_kernel, dropout)
        self.block3 = ChannelMixedConvBlock(f2, f3, narrow_kernel, dropout)
        self.block4 = ChannelMixedConvBlock(f3, f4, narrow_kernel_2, dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f4, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv trunk + GAP, returning (B, F4) before the head."""
        # x: (B, 12, 2500)
        x = x.unsqueeze(2)        # (B, 12, 1, 2500)

        # Per-lead regime
        x = self.block1(x)        # (B, 12, F1, 1250)

        # Pivot
        B, L, C, T = x.shape
        x = x.reshape(B, L * C, T)
        x = self.fuse(x)          # (B, F2, 1250)

        # Channel-mixed regime
        x = self.block2(x)        # (B, F2, 625)
        x = self.block3(x)        # (B, F3, 312)
        x = self.block4(x)        # (B, F4, 156)

        return self.gap(x).squeeze(-1)   # (B, F4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.fc(self.head_drop(feat))


@register_model("repnet_resnet_hybrid")
class RepNetResNetHybridModel(BaseModel):
    """Optuna-compatible wrapper around the 4-block hybrid ResNet.

    Stronger regularization defaults than the crosslead study used, since
    the channel-mixed convs add capacity that needs taming on ~1700 samples.
    """

    def __init__(
        self,
        f1=32,
        f2=64,
        f3=128,
        f4=128,
        wide_kernel=7,
        narrow_kernel=5,
        narrow_kernel_2=3,
        dropout=0.2,
        lr=5e-4,
        weight_decay=1e-3,
        batch_size=64,
        epochs=50,
        loss_fn="weighted",
        focal_gamma=2.0,
        focal_alpha=0.25,
        **kwargs,
    ):
        self.net_params = dict(
            f1=f1, f2=f2, f3=f3, f4=f4,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            narrow_kernel_2=narrow_kernel_2,
            dropout=dropout,
        )
        self.lr = lr
        self.weight_decay = weight_decay
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
        logger.info("RepNetResNetHybrid using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "dropout": trial.suggest_float("resnet_dropout", 0.1, 0.4),
            "lr": trial.suggest_float("resnet_lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("resnet_wd", 1e-4, 5e-3, log=True),
        }

    def _build_criterion(self, y_train: np.ndarray):
        if self.loss_fn == "weighted":
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            weight = torch.tensor(
                [1.0, n_neg / max(n_pos, 1)], dtype=torch.float32,
            ).to(self.device)
            return nn.CrossEntropyLoss(weight=weight)
        if self.loss_fn == "focal":
            return FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        return nn.CrossEntropyLoss()

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = RepNetResNetHybrid(**self.net_params).to(self.device)

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
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        patience = 10

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
