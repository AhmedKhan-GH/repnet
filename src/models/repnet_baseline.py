"""Simple 4-block 1D Residual CNN — project baseline for comparison with ECG-AI.

Architecture (channels-first: batch, channels, time):

  Block 1 (12 → 32, k=7): Conv→BN→ReLU→Conv→BN, proj skip, Add→ReLU→MaxPool(2)→Dropout
  Block 2 (32 → 32, k=7): Conv→BN→ReLU→Conv→BN, identity skip, Add→ReLU→MaxPool(2)→Dropout
  Block 3 (32 → 64, k=5): Conv→BN→ReLU→Conv→BN, proj skip, Add→ReLU→MaxPool(2)→Dropout
  Block 4 (64 → 64, k=5): Conv→BN→ReLU→Conv→BN, identity skip, Add→ReLU→MaxPool(2)→Dropout

  GlobalAveragePool → Dropout → Linear(n_classes)

Design choices motivated by EDA:
  - Wider kernels (7, 5) capture low-frequency (2 Hz) discriminative content found in PSD analysis
  - Global average pooling instead of flatten — better generalisation on small dataset (369 samples)
  - 4 blocks vs ECG-AI's 6 — lighter model for direct comparison
  - Standard ReLU (not LeakyReLU) to keep it architecturally distinct from ECG-AI
"""

import logging

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


class ResBlock1D(nn.Module):
    """Standard 2-conv residual block for 1D signals.

    Uses a 1x1 projection skip when in_channels != out_channels.
    No stride — downsampling is handled by the following MaxPool.
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
        residual = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + residual)
        x = self.dropout(self.pool(x))
        return x


class RepNet(nn.Module):
    """Configurable 1D ResNet baseline with global average pooling head.

    n_blocks=3: 12→32(k=7), 32→64(k=5), 64→64(k=5)  — ~72K params
    n_blocks=4: 12→32(k=7), 32→32(k=7), 32→64(k=5), 64→64(k=5)  — ~100K params
    """

    def __init__(
        self,
        n_leads: int = 12,
        stage_filters: tuple[int, int] = (32, 64),
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        dropout: float = 0.1,
        n_classes: int = 2,
        n_blocks: int = 4,
    ):
        super().__init__()
        f1, f2 = stage_filters
        if n_blocks == 3:
            # 3-block: one wide, then two narrow
            blocks = [
                ResBlock1D(n_leads, f1, wide_kernel, dropout),
                ResBlock1D(f1, f2, narrow_kernel, dropout),
                ResBlock1D(f2, f2, narrow_kernel, dropout),
            ]
        else:
            # 4-block: two wide, then two narrow
            blocks = [
                ResBlock1D(n_leads, f1, wide_kernel, dropout),
                ResBlock1D(f1, f1, wide_kernel, dropout),
                ResBlock1D(f1, f2, narrow_kernel, dropout),
                ResBlock1D(f2, f2, narrow_kernel, dropout),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f2, n_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_baseline")
class RepNetBaselineModel(BaseModel):
    """Optuna-compatible wrapper around the RepNet 1D ResNet baseline."""

    def __init__(self, stage_filters=(32, 64), wide_kernel=7, narrow_kernel=5,
                 dropout=0.1, n_blocks=4, lr=1e-3, batch_size=32, epochs=50,
                 loss_fn="weighted", focal_gamma=2.0, focal_alpha=0.25,
                 **kwargs):
        self.net_params = dict(
            stage_filters=stage_filters,
            n_blocks=n_blocks,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            dropout=dropout,
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
        logger.info("RepNetBaseline using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        f1 = trial.suggest_categorical("repnet_stage1_filters", [16, 32])
        f2 = trial.suggest_categorical("repnet_stage2_filters", [32, 64])
        loss_fn = trial.suggest_categorical(
            "repnet_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": (f1, f2),
            "n_blocks": trial.suggest_categorical("repnet_n_blocks", [3, 4]),
            "wide_kernel": trial.suggest_categorical("repnet_wide_kernel", [5, 7, 9]),
            "narrow_kernel": trial.suggest_categorical("repnet_narrow_kernel", [3, 5]),
            "dropout": trial.suggest_float("repnet_dropout", 0.05, 0.4),
            "lr": trial.suggest_float("repnet_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("repnet_batch_size", [32, 64]),
            "epochs": trial.suggest_int("repnet_epochs", 10, 50),
            "loss_fn": loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("repnet_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("repnet_focal_alpha", 0.1, 0.9)
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
        self.model = RepNet(**self.net_params).to(self.device)

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
