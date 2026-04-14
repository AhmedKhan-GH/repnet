"""Butler et al. ECG-AI 1D Residual CNN.

Faithfully implements the architecture from the paper:
- 6 residual blocks across 3 stages (16, 32, 64 filters)
- Each block: Conv1D(k=3) -> BN -> LeakyReLU -> Conv1D(k=3) -> BN -> Add(skip) -> LeakyReLU -> MaxPool -> Dropout
- Blocks 3 and 5 use 1x1 projection convolutions for channel matching
- Input: (batch, 12, 2250)
- Head: Flatten -> Dense(2) -> Softmax
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Single residual block matching ECG-AI architecture.

    Conv1D(k=3) -> BN -> LeakyReLU -> Conv1D(k=3) -> BN -> Add(skip) -> LeakyReLU -> MaxPool -> Dropout

    When in_channels != out_channels, a 1x1 projection is used on the skip path.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        # 1x1 projection when channel count changes (blocks 3 and 5 in the paper)
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class ECGAIResNet(nn.Module):
    """Butler et al. ECG-AI architecture.

    3 stages x 2 blocks each = 6 residual blocks.
    Stage 1: 16 filters, Stage 2: 32 filters, Stage 3: 64 filters.

    Optuna can tune: kernel_size, dropout, and stage filter widths.
    """

    def __init__(
        self,
        n_leads: int = 12,
        stage_filters: tuple[int, ...] = (16, 32, 64),
        blocks_per_stage: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        layers = []
        in_ch = n_leads

        for stage_ch in stage_filters:
            for _ in range(blocks_per_stage):
                layers.append(ResBlock(in_ch, stage_ch, kernel_size, dropout))
                in_ch = stage_ch

        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(n_classes),
        )

    def forward(self, x):
        # x: (batch, n_leads, seq_len)
        x = self.blocks(x)
        x = self.head(x)
        return x


@register_model("ecg_ai_resnet")
class ECGAIResNetModel(BaseModel):
    """Optuna-compatible wrapper around the ECG-AI ResNet."""

    def __init__(self, stage_filters=(16, 32, 64), kernel_size=3, dropout=0.1,
                 lr=1e-3, batch_size=32, epochs=30, **kwargs):
        self.net_params = dict(
            stage_filters=stage_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        logger.info("Using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        # Keep close to the paper defaults (16, 32, 64) with small variations
        s1 = trial.suggest_categorical("resnet_stage1_filters", [16, 32])
        s2 = trial.suggest_categorical("resnet_stage2_filters", [32, 64])
        s3 = trial.suggest_categorical("resnet_stage3_filters", [64, 128])

        return {
            "stage_filters": (s1, s2, s3),
            "kernel_size": trial.suggest_categorical("resnet_kernel_size", [3, 5]),
            "dropout": trial.suggest_float("resnet_dropout", 0.05, 0.3),
            "lr": trial.suggest_float("resnet_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("resnet_batch_size", [32, 64]),
            "epochs": trial.suggest_int("resnet_epochs", 10, 30),
        }

    def _to_tensors(self, X, y=None):
        Xt = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.long)
            return Xt, yt
        return Xt

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = ECGAIResNet(**self.net_params).to(self.device)

        # Lazy init — run one dummy forward pass to materialize LazyLinear
        with torch.no_grad():
            dummy = torch.zeros(1, X_train.shape[1], X_train.shape[2], device=self.device)
            self.model(dummy)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        Xt, yt = self._to_tensors(X_train, y_train)
        train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        patience = 5

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            val_auc = self.score(X_val, y_val)
            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            print(
                f"\r  Epoch {epoch+1:2d}/{self.epochs} | loss={avg_loss:.4f} "
                f"| val_AUROC={val_auc:.4f} | best={best_val_auc:.4f}{marker}",
                end="", flush=True,
            )

            if patience_counter >= patience:
                print(f" | early stop", flush=True)
                break
        else:
            print(flush=True)  # newline after final epoch

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        Xt = self._to_tensors(X).to(self.device)
        dl = DataLoader(TensorDataset(Xt), batch_size=self.batch_size)
        probs = []
        for (xb,) in dl:
            logits = self.model(xb.to(self.device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)
