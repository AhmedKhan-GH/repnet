"""Per-lead 1D ResNet baseline with shared weights and lead-attention head.

Architecture (designed from EDA findings):
  Input (B, 12, 2500)
    flatten leads → (B*12, 1, 2500)                 # weights shared across leads
    Stem: Conv1d(1→32, k=15, stride=2) → BN → ReLU  # k=15 captures QRS (~60 ms @ 250 Hz)
    ResBlock1D × 4: 32→64→128→256→256, stride 2,2,2,1
    GlobalAvgPool → (B*12, 256)
    reshape → (B, 12, 256)
    LeadAttention (softmax over 12) → (B, 256)
    Dropout → Linear(256 → 2)

Why this design:
  - Per-lead with shared weights: EDA Frobenius corr-diff = 1.026 → cross-lead 2D conv
    not earning extra params; shared weights = parameter efficiency on small dataset.
  - ResNet skip connections: more reliable training than plain CNN at this depth.
  - Stride-2 downsampling (vs MaxPool): keeps gradient flow cleaner.
  - Lead attention: EDA CV(−log10 p) across leads = 2.236 → leads contribute unequally.
  - No self/cross attention: ~1700 recordings is too few; defer to v2 if baseline plateaus.
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


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class ResBlock1D(nn.Module):
    """Standard 1D pre-activation residual block with optional stride-2 downsample.

    Input:  (N, C_in,  T)
    Output: (N, C_out, T // stride)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
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


class LeadAttention(nn.Module):
    """Softmax-weighted aggregation over 12 leads.

    Input:  (B, 12, C)
    Output: (B, C)  along with attention weights (B, 12) accessible via last_weights.
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
        # x: (B, 12, C)
        scores = self.score(x).squeeze(-1)       # (B, 12)
        weights = torch.softmax(scores, dim=1)   # (B, 12)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * x).sum(dim=1)   # (B, C)


class BaselineResNet1D(nn.Module):
    """Per-lead shared-weight 1D ResNet with lead-attention aggregation."""

    def __init__(
        self,
        n_leads:    int = 12,
        stem_channels: int = 32,
        block_channels: tuple[int, int, int, int] = (64, 128, 256, 256),
        block_strides:  tuple[int, int, int, int] = (2, 2, 2, 1),
        kernel_size:    int = 7,
        stem_kernel:    int = 15,
        dropout:        float = 0.3,
        n_classes:      int = 2,
    ):
        super().__init__()
        assert len(block_channels) == len(block_strides) == 4

        self.n_leads = n_leads
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=stem_kernel,
                      stride=2, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        in_c = stem_channels
        blocks = []
        for out_c, s in zip(block_channels, block_strides):
            blocks.append(ResBlock1D(in_c, out_c, kernel_size=kernel_size,
                                     stride=s, dropout=0.0))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.lead_attn = LeadAttention(in_c, hidden=64)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        B, L, T = x.shape
        assert L == self.n_leads, f"Expected {self.n_leads} leads, got {L}"

        # Flatten leads into batch — shared weights across all 12
        x = x.reshape(B * L, 1, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)        # (B*L, C)

        # Per-sample, per-lead embedding → lead attention
        C = x.shape[-1]
        x = x.reshape(B, L, C)
        x = self.lead_attn(x)              # (B, C)

        return self.fc(self.head_drop(x))  # logits (B, n_classes)


@register_model("baseline_resnet1d")
class BaselineResNet1DModel(BaseModel):
    """Optuna-compatible wrapper — matches RepNetBaselineModel's training pattern."""

    def __init__(
        self,
        stem_channels:   int = 32,
        block_channels:  tuple[int, int, int, int] = (64, 128, 256, 256),
        block_strides:   tuple[int, int, int, int] = (2, 2, 2, 1),
        kernel_size:     int = 7,
        stem_kernel:     int = 15,
        dropout:         float = 0.3,
        lr:              float = 1e-3,
        weight_decay:    float = 1e-4,
        batch_size:      int = 32,
        epochs:          int = 50,
        loss_fn:         str = "cross_entropy",
        focal_gamma:     float = 2.0,
        focal_alpha:     float = 0.25,
        patience:        int = 10,
        **kwargs,
    ):
        self.net_params = dict(
            stem_channels=stem_channels,
            block_channels=tuple(block_channels),
            block_strides=tuple(block_strides),
            kernel_size=kernel_size,
            stem_kernel=stem_kernel,
            dropout=dropout,
        )
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.loss_fn      = loss_fn
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha
        self.patience     = patience

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.history: dict | None = None
        logger.info("BaselineResNet1D using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        loss_fn = trial.suggest_categorical(
            "baseline_resnet1d_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stem_channels":  trial.suggest_categorical("baseline_resnet1d_stem_channels", [16, 32]),
            "block_channels": trial.suggest_categorical(
                "baseline_resnet1d_block_channels",
                [(32, 64, 128, 128), (64, 128, 256, 256), (32, 64, 128, 256)],
            ),
            "kernel_size":    trial.suggest_categorical("baseline_resnet1d_kernel_size", [5, 7, 9]),
            "stem_kernel":    trial.suggest_categorical("baseline_resnet1d_stem_kernel", [11, 15, 21]),
            "dropout":        trial.suggest_float("baseline_resnet1d_dropout", 0.1, 0.5),
            "lr":             trial.suggest_float("baseline_resnet1d_lr", 1e-4, 5e-3, log=True),
            "weight_decay":   trial.suggest_float("baseline_resnet1d_wd", 1e-5, 1e-3, log=True),
            "batch_size":     trial.suggest_categorical("baseline_resnet1d_batch_size", [32, 64]),
            "epochs":         trial.suggest_int("baseline_resnet1d_epochs", 20, 60),
            "loss_fn":        loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("baseline_resnet1d_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("baseline_resnet1d_focal_alpha", 0.1, 0.9)
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
        self.model = BaselineResNet1D(**self.net_params).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=self.weight_decay,
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

            if patience_counter >= self.patience:
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
