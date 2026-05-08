"""InceptionTime-1D for 12-lead ECG.

Reference:
  Ismail Fawaz et al., "InceptionTime: Finding AlexNet for time series
  classification," Data Mining and Knowledge Discovery 34 (2020), 1936–1962.

Architecture (this implementation, sized for ~80K params):
  Input: (B, 12, T) — 12 leads as input channels.

  Each InceptionBlock has 4 parallel branches:
    - Bottleneck (1x1, in→32) → Conv1d(k=9,  32→32)
    - Bottleneck (1x1, in→32) → Conv1d(k=19, 32→32)
    - Bottleneck (1x1, in→32) → Conv1d(k=39, 32→32)
    - MaxPool1d(3) → Conv1d(1x1, in→32)
  → Concat to 128 channels → BN → ReLU.
  Residual connection (1x1 projecting from in_ch → 128) every block.

  3 InceptionBlocks (stride-2 max-pool between blocks) → GAP → Dropout → Linear(2).

Kernel sizes are odd so that `padding = k // 2` exactly preserves length under
stride-1 convs (with even kernels, that padding rule produces T+1 outputs, which
breaks concatenation across branches). The triple (9, 19, 39) at 250 Hz covers
36 ms / 76 ms / 156 ms — sub-QRS, QRS-scale, and early-T-wave scale, all at every
depth. This matches the original InceptionTime convention (Ismail Fawaz et al.
use 10, 20, 40 minus 1).

Total params at default config (n_filters=32): ~270K. Drop n_filters to 16 for
a tighter ~70K model — this puts param count in the same band as the smallest
RepNet variants on this dataset.
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


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters:   int = 32,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        # Pool branch: pool, then 1x1 conv from in_channels (NOT bottlenecked)
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1, bias=False),
        )

        self.out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        outs = [conv(b) for conv in self.conv_branches]
        outs.append(self.pool_branch(x))
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(out))


class InceptionStack(nn.Module):
    """A stack of InceptionBlocks with a residual every `residual_every` blocks."""

    def __init__(
        self,
        in_channels:    int = 12,
        n_blocks:       int = 3,
        n_filters:      int = 32,
        kernel_sizes:   tuple[int, ...] = (9, 19, 39),
        residual_every: int = 1,
        downsample_every: int = 1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.pools = nn.ModuleList()
        block_out = n_filters * (len(kernel_sizes) + 1)

        cur_in = in_channels
        for i in range(n_blocks):
            self.blocks.append(InceptionBlock(cur_in, n_filters, kernel_sizes))
            if (i + 1) % residual_every == 0:
                self.residuals.append(nn.Sequential(
                    nn.Conv1d(cur_in, block_out, 1, bias=False),
                    nn.BatchNorm1d(block_out),
                ))
            else:
                self.residuals.append(None)  # type: ignore[arg-type]
            self.pools.append(
                nn.MaxPool1d(2, 2) if (i + 1) % downsample_every == 0 else nn.Identity()
            )
            cur_in = block_out

        self.out_channels = block_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each block consumes the previous block's post-pool output and emits the
        # next post-pool output. Residuals (when present) project the block's input
        # through a 1x1 conv to match channel count; temporal length is preserved
        # within the block (the InceptionBlock convs use padding), so the residual
        # sum happens before the pool. Then we pool both.
        for blk, res, pool in zip(self.blocks, self.residuals, self.pools):
            out = blk(x)
            if res is not None:
                out = out + res(x)
            x = pool(out)
        return x


class InceptionTime1D(nn.Module):
    """InceptionTime-1D classifier for 12-lead ECG.

    Default config is sized for ~80K params. Tune `n_filters` to scale capacity.
    """

    def __init__(
        self,
        n_leads:      int = 12,
        n_blocks:     int = 3,
        n_filters:    int = 32,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        dropout:      float = 0.1,
        n_classes:    int = 2,
    ):
        super().__init__()
        self.stack = InceptionStack(
            in_channels=n_leads,
            n_blocks=n_blocks,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            residual_every=1,
            downsample_every=1,
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.stack.out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        x = self.stack(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("inception_time")
class InceptionTime1DModel(BaseModel):
    """Optuna-compatible wrapper for InceptionTime-1D."""

    def __init__(
        self,
        n_blocks:     int = 3,
        n_filters:    int = 32,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        dropout:      float = 0.1,
        lr:           float = 1e-3,
        batch_size:   int = 64,
        epochs:       int = 50,
        weight_decay: float = 1e-4,
        loss_fn:      str = "cross_entropy",
        **kwargs,
    ):
        self.net_params = dict(
            n_blocks=n_blocks,
            n_filters=n_filters,
            kernel_sizes=tuple(kernel_sizes),
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
        logger.info("InceptionTime1D using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "n_blocks":     trial.suggest_int("inception_n_blocks", 2, 5),
            "n_filters":    trial.suggest_categorical("inception_n_filters", [16, 24, 32, 48]),
            "kernel_sizes": trial.suggest_categorical(
                "inception_kernels", [(9, 19, 39), (7, 15, 31), (11, 23, 47), (9, 29, 59)]
            ),
            "dropout":      trial.suggest_float("inception_dropout", 0.05, 0.4),
            "lr":           trial.suggest_float("inception_lr", 1e-4, 5e-3, log=True),
            "batch_size":   trial.suggest_categorical("inception_batch_size", [32, 64]),
            "weight_decay": trial.suggest_float("inception_weight_decay", 1e-5, 1e-2, log=True),
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
        self.model = InceptionTime1D(**self.net_params).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("InceptionTime1D parameters: %d", n_params)

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
