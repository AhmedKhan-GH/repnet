"""RepNet with Squeeze-and-Excitation attention between residual blocks.

Same 3-block 1D ResNet backbone as RepNet baseline, with SE blocks inserted
after each ResBlock to recalibrate feature channel importance.

Architecture:
  ResBlock(12→32, k=7) → SE(32) → ResBlock(32→64, k=5) → SE(64) → ResBlock(64→64, k=5) → SE(64) → GAP → Dropout → Linear(2)

SE block (Hu et al. 2018):
  1. Squeeze: AdaptiveAvgPool1d → (batch, C, 1)
  2. Excite:  Linear(C, C//r) → ReLU → Linear(C//r, C) → Sigmoid
  3. Scale:   element-wise multiply with input feature map

EDA justification (Step 9): lead discriminability CV=2.236 indicates highly variable
channel importance. While SE operates on learned feature channels (not raw leads),
it enables the network to dynamically weight which feature representations matter
at each depth level.
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_baseline import ResBlock1D, FocalLoss

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps.

    Learns per-channel importance weights via a bottleneck FC layer.
    Reduction ratio r controls bottleneck width (lower r = more params).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, C, T)
        w = self.squeeze(x).squeeze(-1)   # (batch, C)
        w = self.excite(w).unsqueeze(-1)   # (batch, C, 1)
        return x * w


class RepNetAttention(nn.Module):
    """3-block 1D ResNet with SE attention after each block."""

    def __init__(
        self,
        n_leads: int = 12,
        stage_filters: tuple[int, int] = (32, 64),
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        dropout: float = 0.1,
        se_reduction: int = 4,
        n_classes: int = 2,
        n_blocks: int = 3,
    ):
        super().__init__()
        f1, f2 = stage_filters
        if n_blocks == 3:
            block_cfgs = [
                (n_leads, f1, wide_kernel),
                (f1, f2, narrow_kernel),
                (f2, f2, narrow_kernel),
            ]
        else:
            block_cfgs = [
                (n_leads, f1, wide_kernel),
                (f1, f1, wide_kernel),
                (f1, f2, narrow_kernel),
                (f2, f2, narrow_kernel),
            ]

        layers = []
        for in_ch, out_ch, k in block_cfgs:
            layers.append(ResBlock1D(in_ch, out_ch, k, dropout))
            layers.append(SEBlock(out_ch, se_reduction))
        self.blocks = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f2, n_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_attention")
class RepNetAttentionModel(BaseModel):
    """Optuna-compatible wrapper around RepNet with SE attention."""

    def __init__(self, stage_filters=(32, 64), wide_kernel=7, narrow_kernel=5,
                 dropout=0.1, se_reduction=4, n_blocks=3,
                 lr=1e-3, batch_size=32, epochs=50,
                 loss_fn="weighted", focal_gamma=2.0, focal_alpha=0.25,
                 **kwargs):
        self.net_params = dict(
            stage_filters=stage_filters,
            n_blocks=n_blocks,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            dropout=dropout,
            se_reduction=se_reduction,
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
        logger.info("RepNetAttention using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        f1 = trial.suggest_categorical("repnet_attn_stage1_filters", [16, 32])
        f2 = trial.suggest_categorical("repnet_attn_stage2_filters", [32, 64])
        loss_fn = trial.suggest_categorical(
            "repnet_attn_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": (f1, f2),
            "n_blocks": trial.suggest_categorical("repnet_attn_n_blocks", [3, 4]),
            "wide_kernel": trial.suggest_categorical("repnet_attn_wide_kernel", [5, 7, 9]),
            "narrow_kernel": trial.suggest_categorical("repnet_attn_narrow_kernel", [3, 5]),
            "dropout": trial.suggest_float("repnet_attn_dropout", 0.05, 0.4),
            "se_reduction": trial.suggest_categorical("repnet_attn_se_reduction", [2, 4, 8]),
            "lr": trial.suggest_float("repnet_attn_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("repnet_attn_batch_size", [32, 64]),
            "epochs": trial.suggest_int("repnet_attn_epochs", 10, 50),
            "loss_fn": loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("repnet_attn_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("repnet_attn_focal_alpha", 0.1, 0.9)
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
        self.model = RepNetAttention(**self.net_params).to(self.device)

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
