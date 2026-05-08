"""RepNet CrossLead Hybrid: per-lead at the bottom, channel-mixed at the top.

Architectural philosophy:
  - Per-lead processing where lead-specific morphology lives (low scales)
  - Cross-lead attention exactly once, where lead identity is most discriminative
  - Channel-mixed convs above the attention pivot, where cross-lead correlations
    can be learned by ordinary 1D filters across all 12 leads at once
  - Temporal self-attention near the head to weight time windows

Forward pass (input x: (B, 12, 2500)):

  Regime 1 (per-lead, lead axis preserved):
    unsqueeze(2)                                   -> (B, 12, 1, 2500)
    conv1: PerLeadConvBlock(1 -> F1, k=wide_kernel)-> (B, 12, F1, 1250)
    attn1: PatchCrossLeadAttentionGate(d=F1, 8 patches)
                                                   -> (B, 12, F1, 1250)

  Pivot: collapse lead axis into channels
    reshape (B, 12, F1, 1250) -> (B, 12*F1, 1250)
    fuse: Conv1d(12*F1 -> F2, k=1) + BN + ReLU     -> (B, F2, 1250)

  Regime 2 (channel-mixed, no lead axis):
    conv2: ChannelMixedConvBlock(F2 -> F2, k=narrow_kernel)  -> (B, F2, 625)
    conv3: ChannelMixedConvBlock(F2 -> F3, k=narrow_kernel_2)-> (B, F3, 312)
    attn_temporal: TemporalSelfAttention(d=F3, 8 tokens)
                                                   -> (B, F3)

  Head:
    Dropout -> Linear(F3, 2)                       -> (B, 2)

Reuses:
  - PerLeadConvBlock    from repnet_baseline_large
  - PatchCrossLeadAttentionGate from repnet_crosslead_large_attn
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
from .repnet_crosslead_large_attn import PatchCrossLeadAttentionGate

logger = logging.getLogger(__name__)


class ChannelMixedConvBlock(nn.Module):
    """Standard 1D conv block over flat channels (no lead axis).

    Conv-BN-ReLU-Conv-BN, projection skip, ReLU, MaxPool(2), Dropout.
    Input/output: (B, C, T).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=p)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=p)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.dropout(self.pool(out))


class TemporalSelfAttention(nn.Module):
    """Self-attention over n_tokens temporal patches (no lead axis).

    Input:  (B, C, T)
    Output: (B, C) -- attention-weighted, mean-pooled across tokens.
    """

    def __init__(self, d_model: int, n_tokens: int = 8, n_heads: int = 4,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_pool = nn.AdaptiveAvgPool1d(n_tokens)

        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.token_pool(x)                  # (B, C, n_tokens)
        z = z.permute(0, 2, 1).contiguous()     # (B, n_tokens, C)
        z = z + self.pos_emb

        h = self.norm1(z)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        z = z + attn_out
        h = self.norm2(z)
        z = z + self.mlp(h)
        return z.mean(dim=1)                    # (B, C)


class RepNetCrossLeadHybrid(nn.Module):
    """Per-lead conv -> cross-lead attn -> channel-mixed convs -> temporal attn -> head."""

    def __init__(
        self,
        n_leads: int = 12,
        f1: int = 32,
        f2: int = 64,
        f3: int = 128,
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        narrow_kernel_2: int = 3,
        dropout: float = 0.06355381998641418,
        n_tokens: int = 8,
        n_attn_heads: int = 4,
        attn_mlp_ratio: int = 4,
        attn_dropout: float = 0.06355381998641418,
        use_temporal_attn: bool = True,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.use_temporal_attn = use_temporal_attn

        # Regime 1: per-lead
        self.conv1 = PerLeadConvBlock(1, f1, wide_kernel, dropout)
        self.attn1 = PatchCrossLeadAttentionGate(
            d_model=f1, n_leads=n_leads, n_tokens=n_tokens,
            n_heads=n_attn_heads, mlp_ratio=attn_mlp_ratio,
            dropout=dropout, attn_dropout=attn_dropout,
        )

        # Pivot: 1x1 projection from concat'd leads to f2 channels
        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * f1, f2, kernel_size=1),
            nn.BatchNorm1d(f2),
            nn.ReLU(inplace=True),
        )

        # Regime 2: channel-mixed
        self.conv2 = ChannelMixedConvBlock(f2, f2, narrow_kernel, dropout)
        self.conv3 = ChannelMixedConvBlock(f2, f3, narrow_kernel_2, dropout)

        if use_temporal_attn:
            self.attn_temporal = TemporalSelfAttention(
                d_model=f3, n_tokens=n_tokens, n_heads=n_attn_heads,
                mlp_ratio=attn_mlp_ratio, dropout=attn_dropout,
            )
            self.gap = None
        else:
            self.attn_temporal = None
            self.gap = nn.AdaptiveAvgPool1d(1)

        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 2500)
        x = x.unsqueeze(2)        # (B, 12, 1, 2500)

        # Per-lead regime
        x = self.conv1(x)         # (B, 12, F1, 1250)
        x = self.attn1(x)         # (B, 12, F1, 1250)

        # Pivot: collapse lead axis into channels
        B, L, C, T = x.shape
        x = x.reshape(B, L * C, T)
        x = self.fuse(x)          # (B, F2, T)

        # Channel-mixed regime
        x = self.conv2(x)         # (B, F2, T/2)
        x = self.conv3(x)         # (B, F3, T/4)

        if self.attn_temporal is not None:
            x = self.attn_temporal(x)         # (B, F3)
        else:
            x = self.gap(x).squeeze(-1)       # (B, F3)

        return self.fc(self.head_drop(x))


@register_model("repnet_crosslead_hybrid")
class RepNetCrossLeadHybridModel(BaseModel):
    """Optuna-compatible wrapper around RepNetCrossLeadHybrid."""

    def __init__(
        self,
        f1=32,
        f2=64,
        f3=128,
        wide_kernel=7,
        narrow_kernel=5,
        narrow_kernel_2=3,
        dropout=0.06355381998641418,
        n_tokens=8,
        n_attn_heads=4,
        attn_mlp_ratio=4,
        attn_dropout=0.06355381998641418,
        use_temporal_attn=True,
        lr=0.0008756917546352803,
        batch_size=64,
        epochs=50,
        loss_fn="weighted",
        focal_gamma=2.0,
        focal_alpha=0.25,
        **kwargs,
    ):
        self.net_params = dict(
            f1=f1, f2=f2, f3=f3,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            narrow_kernel_2=narrow_kernel_2,
            dropout=dropout,
            n_tokens=n_tokens,
            n_attn_heads=n_attn_heads,
            attn_mlp_ratio=attn_mlp_ratio,
            attn_dropout=attn_dropout,
            use_temporal_attn=use_temporal_attn,
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
        logger.info("RepNetCrossLeadHybrid using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "n_tokens": trial.suggest_categorical("hybrid_n_tokens", [4, 8, 16]),
            "dropout": trial.suggest_float("hybrid_dropout", 0.05, 0.40),
            "lr": trial.suggest_float("hybrid_lr", 1e-4, 5e-3, log=True),
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
        self.model = RepNetCrossLeadHybrid(**self.net_params).to(self.device)

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
