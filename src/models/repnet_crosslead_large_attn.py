"""Depth-3 per-lead CNN with 3 interleaved 8-token cross-lead attention gates.

Builds on the RepNet crosslead Optuna study (best lr=8.76e-4, dropout=0.0636) by:
  - Adding a third per-lead conv stage (depth-2 → depth-3, kernels 7 → 5 → 3)
  - Adding a third cross-lead attention layer (one after each conv stage)
  - Replacing the original single-vector cross-lead pool with an 8-patch
    Pre-LN transformer block (richer attention signal at each stage)

Architecture:

  Per-lead conv trunk (3 stages, weights shared across the 12 leads):
    conv1 (1   → F1, k=7) → attn1 (8 patches, d=F1)
    conv2 (F1  → F2, k=5) → attn2 (8 patches, d=F2)
    conv3 (F2  → F3, k=3) → attn3 (8 patches, d=F3)

  Each PatchCrossLeadAttentionGate:
    AdaptiveAvgPool1d(8) per lead → (B, 12, C, 8)
    + learnable pos_emb(8, C) and lead_emb(12, C)
    Reshape (B, 8, 12, C) → (B*8, 12, C); cross-lead Pre-LN MHA + MLP; reshape back
    Linear(C → C) + sigmoid → gate (B, 12, C, 8)
    F.interpolate gate to T_conv (linear) → multiply with conv stream

  Fusion (matches repnet_crosslead's head — held fixed in the study):
    Concatenate 12 leads → Conv1d(12*F3 → F3, k=1) → BN → ReLU → GAP →
    Dropout → Linear(F3, 2)
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_baseline import FocalLoss
from .repnet_baseline_large import PerLeadConvBlock

logger = logging.getLogger(__name__)


class PatchCrossLeadAttentionGate(nn.Module):
    """8-patch cross-lead attention used as a multiplicative gate on the conv stream.

    Input/output shape: (B, L, C, T_conv) — the conv stream's temporal resolution
    is preserved so the next conv stage receives full-length inputs.

    Internally the input is pooled to (B, L, C, n_tokens), runs a Pre-LN
    transformer block with the L=12 leads as attention tokens at each of the
    n_tokens positions, then is upsampled back to T_conv as a sigmoid gate.
    """

    def __init__(
        self,
        d_model: int,
        n_leads: int = 12,
        n_tokens: int = 8,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0636,
        attn_dropout: float = 0.0636,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_leads = n_leads

        self.token_pool = nn.AdaptiveAvgPool1d(n_tokens)

        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, 1, d_model))
        self.lead_emb = nn.Parameter(torch.zeros(1, 1, n_leads, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.lead_emb, std=0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, T_conv)
        B, L, C, T_conv = x.shape

        # Pool each lead's temporal axis to n_tokens patches
        z = self.token_pool(x.reshape(B * L, C, T_conv))   # (B*L, C, n_tokens)
        z = z.reshape(B, L, C, self.n_tokens)
        z = z.permute(0, 3, 1, 2).contiguous()             # (B, n_tokens, L, C)

        # Add learnable position + lead embeddings
        z = z + self.pos_emb + self.lead_emb

        # Cross-lead attention at each of n_tokens patches
        z_flat = z.reshape(B * self.n_tokens, L, C)
        h = self.norm1(z_flat)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        z_flat = z_flat + attn_out
        h = self.norm2(z_flat)
        z_flat = z_flat + self.mlp(h)
        z = z_flat.reshape(B, self.n_tokens, L, C)         # (B, T, L, C)

        # Build gate: project then sigmoid
        gate = torch.sigmoid(self.gate_proj(z))            # (B, T, L, C)
        gate = gate.permute(0, 2, 3, 1).contiguous()       # (B, L, C, T)

        # Upsample gate to T_conv along temporal axis
        gate_full = F.interpolate(
            gate.reshape(B * L, C, self.n_tokens),
            size=T_conv, mode="linear", align_corners=False,
        ).reshape(B, L, C, T_conv)

        return x * gate_full


class RepNetCrossLeadLargeAttn(nn.Module):
    """Per-lead CNN with N interleaved cross-lead attention gates (variable depth).

    Filters double per stage starting at `base_filters`. Pass `stage_filters`
    explicitly to override (back-compat with the original benchmark config).
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_layers: int = 3,
        base_filters: int = 32,
        stage_filters: tuple[int, ...] | None = None,
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        dropout: float = 0.06355381998641418,
        n_tokens: int = 8,
        n_attn_heads: int = 4,
        attn_mlp_ratio: int = 4,
        attn_dropout: float = 0.06355381998641418,
        n_classes: int = 2,
        **kwargs,
    ):
        super().__init__()
        if stage_filters is None:
            stage_filters = tuple(base_filters * (2 ** i) for i in range(n_layers))
        else:
            stage_filters = tuple(stage_filters)
            n_layers = len(stage_filters)

        self.n_leads = n_leads
        self.n_layers = n_layers
        self.stage_filters = stage_filters

        # Per-lead conv + cross-lead attention gate at each stage
        self.convs = nn.ModuleList()
        self.attns = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(stage_filters):
            kernel = wide_kernel if i == 0 else narrow_kernel
            self.convs.append(PerLeadConvBlock(in_ch, out_ch, kernel, dropout))
            self.attns.append(PatchCrossLeadAttentionGate(
                d_model=out_ch, n_leads=n_leads, n_tokens=n_tokens,
                n_heads=n_attn_heads, mlp_ratio=attn_mlp_ratio,
                dropout=dropout, attn_dropout=attn_dropout,
            ))
            in_ch = out_ch

        # Fusion: concat leads -> pointwise conv -> GAP -> head
        last_ch = stage_filters[-1]
        self.fuse = nn.Sequential(
            nn.Conv1d(n_leads * last_ch, last_ch, kernel_size=1),
            nn.BatchNorm1d(last_ch),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(last_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 2500)
        x = x.unsqueeze(2)        # (B, 12, 1, T)
        for conv, attn in zip(self.convs, self.attns):
            x = conv(x)
            x = attn(x)
        B, L, F_, T_out = x.shape
        x = x.reshape(B, L * F_, T_out)
        x = self.fuse(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.head_drop(x))


@register_model("repnet_crosslead_large_attn")
class RepNetCrossLeadLargeAttnModel(BaseModel):
    """Optuna-compatible wrapper around the depth-3 + 3-attention crosslead model."""

    def __init__(
        self,
        n_layers=3,
        base_filters=32,
        stage_filters=None,
        wide_kernel=7,
        narrow_kernel=5,
        dropout=0.06355381998641418,
        n_tokens=8,
        n_attn_heads=4,
        attn_mlp_ratio=4,
        attn_dropout=0.06355381998641418,
        lr=0.0008756917546352803,
        batch_size=64,
        epochs=50,
        loss_fn="weighted",
        focal_gamma=2.0,
        focal_alpha=0.25,
        **kwargs,
    ):
        self.net_params = dict(
            n_layers=n_layers,
            base_filters=base_filters,
            stage_filters=stage_filters,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            dropout=dropout,
            n_tokens=n_tokens,
            n_attn_heads=n_attn_heads,
            attn_mlp_ratio=attn_mlp_ratio,
            attn_dropout=attn_dropout,
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
        logger.info("RepNetCrossLeadLargeAttn using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        # Architectural search over depth + attention granularity. lr/dropout
        # stay fixed at the crosslead Optuna-study values.
        return {
            "n_layers": trial.suggest_int("n_layers", 2, 4),
            "n_tokens": trial.suggest_categorical("n_tokens", [4, 8, 16]),
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
        self.model = RepNetCrossLeadLargeAttn(**self.net_params).to(self.device)

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
