"""Per-lead 1D Residual CNN (depth 3) + 8-token cross-lead attention.

Architecture:

  Per-lead conv trunk (3 stages, weights shared across the 12 leads):
    Block 1 (1 → F1, k=7):  Conv-BN-ReLU-Conv-BN + proj skip → ReLU → MaxPool(2) → Dropout
    Block 2 (F1 → F2, k=5): same shape
    Block 3 (F2 → F3, k=3): same shape

  Token compression:
    AdaptiveAvgPool1d(n_tokens) per lead → (B, 12, F3, T=n_tokens)
    permute → (B, 12, T, F3) = (B, L, T, D)

  Position + lead embeddings (learnable):
    pos_emb (T, D) broadcast over leads
    lead_emb (12, D) broadcast over tokens
    x = x + pos_emb + lead_emb

  Cross-lead attention block(s):
    Reshape (B, L, T, D) → (B*T, L, D) so each of T positions sees 12 leads as tokens
    Pre-LN MultiheadAttention(D, n_heads) + residual
    Pre-LN MLP(D → 4D → D, GELU, dropout) + residual
    Reshape back to (B, L, T, D)

  Head:
    mean over L → (B, T, D)
    mean over T → (B, D)
    Dropout → Linear(D, 2)

Receptive field: 52 samples (208 ms) per output position before pooling. After
AdaptiveAvgPool1d to 8 tokens, each token integrates ~1.25 s of the 10-s recording,
covering 1–2 cardiac beats; cross-lead attention then mixes the 12 leads at each token.
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

logger = logging.getLogger(__name__)


class CrossLeadAttentionBlock(nn.Module):
    """Pre-LN transformer block over the lead axis.

    Input/output: (B, T, L, D).  At each of T positions, the 12 leads attend to
    each other.  T is folded into batch so attention is purely cross-lead.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
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
        # x: (B, T, L, D)
        B, T, L, D = x.shape
        x_flat = x.reshape(B * T, L, D)

        h = self.norm1(x_flat)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x_flat = x_flat + attn_out

        h = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(h)

        return x_flat.reshape(B, T, L, D)


class RepNetLargeAttn(nn.Module):
    """Depth-3 per-lead CNN with 8-token cross-lead attention head."""

    def __init__(
        self,
        n_leads: int = 12,
        stage_filters: tuple[int, int, int] = (32, 64, 128),
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        narrow_kernel_2: int = 3,
        dropout: float = 0.1,
        n_tokens: int = 8,
        n_attn_blocks: int = 1,
        n_attn_heads: int = 4,
        attn_mlp_ratio: int = 4,
        attn_dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        f1, f2, f3 = stage_filters
        self.n_leads = n_leads
        self.n_tokens = n_tokens

        # Per-lead conv trunk
        self.conv1 = PerLeadConvBlock(1, f1, wide_kernel, dropout)
        self.conv2 = PerLeadConvBlock(f1, f2, narrow_kernel, dropout)
        self.conv3 = PerLeadConvBlock(f2, f3, narrow_kernel_2, dropout)

        # Per-lead temporal pool to n_tokens
        self.token_pool = nn.AdaptiveAvgPool1d(n_tokens)

        # Learnable position + lead embeddings (additive)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, 1, f3))
        self.lead_emb = nn.Parameter(torch.zeros(1, 1, n_leads, f3))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.lead_emb, std=0.02)

        # Cross-lead attention stack
        self.attn_blocks = nn.ModuleList([
            CrossLeadAttentionBlock(
                d_model=f3, n_heads=n_attn_heads,
                mlp_ratio=attn_mlp_ratio, dropout=attn_dropout,
            )
            for _ in range(n_attn_blocks)
        ])

        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 2500)
        x = x.unsqueeze(2)  # (B, 12, 1, T)

        x = self.conv1(x)   # (B, 12, F1, T/2)
        x = self.conv2(x)   # (B, 12, F2, T/4)
        x = self.conv3(x)   # (B, 12, F3, T/8)

        # Pool each lead's temporal axis to n_tokens
        B, L, F, T_conv = x.shape
        x = x.reshape(B * L, F, T_conv)
        x = self.token_pool(x)              # (B*L, F, n_tokens)
        x = x.reshape(B, L, F, self.n_tokens)
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, T, L, D)

        # Position + lead embeddings
        x = x + self.pos_emb + self.lead_emb

        for blk in self.attn_blocks:
            x = blk(x)

        # Pool: mean over leads then mean over tokens → (B, D)
        x = x.mean(dim=2)   # (B, T, D)
        x = x.mean(dim=1)   # (B, D)

        return self.fc(self.head_drop(x))


@register_model("repnet_baseline_large_attn")
class RepNetBaselineLargeAttnModel(BaseModel):
    """Optuna-compatible wrapper around the depth-3 + cross-lead attention model."""

    def __init__(self, stage_filters=(32, 64, 128), wide_kernel=7, narrow_kernel=5,
                 narrow_kernel_2=3, dropout=0.1,
                 n_tokens=8, n_attn_blocks=1, n_attn_heads=4,
                 attn_mlp_ratio=4, attn_dropout=0.1,
                 lr=1e-3, batch_size=64, epochs=50,
                 loss_fn="weighted", focal_gamma=2.0, focal_alpha=0.25,
                 **kwargs):
        self.net_params = dict(
            stage_filters=stage_filters,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            narrow_kernel_2=narrow_kernel_2,
            dropout=dropout,
            n_tokens=n_tokens,
            n_attn_blocks=n_attn_blocks,
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
        logger.info("RepNetBaselineLargeAttn using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        f1 = trial.suggest_categorical("repnet_attn_stage1_filters", [16, 32])
        f2 = trial.suggest_categorical("repnet_attn_stage2_filters", [32, 64])
        f3 = trial.suggest_categorical("repnet_attn_stage3_filters", [64, 128])
        loss_fn = trial.suggest_categorical(
            "repnet_attn_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "stage_filters": (f1, f2, f3),
            "wide_kernel": trial.suggest_categorical("repnet_attn_wide_kernel", [5, 7, 9]),
            "narrow_kernel": trial.suggest_categorical("repnet_attn_narrow_kernel", [3, 5]),
            "narrow_kernel_2": trial.suggest_categorical("repnet_attn_narrow_kernel_2", [3, 5]),
            "dropout": trial.suggest_float("repnet_attn_dropout", 0.05, 0.4),
            "n_tokens": trial.suggest_categorical("repnet_attn_n_tokens", [4, 8, 16]),
            "n_attn_blocks": trial.suggest_int("repnet_attn_n_blocks", 1, 3),
            "n_attn_heads": trial.suggest_categorical("repnet_attn_n_heads", [2, 4, 8]),
            "attn_dropout": trial.suggest_float("repnet_attn_attn_dropout", 0.0, 0.3),
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
        self.model = RepNetLargeAttn(**self.net_params).to(self.device)

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
        train_dl = DataLoader(
            TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True,
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
