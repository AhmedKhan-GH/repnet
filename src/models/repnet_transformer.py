"""RepNet Transformer — per-lead CNN encoder + patch-level cross-lead transformer.

Architecture:
  1. Per-lead CNN encoder: 12 independent Conv1D streams reduce temporal dimension
     Input per lead: (1, 2500) → Conv blocks → (F, T') where T' << 2500
  2. Patch tokens: reshape each lead's feature map into P patches of dimension D
     12 leads × P patches = 12*P tokens, each of dimension D
  3. Transformer encoder: self-attention across all lead×patch tokens
     Learns "lead aVR at 0.3-0.5s matters for this patient"
  4. Classification: CLS token or mean pooling → Linear(2)

EDA justification:
  - Lead discriminability CV=2.236 (Step 9): per-lead attention needed
  - Temporal peak width ~full strip (Step 10): need global temporal context
  - PSD peak at 2 Hz (Step 8): CNN encoder with wide kernels captures this
  - Patch-level attention allows joint lead×time specificity
"""

import logging
import math

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_baseline import FocalLoss

logger = logging.getLogger(__name__)


class LeadEncoder(nn.Module):
    """Lightweight per-lead CNN that compresses (1, T) → (embed_dim, T').

    Two conv layers with MaxPool reduce the temporal dimension.
    Applied independently to each of 12 leads.
    """

    def __init__(self, embed_dim: int = 64, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, kernel_size, padding=p),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size, padding=p),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch * n_leads, 1, T)
        return self.net(x)  # (batch * n_leads, embed_dim, T')


class RepNetTransformer(nn.Module):
    """Per-lead CNN encoder → patch tokenization → transformer → classifier.

    Each lead's CNN output is split into P non-overlapping patches.
    12 leads × P patches = N_tokens. A learnable CLS token is prepended.
    Transformer self-attention operates over all N_tokens + CLS.
    """

    def __init__(
        self,
        n_leads: int = 12,
        embed_dim: int = 64,
        encoder_kernel: int = 7,
        n_patches: int = 10,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.n_patches = n_patches
        self.embed_dim = embed_dim

        # Per-lead CNN encoder
        self.lead_encoder = LeadEncoder(embed_dim, encoder_kernel, dropout)

        # Patch projection: stored for forward pass
        self.n_patches_val = n_patches

        # Learnable embeddings
        n_tokens = n_leads * n_patches + 1  # +1 for CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.lead_embed = nn.Parameter(torch.randn(1, n_leads, 1, embed_dim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(1, 1, n_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x):
        B, L, T = x.shape  # (batch, 12, 2500)

        # Per-lead CNN encoding
        x = x.reshape(B * L, 1, T)                    # (B*12, 1, 2500)
        x = self.lead_encoder(x)                        # (B*12, embed_dim, T')

        # Manual patch pooling (MPS-compatible — avoids AdaptiveAvgPool1d)
        # Truncate T' to be divisible by n_patches, then reshape + mean
        P = self.n_patches_val
        T_enc = x.shape[-1]
        T_use = (T_enc // P) * P
        x = x[:, :, :T_use]                            # (B*12, D, T_use)
        x = x.reshape(x.shape[0], self.embed_dim, P, T_use // P).mean(dim=-1)  # (B*12, D, P)
        x = x.reshape(B, L, self.embed_dim, P)         # (B, 12, D, P)
        x = x.permute(0, 1, 3, 2)                      # (B, 12, P, D)

        # Add lead + patch embeddings
        x = x + self.lead_embed + self.patch_embed      # broadcast: (B, 12, P, D)

        # Flatten to token sequence: (B, 12*P, D)
        x = x.reshape(B, L * self.n_patches, self.embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, 12*P+1, D)
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)                         # (B, 12*P+1, D)
        x = self.norm(x[:, 0])                          # CLS token: (B, D)

        return self.head(x)


@register_model("repnet_transformer")
class RepNetTransformerModel(BaseModel):
    """Optuna-compatible wrapper around RepNet Transformer."""

    def __init__(self, embed_dim=64, encoder_kernel=7, n_patches=10,
                 n_heads=4, n_layers=2, dropout=0.1,
                 lr=1e-3, batch_size=32, epochs=50,
                 loss_fn="weighted", focal_gamma=2.0, focal_alpha=0.25,
                 **kwargs):
        self.net_params = dict(
            embed_dim=embed_dim,
            encoder_kernel=encoder_kernel,
            n_patches=n_patches,
            n_heads=n_heads,
            n_layers=n_layers,
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
        logger.info("RepNetTransformer using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        embed_dim = trial.suggest_categorical("transformer_embed_dim", [32, 64])
        n_heads = trial.suggest_categorical("transformer_n_heads", [2, 4])
        loss_fn = trial.suggest_categorical(
            "transformer_loss_fn", ["cross_entropy", "weighted", "focal"]
        )
        params = {
            "embed_dim": embed_dim,
            "encoder_kernel": trial.suggest_categorical("transformer_encoder_kernel", [5, 7, 9]),
            "n_patches": trial.suggest_categorical("transformer_n_patches", [5, 10, 20]),
            "n_heads": n_heads,
            "n_layers": trial.suggest_categorical("transformer_n_layers", [1, 2, 3]),
            "dropout": trial.suggest_float("transformer_dropout", 0.05, 0.4),
            "lr": trial.suggest_float("transformer_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("transformer_batch_size", [32, 64]),
            "epochs": trial.suggest_int("transformer_epochs", 10, 50),
            "loss_fn": loss_fn,
        }
        if loss_fn == "focal":
            params["focal_gamma"] = trial.suggest_float("transformer_focal_gamma", 0.5, 5.0)
            params["focal_alpha"] = trial.suggest_float("transformer_focal_alpha", 0.1, 0.9)
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
        self.model = RepNetTransformer(**self.net_params).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
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
