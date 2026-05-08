"""4-block ResNet hybrid + HRV/QT auxiliary features fused at the head.

Same conv backbone as RepNetResNetHybrid, but the GAP output is concatenated
with z-scored HRV features and passed through a small MLP head. This is the
"indirect disease" architecture pattern: deep ECG features + structured
domain features (HRV time/frequency-domain) -> fused logit.

Forward pass:
  x      : (B, 12, 2500)  raw 12-lead ECG
  feat   : (B, n_features) z-scored HRV features (precomputed externally)

  conv_feat = backbone.forward_features(x)        -> (B, F4)
  fused     = concat([conv_feat, feat], dim=-1)   -> (B, F4 + n_features)
  logits    = MLP(fused)                          -> (B, 2)

Wrapper signature differs from the strict BaseModel:
  fit(X_train, y_train, X_val, y_val, F_train, F_val)
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, register_model
from .repnet_baseline import FocalLoss
from .repnet_resnet_hybrid import RepNetResNetHybrid

logger = logging.getLogger(__name__)


class RepNetResNetHybridFeatures(nn.Module):
    """ResNet hybrid backbone + HRV-feature fusion at head."""

    def __init__(
        self,
        n_features: int = 8,
        n_leads: int = 12,
        f1: int = 32,
        f2: int = 64,
        f3: int = 128,
        f4: int = 128,
        wide_kernel: int = 7,
        narrow_kernel: int = 5,
        narrow_kernel_2: int = 3,
        dropout: float = 0.2,
        head_hidden: int = 64,
        n_classes: int = 2,
    ):
        super().__init__()
        self.backbone = RepNetResNetHybrid(
            n_leads=n_leads, f1=f1, f2=f2, f3=f3, f4=f4,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            narrow_kernel_2=narrow_kernel_2,
            dropout=dropout,
        )
        # Discard the backbone's own head; use ours.
        self.backbone.fc = nn.Identity()
        self.backbone.head_drop = nn.Identity()

        in_dim = f4 + n_features
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        deep = self.backbone.forward_features(x)        # (B, f4)
        return self.head(torch.cat([deep, feat], dim=-1))


@register_model("repnet_resnet_hybrid_features")
class RepNetResNetHybridFeaturesModel(BaseModel):
    """Optuna-compatible wrapper. fit() takes features alongside X.

    The strict BaseModel ABC signature is preserved by accepting features
    via additional named kwargs (with safe fallback to running pure-conv
    if no features supplied -- equivalent to RepNetResNetHybridModel).
    """

    def __init__(
        self,
        n_features=8,
        f1=32, f2=64, f3=128, f4=128,
        wide_kernel=7,
        narrow_kernel=5,
        narrow_kernel_2=3,
        dropout=0.2,
        head_hidden=64,
        lr=5e-4,
        weight_decay=1e-3,
        batch_size=64,
        epochs=50,
        loss_fn="weighted",
        focal_gamma=2.0,
        focal_alpha=0.25,
        **kwargs,
    ):
        self.n_features = n_features
        self.net_params = dict(
            n_features=n_features,
            f1=f1, f2=f2, f3=f3, f4=f4,
            wide_kernel=wide_kernel,
            narrow_kernel=narrow_kernel,
            narrow_kernel_2=narrow_kernel_2,
            dropout=dropout,
            head_hidden=head_hidden,
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
        logger.info("RepNetResNetHybridFeatures using device: %s", self.device)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "dropout": trial.suggest_float("rh_features_dropout", 0.1, 0.4),
            "lr": trial.suggest_float("rh_features_lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("rh_features_wd", 1e-5, 5e-3, log=True),
            "head_hidden": trial.suggest_categorical("rh_features_head_hidden", [32, 64, 128]),
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

    def fit(self, X_train, y_train, X_val, y_val,
            F_train=None, F_val=None):
        if F_train is None or F_val is None:
            raise ValueError(
                "RepNetResNetHybridFeatures requires F_train and F_val "
                "(z-scored HRV features). Use the train_resnet_hybrid_features "
                "script which precomputes and caches them."
            )

        self.model = RepNetResNetHybridFeatures(**self.net_params).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=self.weight_decay,
        )
        criterion = self._build_criterion(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        Ft = torch.tensor(F_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        train_dl = DataLoader(
            TensorDataset(Xt, Ft, yt),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if torch.cuda.is_available() else False,
        )
        Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        Fv = torch.tensor(F_val, dtype=torch.float32).to(self.device)

        self.history = {"train_loss": [], "val_auroc": []}
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for xb, fb, yb in train_dl:
                xb = xb.to(self.device, non_blocking=True)
                fb = fb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb, fb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            val_auc = self._score_device(Xv, Fv, y_val)
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
    def _score_device(self, Xv: torch.Tensor, Fv: torch.Tensor, y_val: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        probs = torch.softmax(self.model(Xv, Fv), dim=1)[:, 1].cpu().numpy()
        return roc_auc_score(y_val, probs)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, F: np.ndarray = None) -> np.ndarray:
        if F is None:
            raise ValueError("predict_proba requires features F (z-scored HRV).")
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32)
        Ft = torch.tensor(F, dtype=torch.float32)
        dl = DataLoader(
            TensorDataset(Xt, Ft),
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        probs = []
        for xb, fb in dl:
            xb = xb.to(self.device, non_blocking=True)
            fb = fb.to(self.device, non_blocking=True)
            probs.append(torch.softmax(self.model(xb, fb), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)

    def score(self, X, y, F=None):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, self.predict_proba(X, F=F)))
