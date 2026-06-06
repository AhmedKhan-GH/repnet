"""PerLeadCNN — the final model.

A 3-stage, weight-shared 1D CNN applied independently to each of the 12 ECG
leads. Per-lead features are global-average-pooled, concatenated across leads,
and mapped to logits by a single linear layer.

Default config (the one in the released checkpoints): 29,490 parameters.

    Input: (B, 12, 2500)                     # 12 leads, 2500 samples @ 250 Hz
      reshape -> (B*12, 1, 2500)             # each lead processed independently
      Conv1d(1 ->16, k=31, s=2) + BN + Mish
      Conv1d(16->32, k=21, s=2) + BN + Mish  # weights shared across all 12 leads
      Conv1d(32->48, k=11, s=2) + BN + Mish
      AdaptiveAvgPool1d(1) -> (B*12, 48)
      reshape -> (B, 576)                     # FUSION: concatenate the 12 leads
      Dropout(0.15) -> Linear(576, 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerLeadCNN(nn.Module):
    def __init__(self, n_leads: int = 12, filters=(16, 32, 48),
                 kernels=(31, 21, 11), dropout: float = 0.15,
                 n_classes: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        for f, k in zip(filters, kernels):
            layers.extend([
                nn.Conv1d(in_ch, f, k, stride=2, padding=k // 2, bias=False),
                nn.BatchNorm1d(f),
                nn.Mish(),
            ])
            in_ch = f
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.n_leads = n_leads
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[-1] * n_leads, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_leads, T)
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)          # each lead independent
        x = self.backbone(x)                # shared weights
        x = self.pool(x).squeeze(-1)        # (B*L, C)
        x = x.reshape(B, L * x.shape[-1])   # concatenate leads -> (B, L*C)
        return self.fc(self.head_drop(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = PerLeadCNN()
    print(f"PerLeadCNN parameters: {count_parameters(m):,}")
