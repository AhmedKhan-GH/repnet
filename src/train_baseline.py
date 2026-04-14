"""Single baseline training run: ECG-AI paper defaults on Nightingale.

Usage:
    python -m src.train_baseline
"""

import logging

from src.data.dataset import load_nightingale, split_holdout
from src.models.ecg_ai_resnet import ECGAIResNetModel
from src.preprocessing.normalization import ZScoreNormalization

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Load
X, y = load_nightingale("data/Nightingale Dataset")

# Z-score normalize
zscore = ZScoreNormalization(per_lead=True)
X, _ = zscore.transform(X)

# 80/20 split
X_train, X_val, y_train, y_val = split_holdout(X, y, test_size=0.20, seed=42)
logger.info("Train: %d, Val: %d (pos rate: %.1f%%)", len(y_train), len(y_val), 100 * y.mean())

# Paper defaults
model = ECGAIResNetModel(
    stage_filters=(16, 32, 64),
    kernel_size=3,
    dropout=0.1,
    lr=1e-3,
    batch_size=32,
    epochs=30,
)

model.fit(X_train, y_train, X_val, y_val)
auroc = model.score(X_val, y_val)
print(f"\nVal AUROC: {auroc:.4f}")
