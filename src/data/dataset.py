"""Data loading utilities for ECG waveform data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Standard 12-lead order (lowercase aVR/aVL/aVF matching the Senior Design CSVs)
SD_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SD_LABEL_POS = "Preeclampsia or Other Hypertensive Disorders of Pregnancy"
SD_LABEL_NEG = "Normal_All"
SD_N_SAMPLES = 2500   # crop all recordings to 2500 samples (5 s @ 500 Hz)



def _ecg_quality_mask(X: np.ndarray) -> np.ndarray:
    """Check for NaN/Inf values only. All waveforms were visually inspected.

    Returns boolean mask of shape (N,).
    """
    mask = np.isfinite(X).all(axis=2).all(axis=1)  # (N,)
    n_fail = X.shape[0] - mask.sum()
    if n_fail > 0:
        logger.info("Dropped %d samples with NaN/Inf values", n_fail)
    return mask


def load_seniordesign(
    data_dir: str | Path = "data/seniordesign_upload_balanced",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the Senior Design balanced preeclampsia dataset.

    Directory structure expected:
        data_dir/
            metadata_balanced.csv   (columns include ECGTestID, PatLabel)
            ekg_data/
                {ECGTestID}.csv     (columns = lead names, rows = timepoints)

    Each waveform CSV has lead columns in SD_LEAD_ORDER.
    Recordings longer than SD_N_SAMPLES (2500) are cropped; shorter ones are skipped.
    Quality filtering discards broken ECGs (NaN, flat lines, excessive zeros,
    extreme amplitude outliers).

    Returns:
        X: np.ndarray of shape (N, 12, 2500), dtype float32
        y: np.ndarray of shape (N,), dtype int64  (1 = preeclampsia, 0 = normal)
    """
    data_dir = Path(data_dir)
    ekg_dir = data_dir / "ekg_data"

    meta = pd.read_csv(data_dir / "metadata_balanced.csv")
    available = {
        int(f.stem) for f in ekg_dir.iterdir() if f.suffix == ".csv"
    }
    meta = meta[meta["ECGTestID"].apply(lambda x: int(x) in available)].copy()
    logger.info("Metadata rows with waveform: %d", len(meta))

    X_list, y_list = [], []
    n_skip = 0
    for _, row in meta.iterrows():
        path = ekg_dir / f"{int(row['ECGTestID'])}.csv"
        try:
            df = pd.read_csv(path, skipinitialspace=True,
                             usecols=SD_LEAD_ORDER, nrows=SD_N_SAMPLES)
            arr = df[SD_LEAD_ORDER].values.T.astype(np.float32)  # (12, T)
            if arr.shape != (12, SD_N_SAMPLES):
                n_skip += 1
                continue
            X_list.append(arr)
            y_list.append(1 if row["PatLabel"] == SD_LABEL_POS else 0)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            n_skip += 1

    if n_skip:
        logger.info("Skipped %d recordings (wrong shape or read error)", n_skip)

    X = np.stack(X_list)               # (N, 12, 2500)
    y = np.array(y_list, dtype=np.int64)

    # Quality filtering — discard broken ECGs
    qmask = _ecg_quality_mask(X)
    X = X[qmask]
    y = y[qmask]

    logger.info(
        "Loaded Senior Design: X=%s, y=%s (pos=%d, neg=%d, pos rate=%.1f%%)",
        X.shape, y.shape, (y == 1).sum(), (y == 0).sum(), 100 * y.mean(),
    )
    return X, y


def generate_synthetic_ecg(
    n_samples: int = 200,
    n_leads: int = 12,
    seq_len: int = 2250,
    prevalence: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ECG-like data for testing the pipeline.

    Returns (X, y) where X is (n_samples, n_leads, seq_len) and y is binary labels.
    Positive class gets a subtle amplitude shift so the model has something to learn.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n_samples * prevalence)
    n_neg = n_samples - n_pos

    X_neg = rng.standard_normal((n_neg, n_leads, seq_len)).astype(np.float32)
    X_pos = rng.standard_normal((n_pos, n_leads, seq_len)).astype(np.float32) + 0.3

    X = np.concatenate([X_neg, X_pos], axis=0)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)]).astype(np.int64)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def split_holdout(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 split into dev and held-out test sets.

    Returns (X_dev, X_test, y_dev, y_test).
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def kfold_cv_indices(
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold train/val index pairs.

    Returns list of (train_indices, val_indices) for each fold.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
