"""Data loading utilities for ECG waveform data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)

logger = logging.getLogger(__name__)

# Standard 12-lead order (lowercase aVR/aVL/aVF matching the Senior Design CSVs)
SD_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SD_LABEL_POS = "Preeclampsia or Other Hypertensive Disorders of Pregnancy"
SD_LABEL_NEG = "Normal_All"
SD_N_SAMPLES = 2500   # crop all recordings to 2500 samples (10 s @ 250 Hz)



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
    data_dir: str | Path = "data/seniordesign_upload",
    return_patient_ids: bool = False,
):
    """Load the full Senior Design preeclampsia dataset.

    Directory structure expected:
        data_dir/
            metadata.csv          (columns include ECGTestID, PatLabel, Pat_Obfus_MRN)
            ekg_data/
                {ECGTestID}.csv   (columns = lead names, rows = timepoints)

    Args:
        data_dir: Path to the dataset directory.
        return_patient_ids: If True, also return aligned patient IDs (Pat_Obfus_MRN).

    Returns:
        If return_patient_ids is False (default): (X, y)
        If return_patient_ids is True: (X, y, patient_ids)
            X: np.ndarray of shape (N, 12, 2500), dtype float32
            y: np.ndarray of shape (N,), dtype int64  (1 = preeclampsia, 0 = normal)
            patient_ids: np.ndarray of shape (N,) — Pat_Obfus_MRN values, aligned to X/y
    """
    data_dir = Path(data_dir)
    ekg_dir = data_dir / "ekg_data"

    meta_path = next(
        (data_dir / name for name in ("metadata.csv", "metadata_balanced.csv")
         if (data_dir / name).exists()),
        None,
    )
    if meta_path is None:
        raise FileNotFoundError(
            f"No metadata.csv or metadata_balanced.csv found in {data_dir}"
        )
    meta = pd.read_csv(meta_path)
    available = {
        int(f.stem) for f in ekg_dir.iterdir() if f.suffix == ".csv"
    }
    meta = meta[meta["ECGTestID"].apply(lambda x: int(x) in available)].copy()
    logger.info("Metadata rows with waveform: %d", len(meta))

    has_pat_id = "Pat_Obfus_MRN" in meta.columns
    if return_patient_ids and not has_pat_id:
        raise KeyError(
            f"Pat_Obfus_MRN column missing from {meta_path}; cannot return patient IDs."
        )

    X_list, y_list, pat_list = [], [], []
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
            if has_pat_id:
                pat_list.append(row["Pat_Obfus_MRN"])
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            n_skip += 1

    if n_skip:
        logger.info("Skipped %d recordings (wrong shape or read error)", n_skip)

    X = np.stack(X_list)               # (N, 12, 2500)
    y = np.array(y_list, dtype=np.int64)
    pat_ids = np.array(pat_list) if has_pat_id else None

    # Quality filtering — NaN/Inf check
    qmask = _ecg_quality_mask(X)
    X = X[qmask]
    y = y[qmask]
    if pat_ids is not None:
        pat_ids = pat_ids[qmask]

    logger.info(
        "Loaded: X=%s, y=%s (pos=%d, neg=%d, pos rate=%.1f%%)",
        X.shape, y.shape, (y == 1).sum(), (y == 0).sum(), 100 * y.mean(),
    )
    if return_patient_ids:
        n_unique_pat = len(np.unique(pat_ids))
        logger.info("Unique patients: %d (%.2f recordings/patient avg)",
                    n_unique_pat, len(pat_ids) / max(n_unique_pat, 1))
        return X, y, pat_ids
    return X, y


def majority_undersample(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the majority class to achieve the target ratio.

    ratio=1.0 means equal classes (1:1).
    ratio=2.0 means 2 majority per 1 minority.

    Applied per-fold on training data only — never on val/test.
    """
    rng = np.random.default_rng(seed)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos >= n_neg:
        # Majority is positive — undersample positives
        target_maj = int(n_neg * ratio)
        maj_idx = np.where(y == 1)[0]
        min_idx = np.where(y == 0)[0]
    else:
        # Majority is negative — undersample negatives
        target_maj = int(n_pos * ratio)
        maj_idx = np.where(y == 0)[0]
        min_idx = np.where(y == 1)[0]

    target_maj = min(target_maj, len(maj_idx))  # can't upsample
    keep_maj = rng.choice(maj_idx, size=target_maj, replace=False)
    keep = np.sort(np.concatenate([min_idx, keep_maj]))

    logger.info(
        "Undersampled: %d -> %d (pos=%d, neg=%d)",
        len(y), len(keep),
        int((y[keep] == 1).sum()), int((y[keep] == 0).sum()),
    )
    return X[keep], y[keep]


def minority_oversample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate minority-class samples with replacement until classes are balanced.

    Applied per-fold on TRAINING data only. Pair with augmentation downstream so
    the replicated minority samples receive independent transformations and the
    model isn't trained on identical copies.
    """
    rng = np.random.default_rng(seed)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == n_neg:
        return X, y

    if n_pos < n_neg:
        minor_idx = np.where(y == 1)[0]
        n_extra = n_neg - n_pos
    else:
        minor_idx = np.where(y == 0)[0]
        n_extra = n_pos - n_neg

    extra_idx = rng.choice(minor_idx, size=n_extra, replace=True)
    X_out = np.concatenate([X, X[extra_idx]], axis=0)
    y_out = np.concatenate([y, y[extra_idx]], axis=0)

    logger.info(
        "Oversampled minority: %d -> %d (pos=%d, neg=%d)",
        len(y), len(y_out),
        int((y_out == 1).sum()), int((y_out == 0).sum()),
    )
    return X_out, y_out


def balance_downsample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample majority to match minority count -> exact 1:1 balance.

    For val/early-stop sets where you want a clean unaugmented balanced
    evaluation signal. Test holdout should still keep real prevalence.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    target = min(len(pos_idx), len(neg_idx))
    if target == 0:
        return X, y

    keep_pos = rng.choice(pos_idx, size=target, replace=False)
    keep_neg = rng.choice(neg_idx, size=target, replace=False)
    keep = np.sort(np.concatenate([keep_pos, keep_neg]))

    logger.info(
        "Balanced (downsampled): %d -> %d (pos=%d, neg=%d)",
        len(y), len(keep), target, target,
    )
    return X[keep], y[keep]


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


def split_holdout_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Patient-grouped 80/20 holdout split — no patient appears on both sides.

    Uses GroupShuffleSplit (not stratified — sklearn has no stratified-grouped
    single-shot splitter). Class balance is reported and a warning is logged
    if test pos-rate diverges from dev by >5 percentage points.

    Returns (X_dev, X_test, y_dev, y_test, groups_dev, groups_test).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(splitter.split(X, y, groups))

    X_dev, X_test = X[dev_idx], X[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]
    g_dev, g_test = groups[dev_idx], groups[test_idx]

    # Sanity: patient sets must be disjoint
    leak = set(g_dev).intersection(set(g_test))
    if leak:
        raise RuntimeError(f"Patient leakage in holdout: {len(leak)} shared IDs")

    pos_rate_dev = y_dev.mean()
    pos_rate_test = y_test.mean()
    logger.info(
        "Grouped holdout: dev=%d (%d patients, pos=%.1f%%) test=%d (%d patients, pos=%.1f%%)",
        len(y_dev), len(np.unique(g_dev)), 100 * pos_rate_dev,
        len(y_test), len(np.unique(g_test)), 100 * pos_rate_test,
    )
    if abs(pos_rate_dev - pos_rate_test) > 0.05:
        logger.warning(
            "Class balance drift: dev pos=%.1f%% vs test pos=%.1f%% (>5pt). "
            "Consider re-seeding or stratifying within groups.",
            100 * pos_rate_dev, 100 * pos_rate_test,
        )
    return X_dev, X_test, y_dev, y_test, g_dev, g_test


def kfold_cv_indices_grouped(
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold split that keeps patients within a single fold.

    Uses StratifiedGroupKFold — preserves both class balance and group integrity.
    Note: StratifiedGroupKFold is not strictly random; the seed only affects shuffle
    when shuffle=True is passed.

    Returns list of (train_indices, val_indices) for each fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups=groups))

    # Sanity: every fold must have disjoint patient sets between train and val
    for i, (train_idx, val_idx) in enumerate(folds):
        leak = set(groups[train_idx]).intersection(set(groups[val_idx]))
        if leak:
            raise RuntimeError(
                f"Patient leakage in fold {i}: {len(leak)} shared IDs"
            )
    return folds
