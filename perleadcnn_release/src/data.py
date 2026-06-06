"""Data loading, preprocessing, and patient-grouped splitting.

The raw dataset is 12-lead ECG recordings (PHI — not shipped; see DATA.md).
This module is the single source of truth for how waveforms are read,
filtered, normalized, downsampled, and split — shared by both training
(`train.py`) and evaluation (`evaluate.py`) so the two are always consistent.

Expected layout (configurable via the REPNET_DATA_DIR environment variable):

    <data_dir>/
      metadata.csv            # columns: ECGTestID, Pat_Obfus_MRN, PatLabel, ...
      ekg_data/
        <ECGTestID>.csv       # one file per recording; columns I,II,...,V6
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, resample, sosfiltfilt, tf2sos
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------------------------------
# Dataset constants
# --------------------------------------------------------------------------
N_LEADS = 12
SEQ_LEN = 5000           # native length @ 500 Hz
FS = 500.0               # native sampling rate (Hz)
DOWNSAMPLE = 2           # 500 Hz -> 250 Hz (take every 2nd sample)
SEQ_LEN_MODEL = SEQ_LEN // DOWNSAMPLE  # 2500 samples @ 250 Hz (model input)

SD_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
SD_LABEL_POS = "Preeclampsia or Other Hypertensive Disorders of Pregnancy"

_PACKAGE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "seniordesign_upload",
)


def resolve_data_dir() -> str:
    """The data directory, read from REPNET_DATA_DIR at call time.

    Resolved live (not bound at import) so setting REPNET_DATA_DIR after
    `import src.data` — e.g. in a notebook cell — is honoured.
    """
    return os.environ.get("REPNET_DATA_DIR", _PACKAGE_DATA_DIR)


# Import-time snapshot, kept for "is the dataset present?" checks in
# analyze.py / tests. The loader resolves the dir live via resolve_data_dir().
DEFAULT_DATA_DIR = resolve_data_dir()


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _apply_sos(sos, X):
    N, C, T = X.shape
    flat = X.reshape(N * C, T)
    X[:] = sosfiltfilt(sos, flat, axis=-1).reshape(N, C, T)
    return X


def preprocess(X: np.ndarray) -> np.ndarray:
    """0.5 Hz high-pass (baseline wander) + 60 Hz notch + per-lead z-score.

    Operates at the native 500 Hz; expects X of shape (N, 12, 5000).
    """
    X = X.copy()
    X = _apply_sos(butter(4, 0.5, btype="high", fs=FS, output="sos"), X)
    b, a = iirnotch(60.0, 30.0, fs=FS)
    X = _apply_sos(tf2sos(b, a), X)
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


def load_ecg_data(data_dir: str | None = None):
    """Read raw waveforms + labels. Returns (X, y, patient_ids, is_upsampled).

    X has shape (N, 12, 5000) at 500 Hz (no preprocessing applied yet).
    Recordings stored at 2500 samples are resampled up to 5000 so the filter
    design is identical for every record.
    """
    data_dir = os.path.normpath(data_dir or resolve_data_dir())
    ekg_dir = os.path.join(data_dir, "ekg_data")
    meta_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(data_dir, "metadata_balanced.csv")

    # Fail fast with actionable guidance if the dataset is absent, rather than
    # surfacing a deep pandas/os traceback later. The data is PHI (see DATA.md).
    if not (os.path.isdir(data_dir) and os.path.isdir(ekg_dir)
            and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"ECG dataset not found at: {data_dir}\n"
            f"Expected {os.path.join(data_dir, 'metadata.csv')} and "
            f"{ekg_dir}{os.sep} (one CSV per recording).\n"
            f"The dataset is PHI and is not shipped with this release. Set the "
            f"REPNET_DATA_DIR environment variable to your data directory; see "
            f"DATA.md for the expected layout."
        )

    meta = pd.read_csv(meta_path)

    available = {int(os.path.splitext(f)[0])
                 for f in os.listdir(ekg_dir) if f.endswith(".csv")}
    meta = meta[meta["ECGTestID"].apply(lambda x: int(x) in available)].copy()

    X_list, y_list, pat_list, up_list = [], [], [], []
    for _, row in meta.iterrows():
        path = os.path.join(ekg_dir, f"{int(row['ECGTestID'])}.csv")
        try:
            df = pd.read_csv(path, skipinitialspace=True, usecols=SD_LEAD_ORDER)
            arr = df[SD_LEAD_ORDER].values.T.astype(np.float32)
            if arr.shape[0] != 12:
                continue
            n = arr.shape[1]
            if n == SEQ_LEN:
                was_up = False
            elif n == 2500:
                arr = resample(arr, SEQ_LEN, axis=1).astype(np.float32)
                was_up = True
            else:
                continue
            if arr.shape != (12, SEQ_LEN):
                continue
            X_list.append(arr)
            y_list.append(1 if row["PatLabel"] == SD_LABEL_POS else 0)
            pat_list.append(row["Pat_Obfus_MRN"])
            up_list.append(was_up)
        except Exception:
            continue

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    patient_ids = np.array(pat_list)
    is_upsampled = np.array(up_list, dtype=bool)

    # Drop non-finite, flat-lead, and unlabelled-patient records.
    mask = np.isfinite(X).all(axis=2).all(axis=1)
    X, y, patient_ids, is_upsampled = X[mask], y[mask], patient_ids[mask], is_upsampled[mask]
    keep = ~((X.std(axis=2) < 1e-4).any(axis=1))
    X, y, patient_ids, is_upsampled = X[keep], y[keep], patient_ids[keep], is_upsampled[keep]
    try:
        nan_mask = np.isnan(patient_ids.astype(float))
    except (ValueError, TypeError):
        nan_mask = np.array([str(p).strip() in ("", "nan", "None") for p in patient_ids])
    keep = ~nan_mask
    return X[keep], y[keep], patient_ids[keep], is_upsampled[keep]


def load_dataset(data_dir: str | None = None):
    """Convenience: load + preprocess + downsample to model input.

    Returns (X, y, patient_ids) with X of shape (N, 12, 2500) @ 250 Hz.
    """
    X, y, patient_ids, _ = load_ecg_data(data_dir)
    X = preprocess(X)
    X = X[:, :, ::DOWNSAMPLE]
    return X, y, patient_ids


# --------------------------------------------------------------------------
# Patient-grouped splits (deterministic, seeded by split index)
# --------------------------------------------------------------------------
def split_seed(split_i: int) -> int:
    """The exact seed scheme used for the released results."""
    return split_i * 7 + 1000


def test_split(split_i: int, y, groups):
    """Return the test-set indices for `split_i`.

    Outer StratifiedGroupKFold(n_splits=5); the first fold is the held-out
    test set. This matches the released `multisplit_dbb6f49` evaluation.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                random_state=split_seed(split_i))
    _, test_idx = next(iter(sgkf.split(np.zeros(len(y)), y, groups=groups)))
    return test_idx


def train_val_test_split(split_i: int, y, groups):
    """Return (train_idx, val_idx, test_idx) for `split_i` (training pipeline).

    Outer StratifiedGroupKFold(5) -> dev / test; inner StratifiedGroupKFold(8)
    on dev -> train / val. All patient-grouped; no patient appears in more
    than one partition.
    """
    s = split_seed(split_i)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
    dev_idx, test_idx = next(iter(sgkf.split(np.zeros(len(y)), y, groups=groups)))
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=s + 1)
    tr_sub, va_sub = next(iter(inner.split(
        np.zeros(len(dev_idx)), y[dev_idx], groups=groups[dev_idx])))
    return dev_idx[tr_sub], dev_idx[va_sub], test_idx
