"""Heart-rate-variability features extracted from lead II.

Standard time-domain + frequency-domain HRV per Task Force of the European
Society of Cardiology (1996). These are the autonomic-dysfunction proxies
that the literature has shown discriminate hypertensive disorders.

Features (per sample):
  hr           : mean heart rate (BPM)
  mean_rr      : mean RR interval (ms)
  sdnn         : standard deviation of NN intervals (ms)
  rmssd        : root mean square of successive differences (ms)
  pnn50        : % of |dRR| > 50 ms
  lf_hf_ratio  : LF (0.04-0.15 Hz) / HF (0.15-0.40 Hz) power ratio
  total_power  : total RR-series power (ms^2)
  n_beats      : number of R-peaks detected (sanity check)

Lead II is the HRV standard because R-peaks align with the heart's mean
electrical axis and have the highest amplitude. Input X must already be
z-score-normalized (the project's preprocessing does this).

Note on caching: features depend only on the raw signal, not on train/val
splits, so extraction is run once over the full dataset and cached.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    "hr", "mean_rr", "sdnn", "rmssd",
    "pnn50", "lf_hf_ratio", "total_power", "n_beats",
)
N_FEATURES = len(FEATURE_NAMES)

LEAD_II_IDX = 1  # SD_LEAD_ORDER = ["I", "II", "III", ...]
FS = 250         # sampling rate Hz
MAX_HR = 200     # BPM (defines min R-R distance for peak detection)


def _detect_r_peaks(signal_lead2: np.ndarray, fs: int = FS) -> np.ndarray:
    """Heuristic R-peak detection on a z-scored lead-II trace."""
    min_dist = int(fs * 60 / MAX_HR)        # samples between peaks (~75 @ 250Hz / 200BPM)
    threshold = max(0.5, float(signal_lead2.std()) * 1.0)
    peaks, _ = sps.find_peaks(signal_lead2, height=threshold, distance=min_dist)

    # Fallback: if too few peaks, lower the threshold.
    expected_low = int(len(signal_lead2) / fs * 0.5)   # >= 30 BPM
    if len(peaks) < max(3, expected_low):
        peaks, _ = sps.find_peaks(signal_lead2, distance=min_dist)
    return peaks


def _hrv_from_peaks(peaks: np.ndarray, fs: int = FS) -> dict[str, float]:
    """Compute HRV features from R-peak locations."""
    out = {k: 0.0 for k in FEATURE_NAMES}
    if len(peaks) < 4:
        out["n_beats"] = float(len(peaks))
        return out

    rr_ms = np.diff(peaks).astype(np.float64) * 1000.0 / fs

    # Time domain
    out["mean_rr"] = float(rr_ms.mean())
    out["hr"] = 60000.0 / out["mean_rr"]
    out["sdnn"] = float(rr_ms.std())

    diff_rr = np.diff(rr_ms)
    out["rmssd"] = float(np.sqrt((diff_rr ** 2).mean())) if len(diff_rr) else 0.0
    out["pnn50"] = float((np.abs(diff_rr) > 50).mean() * 100) if len(diff_rr) else 0.0

    # Frequency domain via Welch PSD on the RR series resampled at 4 Hz
    if len(rr_ms) >= 8:
        rr_times = np.cumsum(rr_ms) / 1000.0   # seconds
        target_fs = 4.0
        if rr_times[-1] - rr_times[0] > 1.0:
            t_resampled = np.arange(rr_times[0], rr_times[-1], 1.0 / target_fs)
            rr_resampled = np.interp(t_resampled, rr_times, rr_ms)
            nperseg = min(len(rr_resampled), 64)
            if nperseg >= 8:
                freqs, psd = sps.welch(rr_resampled, fs=target_fs, nperseg=nperseg)
                lf = (freqs >= 0.04) & (freqs < 0.15)
                hf = (freqs >= 0.15) & (freqs < 0.40)
                lf_p = float(np.trapezoid(psd[lf], freqs[lf])) if lf.any() else 0.0
                hf_p = float(np.trapezoid(psd[hf], freqs[hf])) if hf.any() else 0.0
                out["lf_hf_ratio"] = lf_p / max(hf_p, 1e-6)
                out["total_power"] = float(np.trapezoid(psd, freqs))

    out["n_beats"] = float(len(peaks))
    return out


def extract_hrv_features(X: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extract HRV features for every sample. X: (N, 12, T) -> (N, N_FEATURES)."""
    if X.ndim != 3:
        raise ValueError(f"X must be (N, 12, T); got {X.shape}")
    n = X.shape[0]
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        peaks = _detect_r_peaks(X[i, LEAD_II_IDX], fs=fs)
        feats = _hrv_from_peaks(peaks, fs=fs)
        out[i] = [feats[k] for k in FEATURE_NAMES]

    # Replace NaN/Inf (from edge-case divisions) with 0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def cached_extract(X: np.ndarray, cache_path: str | Path,
                   force: bool = False) -> np.ndarray:
    """Extract HRV features with disk caching keyed by X's content hash."""
    cache_path = Path(cache_path)
    content_hash = hash(X.tobytes()) & 0xFFFFFFFF
    meta_path = cache_path.with_suffix(".meta.txt")

    if not force and cache_path.exists() and meta_path.exists():
        try:
            cached_hash = int(meta_path.read_text().strip())
            if cached_hash == content_hash:
                feats = np.load(cache_path)
                if feats.shape == (X.shape[0], N_FEATURES):
                    logger.info("HRV cache HIT  (%s, %d samples)", cache_path, len(feats))
                    return feats
        except Exception as e:
            logger.warning("HRV cache read failed (%s); recomputing.", e)

    logger.info("HRV cache MISS (%s, %d samples) -- extracting...", cache_path, X.shape[0])
    feats = extract_hrv_features(X)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feats)
    meta_path.write_text(str(content_hash))
    logger.info("HRV features cached to %s", cache_path)
    return feats
