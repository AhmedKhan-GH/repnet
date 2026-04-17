from __future__ import annotations

import numpy as np
import optuna
from scipy.signal import butter, iirnotch, sosfiltfilt, sosfilt_zi

from .base import PreprocessingStep


def _apply_sos(sos: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Zero-phase SOS filter over (N, C, T) array, in-place."""
    N, C, T = X.shape
    flat = X.reshape(N * C, T)
    X[:] = sosfiltfilt(sos, flat, axis=-1).reshape(N, C, T)
    return X


class BaselineWanderFilter(PreprocessingStep):
    """High-pass Butterworth filter to remove baseline wander.

    Typical ECG baseline wander is below 0.5-0.67 Hz.
    Optuna tunes the cutoff frequency and filter order.
    """

    def __init__(self, cutoff: float = 0.5, order: int = 4, fs: float = 250.0):
        super().__init__()
        self.cutoff = cutoff
        self.order = order
        self.fs = fs

    @property
    def name(self) -> str:
        return "baseline_wander"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.cutoff = trial.suggest_float(
                f"prep_{self.name}_cutoff", 0.3, 0.67, log=True
            )
            self.order = trial.suggest_int(f"prep_{self.name}_order", 3, 5)

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        sos = butter(self.order, self.cutoff, btype="high", fs=self.fs, output="sos")
        return _apply_sos(sos, X), y


class NotchFilter(PreprocessingStep):
    """Notch filter to remove powerline interference.

    US mains = 60 Hz. Q factor controls notch width: higher Q = narrower notch.
    Default Q=30 removes 60 Hz ± ~1 Hz, which is standard for ECG.
    """

    def __init__(self, freq: float = 60.0, Q: float = 30.0, fs: float = 250.0):
        super().__init__()
        self.freq = freq
        self.Q = Q
        self.fs = fs

    @property
    def name(self) -> str:
        return "notch_60hz"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.Q = trial.suggest_float(f"prep_{self.name}_Q", 15.0, 50.0)

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        b, a = iirnotch(self.freq, self.Q, fs=self.fs)
        # Convert to SOS for numerical stability with sosfiltfilt
        from scipy.signal import tf2sos
        sos = tf2sos(b, a)
        return _apply_sos(sos, X), y


class BandpassFilter(PreprocessingStep):
    """Bandpass Butterworth filter to retain clinically relevant ECG frequencies.

    Default 0.5–40 Hz removes both baseline wander and high-frequency noise/muscle artifact.
    When enabled, this replaces the need for a separate BaselineWanderFilter.
    Optuna tunes low/high cutoffs and filter order.
    """

    def __init__(self, low: float = 0.5, high: float = 40.0, order: int = 4,
                 fs: float = 250.0):
        super().__init__()
        self.low = low
        self.high = high
        self.order = order
        self.fs = fs

    @property
    def name(self) -> str:
        return "bandpass"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.low = trial.suggest_float(f"prep_{self.name}_low", 0.3, 1.0, log=True)
            self.high = trial.suggest_float(f"prep_{self.name}_high", 30.0, 100.0)
            self.order = trial.suggest_int(f"prep_{self.name}_order", 3, 6)

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        sos = butter(self.order, [self.low, self.high], btype="band", fs=self.fs, output="sos")
        return _apply_sos(sos, X), y
