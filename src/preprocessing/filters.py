import numpy as np
import optuna
from scipy.signal import butter, sosfiltfilt

from .base import PreprocessingStep


class BaselineWanderFilter(PreprocessingStep):
    """High-pass Butterworth filter to remove baseline wander.

    Typical ECG baseline wander is below 0.5-0.67 Hz.
    Optuna tunes the cutoff frequency and filter order.
    """

    def __init__(self, cutoff: float = 0.5, order: int = 4, fs: float = 500.0):
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
        # sosfiltfilt operates along axis=-1 and broadcasts over leading dims
        # Reshape (N, 12, T) -> (N*12, T), filter, reshape back
        N, C, T = X.shape
        flat = X.reshape(N * C, T)
        filtered = sosfiltfilt(sos, flat, axis=-1)
        X[:] = filtered.reshape(N, C, T)
        return X, y
