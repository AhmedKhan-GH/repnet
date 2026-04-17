"""ECG signal augmentation steps (applied to training data only)."""

from __future__ import annotations

import numpy as np
import optuna

from .base import PreprocessingStep


class GaussianNoise(PreprocessingStep):
    """Add per-sample Gaussian noise to ECG signals.

    Simulates measurement noise and electrode contact variation.
    Optuna tunes the noise standard deviation.
    """

    is_augmentation = True

    def __init__(self, sigma: float = 0.02):
        super().__init__()
        self.sigma = sigma

    @property
    def name(self) -> str:
        return "gaussian_noise"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.sigma = trial.suggest_float(
                f"prep_{self.name}_sigma", 0.005, 0.1, log=True
            )

    def transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        rng = np.random.default_rng()
        return X + rng.normal(0, self.sigma, X.shape).astype(X.dtype), y


class AmplitudeScaling(PreprocessingStep):
    """Randomly scale amplitude per sample.

    Simulates inter-patient electrode placement variation.
    Optuna tunes the maximum fractional deviation from 1.0.
    """

    is_augmentation = True

    def __init__(self, scale_range: float = 0.1):
        super().__init__()
        self.scale_range = scale_range

    @property
    def name(self) -> str:
        return "amplitude_scaling"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.scale_range = trial.suggest_float(
                f"prep_{self.name}_scale_range", 0.05, 0.25
            )

    def transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        rng = np.random.default_rng()
        scales = rng.uniform(
            1 - self.scale_range, 1 + self.scale_range, size=(X.shape[0], 1, 1)
        ).astype(X.dtype)
        return X * scales, y


class RandomTimeShift(PreprocessingStep):
    """Randomly shift each signal in time, zero-padding the vacated end.

    Forces the model not to rely on absolute signal position.
    Optuna tunes the maximum shift in samples.
    """

    is_augmentation = True

    def __init__(self, max_shift: int = 200):
        super().__init__()
        self.max_shift = max_shift

    @property
    def name(self) -> str:
        return "time_shift"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.max_shift = trial.suggest_int(
                f"prep_{self.name}_max_shift", 50, 500
            )

    def transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        rng = np.random.default_rng()
        N, C, T = X.shape
        shifts = rng.integers(-self.max_shift, self.max_shift + 1, size=N)
        out = np.zeros_like(X)
        for i, shift in enumerate(shifts):
            if shift > 0:
                out[i, :, shift:] = X[i, :, : T - shift]
            elif shift < 0:
                out[i, :, : T + shift] = X[i, :, -shift:]
            else:
                out[i] = X[i]
        return out, y
