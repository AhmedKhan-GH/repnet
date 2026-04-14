import numpy as np
import optuna

from .base import PreprocessingStep


class ZScoreNormalization(PreprocessingStep):
    """Z-score normalization: (x - mean) / std.

    Optuna chooses whether to normalize per-lead or globally across all leads.
    """

    def __init__(self, per_lead: bool = True, eps: float = 1e-8):
        super().__init__()
        self.per_lead = per_lead
        self.eps = eps

    @property
    def name(self) -> str:
        return "zscore"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.per_lead = trial.suggest_categorical(
                f"prep_{self.name}_per_lead", [True, False]
            )

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if self.per_lead:
            # Normalize each lead independently: mean/std over time axis
            mean = X.mean(axis=2, keepdims=True)
            std = X.std(axis=2, keepdims=True) + self.eps
        else:
            # Normalize each sample globally: mean/std over leads+time
            mean = X.mean(axis=(1, 2), keepdims=True)
            std = X.std(axis=(1, 2), keepdims=True) + self.eps
        X[:] = (X - mean) / std
        return X, y
