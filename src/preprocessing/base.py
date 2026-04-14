from abc import ABC, abstractmethod

import numpy as np
import optuna


class PreprocessingStep(ABC):
    """Base class for all preprocessing steps.

    Each step can:
    - Declare tunable hyperparameters via suggest_params()
    - Transform ECG data in-place or return transformed copy
    """

    # Override to True in steps that change sample count (undersampling, SMOTE)
    is_resampling = False

    def __init__(self):
        self.enabled = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used as Optuna parameter prefix."""
        ...

    def suggest_params(self, trial: optuna.Trial) -> None:
        """Suggest hyperparameters from an Optuna trial.

        Override this to add tunable parameters. The base implementation
        only suggests the on/off toggle.
        """
        self.enabled = trial.suggest_categorical(
            f"prep_{self.name}_enabled", [True, False]
        )

    @abstractmethod
    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform ECG array of shape (N, n_leads, n_timepoints).

        Steps that only modify signal data can ignore y and pass it through.
        Steps that resample (e.g. undersampling) modify both X and y.

        Returns (X_transformed, y_transformed).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled})"


class PreprocessingPipeline:
    """Composable pipeline of PreprocessingSteps.

    Optuna can toggle each step on/off and tune its parameters.
    Steps are applied in the order they are provided.
    """

    def __init__(self, steps: list[PreprocessingStep]):
        self.steps = steps

    def suggest_and_configure(self, trial: optuna.Trial) -> None:
        """Let each step suggest its hyperparameters (including on/off)."""
        for step in self.steps:
            step.suggest_params(trial)

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply signal transforms only (filters, normalization). Skips resampling."""
        result = X.copy()
        for step in self.steps:
            if step.enabled and not step.is_resampling:
                result, _ = step.transform(result)
        return result, y

    def resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply resampling steps only (undersampling, SMOTE). Call per-fold on train data."""
        for step in self.steps:
            if step.enabled and step.is_resampling:
                X, y = step.transform(X, y)
        return X, y

    def get_active_steps(self) -> list[PreprocessingStep]:
        return [s for s in self.steps if s.enabled]

    def __repr__(self) -> str:
        active = self.get_active_steps()
        return f"Pipeline({' -> '.join(repr(s) for s in active)})"
