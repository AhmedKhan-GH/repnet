from abc import ABC, abstractmethod

import numpy as np
import optuna


MODEL_REGISTRY: dict[str, type["BaseModel"]] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry.

    Usage:
        @register_model("random_forest")
        class RandomForestModel(BaseModel): ...
    """

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


class BaseModel(ABC):
    """Common interface for all models (sklearn, PyTorch, etc.).

    Every model must implement:
    - suggest_params(): let Optuna choose hyperparameters
    - fit(): train on data
    - predict(): produce predictions
    - score(): return primary metric (AUROC by default)
    """

    @staticmethod
    @abstractmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        """Return a dict of hyperparameters suggested by an Optuna trial."""
        ...

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train the model. DL models use X_val for early stopping."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates, shape (N,) for positive class."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute AUROC on the given data. Override for custom metrics."""
        from sklearn.metrics import roc_auc_score

        proba = self.predict_proba(X)
        return roc_auc_score(y, proba)
