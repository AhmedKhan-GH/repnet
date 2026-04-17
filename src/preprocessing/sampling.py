from __future__ import annotations

import numpy as np
import optuna
from sklearn.neighbors import NearestNeighbors

from .base import PreprocessingStep


class MajorityUndersampling(PreprocessingStep):
    """Random majority undersampling to address class imbalance.

    Downsamples the majority class to match the minority class count,
    optionally scaled by a ratio. A ratio of 1.0 means perfect balance;
    ratio of 2.0 means 2x as many majority samples as minority.
    """

    is_resampling = True

    def __init__(self, ratio: float = 1.0, seed: int = 42):
        super().__init__()
        self.ratio = ratio
        self.seed = seed

    @property
    def name(self) -> str:
        return "undersample"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.ratio = trial.suggest_float(
                f"prep_{self.name}_ratio", 1.0, 3.0,
            )

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if y is None:
            return X, y

        rng = np.random.default_rng(self.seed)

        minority_label = 1 if (y == 1).sum() <= (y == 0).sum() else 0
        majority_label = 1 - minority_label

        minority_idx = np.where(y == minority_label)[0]
        majority_idx = np.where(y == majority_label)[0]

        n_minority = len(minority_idx)
        n_keep = min(len(majority_idx), int(n_minority * self.ratio))

        majority_keep = rng.choice(majority_idx, size=n_keep, replace=False)
        keep_idx = np.sort(np.concatenate([minority_idx, majority_keep]))

        return X[keep_idx], y[keep_idx]


class SMOTE(PreprocessingStep):
    """Synthetic Minority Oversampling Technique for ECG signals.

    Generates synthetic minority samples by interpolating between a minority
    sample and one of its k nearest neighbours in flattened feature space.

    Optuna tunes:
      - k_neighbors: number of nearest neighbours (1-10)
      - target_ratio: desired minority:majority ratio after oversampling
        (1.0 = perfect balance)

    Only applies to training data (when y is provided).
    """

    is_resampling = True

    def __init__(self, k_neighbors: int = 5, target_ratio: float = 1.0, seed: int = 42):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.target_ratio = target_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        return "smote"

    def suggest_params(self, trial: optuna.Trial) -> None:
        super().suggest_params(trial)
        if self.enabled:
            self.k_neighbors = trial.suggest_int(f"prep_{self.name}_k", 3, 7)
            self.target_ratio = trial.suggest_float(f"prep_{self.name}_ratio", 0.5, 1.0)

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if y is None:
            return X, y

        minority_label = 1 if (y == 1).sum() <= (y == 0).sum() else 0
        majority_label = 1 - minority_label

        minority_idx = np.where(y == minority_label)[0]
        majority_idx = np.where(y == majority_label)[0]

        n_minority = len(minority_idx)
        n_majority = len(majority_idx)
        n_target = int(n_majority * self.target_ratio)
        n_synthetic = n_target - n_minority

        if n_synthetic <= 0:
            return X, y

        rng = np.random.default_rng(self.seed)
        X_min = X[minority_idx]  # (n_minority, 12, T)
        original_shape = X_min.shape[1:]  # (12, T)

        # Flatten to 2D for kNN: (n_minority, 12*T)
        X_min_flat = X_min.reshape(n_minority, -1)
        k = min(self.k_neighbors, n_minority - 1)
        if k < 1:
            return X, y

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(X_min_flat)
        neighbors = nn.kneighbors(X_min_flat, return_distance=False)
        # Exclude self (column 0)
        neighbors = neighbors[:, 1:]

        # Generate synthetic samples
        synthetic = np.empty((n_synthetic, *original_shape), dtype=X.dtype)
        for i in range(n_synthetic):
            idx = rng.integers(0, n_minority)
            nn_idx = rng.choice(neighbors[idx])
            lam = rng.uniform(0.0, 1.0)
            synthetic[i] = X_min[idx] + lam * (X_min[nn_idx] - X_min[idx])

        X_out = np.concatenate([X, synthetic], axis=0)
        y_out = np.concatenate([y, np.full(n_synthetic, minority_label, dtype=y.dtype)])

        # Shuffle
        perm = rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]
