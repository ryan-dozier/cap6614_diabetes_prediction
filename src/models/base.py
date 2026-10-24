from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.experiments.metrics import probabilities, validate_threshold
from src.experiments.config import positive_integer


class ModelAdapter(ABC):
    """fit prepares a predictor; it need not imply gradient-based training."""

    artifact_name = "model"

    def __init__(
        self,
        features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        target: str,
    ) -> None:
        forbidden = {"id", "row_index", "seqn", "diagnosed_diabetes", target.lower()}
        if not features or len(set(features)) != len(features):
            raise ValueError("features must be nonempty and unique.")
        if any(not isinstance(name, str) or name.lower() in forbidden for name in features):
            raise ValueError("Predictors must be named features without identifiers or the target.")
        if len(set(categorical_features)) != len(categorical_features) or not set(categorical_features) <= set(features):
            raise ValueError("categorical_features must be a unique subset of features.")
        self.features = features
        self.categorical_features = categorical_features

    def _inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != self.features:
            raise ValueError(f"Predictors must have exact feature names and order: {self.features}")
        if X.empty or X.isna().any().any():
            raise ValueError("Predictors must be nonempty and contain no missing values.")
        result = X.copy()
        for feature in self.features:
            column = result[feature]
            if feature in self.categorical_features:
                if column.map(lambda value: isinstance(value, (float, np.floating)) and not np.isfinite(value)).any():
                    raise ValueError(f"{feature}: nonfinite categorical input.")
                result[feature] = column.astype(str)
            else:
                if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_complex_dtype(column):
                    raise ValueError(f"{feature}: expected finite numeric input.")
                if not np.isfinite(column.to_numpy(dtype=float)).all():
                    raise ValueError(f"{feature}: nonfinite numeric input.")
        return result

    def predict_proba_batched(
        self, X: pd.DataFrame, batch_size: int | None,
    ) -> NDArray[np.float64]:
        if batch_size is None:
            return self.predict_proba(X)
        positive_integer(batch_size, "prediction_batch_size")
        if X.empty:
            raise ValueError("Predictors must be nonempty.")
        outputs = []
        for start in range(0, len(X), batch_size):
            chunk = X.iloc[start:start + batch_size]
            probability = self.predict_proba(chunk)
            if probability.shape != (len(chunk),):
                raise ValueError("Prediction chunk did not preserve its row count.")
            outputs.append(probability)
        return np.concatenate(outputs)

    def _positive_probability(
        self, raw: NDArray, classes: NDArray, n_rows: int,
    ) -> NDArray[np.float64]:
        if raw.shape != (n_rows, 2) or classes.shape != (2,) or set(classes) != {0, 1}:
            raise ValueError("Estimator returned unexpected probability shape or class labels.")
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError("Estimator returned invalid probabilities.")
        if not np.allclose(raw.sum(axis=1), 1):
            raise ValueError("Estimator probability rows must sum to 1.")
        positive_column = int(np.flatnonzero(classes == 1)[0])
        return probabilities(raw[:, positive_column])

    def synchronize(self) -> None:
        """CPU adapters require no synchronization."""

    def reset_peak_memory(self) -> None:
        """CPU adapters do not measure GPU memory."""

    def resource_metadata(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Return one-dimensional P(y=1) in input row order."""

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> NDArray[np.int64]:
        validate_threshold(threshold)
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    @abstractmethod
    def save(self, output_path: Path) -> None:
        pass

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        pass
