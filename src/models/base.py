from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.experiments.metrics import validate_threshold


class ModelAdapter(ABC):
    """fit prepares a predictor; it need not imply gradient-based training."""

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
