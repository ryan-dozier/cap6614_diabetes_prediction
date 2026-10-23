from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.experiments.config import positive_integer, validate_model_params
from src.experiments.metrics import binary_labels, probabilities
from src.models.base import ModelAdapter

if TYPE_CHECKING:
    from catboost import CatBoostClassifier


class CatBoostAdapter(ModelAdapter):
    def __init__(
        self,
        features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        target: str,
        *,
        seed: int,
        model_params: dict[str, Any],
        early_stopping_rounds: int,
    ) -> None:
        super().__init__(features, categorical_features, target)
        validate_model_params(model_params)
        positive_integer(early_stopping_rounds, "early_stopping_rounds")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        self.params = {
            "iterations": 1000, "depth": 7, "learning_rate": 0.05,
            "loss_function": "Logloss", "eval_metric": "AUC",
            "task_type": "CPU", "thread_count": 4, "verbose": 100,
            **model_params,
            "random_seed": seed,
            "allow_writing_files": False,
            "use_best_model": True,
            "early_stopping_rounds": early_stopping_rounds,
        }
        self._model: CatBoostClassifier | None = None

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

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> None:
        from catboost import CatBoostClassifier

        if self._model is not None:
            raise RuntimeError("This adapter is already fitted; create a new adapter for a new fit.")
        train, val = self._inputs(X_train), self._inputs(X_val)
        train_labels, val_labels = binary_labels(y_train), binary_labels(y_val)
        for X, labels, name in ((train, train_labels, "training"), (val, val_labels, "validation")):
            if len(X) != len(labels) or len(np.unique(labels)) != 2:
                raise ValueError(f"{name}: X/y lengths must match and labels must contain both 0 and 1.")
        model = CatBoostClassifier(**self.params)
        model.fit(
            train, train_labels,
            cat_features=list(self.categorical_features),
            eval_set=(val, val_labels),
        )
        if set(model.classes_) != {0, 1}:
            raise ValueError(f"Unexpected fitted classes: {model.classes_}")
        self._model = model

    def _require_fitted(self) -> CatBoostClassifier:
        if self._model is None:
            raise RuntimeError("The model must be fitted before prediction, saving, or metadata access.")
        return self._model

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        model = self._require_fitted()
        normalized = self._inputs(X)
        raw = np.asarray(model.predict_proba(normalized))
        classes = np.asarray(model.classes_)
        if raw.shape != (len(X), 2) or classes.shape != (2,) or set(classes) != {0, 1}:
            raise ValueError("CatBoost returned an unexpected probability shape or class labels.")
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError("CatBoost returned invalid probabilities.")
        if not np.allclose(raw.sum(axis=1), 1):
            raise ValueError("CatBoost probability rows must sum to 1.")
        positive_column = int(np.flatnonzero(classes == 1)[0])
        return probabilities(raw[:, positive_column])

    def save(self, output_path: Path) -> None:
        if output_path.suffix != ".cbm":
            raise ValueError("CatBoost models must be saved with a .cbm extension.")
        self._require_fitted().save_model(str(output_path), format="cbm")

    def metadata(self) -> dict[str, Any]:
        model = self._require_fitted()
        return {
            "adapter": type(self).__name__,
            "preprocessing": {
                "categorical": "Native CatBoost handling; astype(str) on copies for fit and prediction.",
                "numeric": "Unscaled, unencoded numeric values.",
                "missing_nonfinite": "Rejected; no imputation.",
                "calibration": "None; no probability calibrator fitted.",
            },
            "feature_order": list(self.features),
            "categorical_features": list(self.categorical_features),
            "classes": model.classes_.tolist(),
            "requested_parameters": dict(self.params),
            "effective_parameters": model.get_all_params(),
            "compute_settings": {
                "task_type": self.params["task_type"],
                "thread_count": self.params["thread_count"],
                "seed": self.params["random_seed"],
            },
            "early_stopping_rounds": self.params["early_stopping_rounds"],
            "best_iteration": int(model.get_best_iteration()),
            "fitted_tree_count": int(model.tree_count_),
        }
