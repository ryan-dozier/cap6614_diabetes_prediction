from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.experiments.metrics import binary_labels
from src.models.base import ModelAdapter
from src.models.ft_transformer_config import (
    FT_SUPPORTED_PARAMS,
    resolve_ft_transformer_params,
    validate_ft_transformer_params,
)

ARTIFACT_FORMAT_VERSION = 1
SELECTION_POLICY = (
    "Full same-domain validation each epoch; highest raw ROC-AUC checkpoint, "
    "earliest exact tie; patience resets at min_delta improvement over the "
    "last patience reference."
)
_ARCHITECTURE_KEYS = (
    "n_blocks",
    "d_block",
    "attention_n_heads",
    "attention_dropout",
    "ffn_d_hidden",
    "ffn_d_hidden_multiplier",
    "ffn_activation",
    "ffn_dropout",
    "residual_dropout",
)


def _runtime() -> tuple[Any, Any]:
    try:
        import torch
        from rtdl_revisiting_models import FTTransformer
    except ImportError as exc:
        raise RuntimeError(
            "FT-Transformer requires the declared PyTorch and "
            "rtdl-revisiting-models==0.0.2 dependencies."
        ) from exc

    return torch, FTTransformer


class FTTransformerAdapter(ModelAdapter):
    """Task-trained FT-Transformer adapter with fitted preprocessing and persistence."""

    artifact_name = "model.ft_transformer.pt"

    def __init__(
        self,
        features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        target: str,
        *,
        seed: int,
        model_params: dict[str, Any] | None = None,
        prediction_batch_size: int = 256,
    ) -> None:
        super().__init__(features, categorical_features, target)
        categorical_set = set(self.categorical_features)
        self.categorical_features = tuple(
            name for name in self.features if name in categorical_set
        )
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if type(prediction_batch_size) is not int or prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be a positive integer.")
        if model_params is not None and not isinstance(model_params, dict):
            raise ValueError("model_params must be a dictionary.")
        supplied = {} if model_params is None else deepcopy(model_params)
        self.params = resolve_ft_transformer_params(supplied)
        self.target = target
        self.seed = seed
        self.prediction_batch_size = prediction_batch_size

        self._torch, self._ft_transformer = _runtime()
        self._device = self._check_device(self.params["device"])
        self._model: Any = None
        self._ready = False
        self._preprocessing_fitted = False
        self._numeric_features = tuple(
            name for name in self.features if name not in self.categorical_features
        )
        self._ordered_categorical_features = tuple(
            name for name in self.features if name in self.categorical_features
        )
        self._numeric_means: tuple[float, ...] = ()
        self._numeric_scales: tuple[float, ...] = ()
        self._category_vocabularies: dict[str, tuple[str, ...]] = {}
        self._category_training_counts: dict[str, dict[str, int]] = {}
        self._category_fallback_indices: dict[str, int] = {}
        self._category_fallback_values: dict[str, str] = {}
        self._training_metadata: dict[str, Any] = {}
        self._determinism_settings: dict[str, Any] = {}
        self._artifact_path: str | None = None
        self._peak_memory_measured = False
        self._training_started = False

    def _check_device(self, device_name: str) -> Any:
        device = self._torch.device(device_name)
        if device.type == "cuda" and (
            not self._torch.cuda.is_available()
            or device.index is None
            or device.index >= self._torch.cuda.device_count()
        ):
            raise RuntimeError(f"CUDA device {device_name} unavailable; CPU fallback is disabled.")
        return device

    def _fit_preprocessing(self, X: pd.DataFrame) -> None:
        """Fit and store train-only transforms without marking the adapter ready."""
        self._fit_preprocessing_normalized(self._inputs(X))

    def _fit_preprocessing_normalized(self, normalized: pd.DataFrame) -> None:
        if self._preprocessing_fitted:
            raise RuntimeError("Preprocessing is already fitted; create a new adapter to refit.")

        means: tuple[float, ...] = ()
        scales: tuple[float, ...] = ()
        if self._numeric_features:
            numeric = normalized.loc[:, self._numeric_features].to_numpy(dtype=np.float64)
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.fit(numeric)
            fitted_means = np.asarray(scaler.mean_, dtype=np.float64)
            fitted_scales = np.asarray(scaler.scale_, dtype=np.float64)
            if not np.isfinite(fitted_means).all() or not np.isfinite(fitted_scales).all():
                raise ValueError("Fitted numerical preprocessing contains non-finite statistics.")
            means = tuple(float(value) for value in fitted_means)
            scales = tuple(float(value) for value in fitted_scales)

        vocabularies: dict[str, tuple[str, ...]] = {}
        training_counts: dict[str, dict[str, int]] = {}
        fallback_indices: dict[str, int] = {}
        fallback_values: dict[str, str] = {}
        for feature in self._ordered_categorical_features:
            counts = normalized[feature].value_counts(sort=False).to_dict()
            vocabulary = tuple(sorted(counts))
            fallback = min(vocabulary, key=lambda category: (-int(counts[category]), category))
            vocabularies[feature] = vocabulary
            training_counts[feature] = {
                category: int(counts[category]) for category in vocabulary
            }
            fallback_indices[feature] = vocabulary.index(fallback)
            fallback_values[feature] = fallback

        self._numeric_means = means
        self._numeric_scales = scales
        self._category_vocabularies = vocabularies
        self._category_training_counts = training_counts
        self._category_fallback_indices = fallback_indices
        self._category_fallback_values = fallback_values
        self._preprocessing_fitted = True

    def _restore_preprocessing(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("Artifact preprocessing state must be a mapping.")
        required = {
            "numeric_features", "numeric_means", "numeric_scales",
            "categorical_features", "category_vocabularies",
            "category_training_counts", "category_cardinalities",
            "category_fallback_indices", "category_fallback_values",
            "unknown_category_policy",
        }
        if set(state) != required:
            raise ValueError("Artifact preprocessing fields are incomplete or unsupported.")
        if state["numeric_features"] != list(self._numeric_features):
            raise ValueError("Artifact numeric feature order is incompatible.")
        if state["categorical_features"] != list(self._ordered_categorical_features):
            raise ValueError("Artifact categorical feature order is incompatible.")
        if state["unknown_category_policy"] != "training_mode":
            raise ValueError("Artifact unknown-category policy is incompatible.")

        means, scales = state["numeric_means"], state["numeric_scales"]
        if (
            not isinstance(means, list)
            or not isinstance(scales, list)
            or len(means) != len(self._numeric_features)
            or len(scales) != len(self._numeric_features)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in means)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in scales)
        ):
            raise ValueError("Artifact numerical means/scales are invalid.")

        vocabularies = state["category_vocabularies"]
        counts = state["category_training_counts"]
        cardinalities = state["category_cardinalities"]
        fallback_indices = state["category_fallback_indices"]
        fallback_values = state["category_fallback_values"]
        if not all(isinstance(value, dict) for value in (
            vocabularies, counts, fallback_indices, fallback_values,
        )) or not isinstance(cardinalities, list):
            raise ValueError("Artifact categorical preprocessing state is invalid.")
        expected_categories = set(self._ordered_categorical_features)
        if any(
            set(value) != expected_categories
            for value in (vocabularies, counts, fallback_indices, fallback_values)
        ) or len(cardinalities) != len(self._ordered_categorical_features):
            raise ValueError("Artifact categorical preprocessing schema is inconsistent.")

        restored_vocabularies: dict[str, tuple[str, ...]] = {}
        restored_counts: dict[str, dict[str, int]] = {}
        restored_indices: dict[str, int] = {}
        restored_values: dict[str, str] = {}
        for feature, cardinality in zip(self._ordered_categorical_features, cardinalities):
            vocabulary = vocabularies[feature]
            feature_counts = counts[feature]
            fallback_index = fallback_indices[feature]
            fallback_value = fallback_values[feature]
            if (
                not isinstance(vocabulary, list)
                or not vocabulary
                or any(not isinstance(value, str) for value in vocabulary)
                or vocabulary != sorted(set(vocabulary))
                or type(cardinality) is not int
                or cardinality != len(vocabulary)
                or not isinstance(feature_counts, dict)
                or set(feature_counts) != set(vocabulary)
                or any(type(count) is not int or count <= 0 for count in feature_counts.values())
                or type(fallback_index) is not int
                or not 0 <= fallback_index < cardinality
                or not isinstance(fallback_value, str)
                or vocabulary[fallback_index] != fallback_value
            ):
                raise ValueError(f"Artifact categorical state for {feature!r} is invalid.")
            maximum = max(feature_counts.values())
            expected_fallback = min(
                value for value in vocabulary if feature_counts[value] == maximum
            )
            if fallback_value != expected_fallback:
                raise ValueError(f"Artifact fallback for {feature!r} is inconsistent with training counts.")
            restored_vocabularies[feature] = tuple(vocabulary)
            restored_counts[feature] = {
                value: feature_counts[value] for value in vocabulary
            }
            restored_indices[feature] = fallback_index
            restored_values[feature] = fallback_value

        self._numeric_means = tuple(float(value) for value in means)
        self._numeric_scales = tuple(float(value) for value in scales)
        self._category_vocabularies = restored_vocabularies
        self._category_training_counts = restored_counts
        self._category_fallback_indices = restored_indices
        self._category_fallback_values = restored_values
        self._preprocessing_fitted = True

    def _validate_labels(
        self, labels: object, X: pd.DataFrame, name: str,
    ) -> NDArray[np.int64]:
        if isinstance(labels, pd.Series) and not labels.index.equals(X.index):
            raise ValueError(f"{name}: label index must exactly match predictor index and order.")
        result = binary_labels(np.asarray(labels))
        if len(result) != len(X):
            raise ValueError(f"{name}: predictor and label lengths must match.")
        if len(np.unique(result)) != 2:
            raise ValueError(f"{name}: labels must contain both 0 and 1.")
        return result

    def _clear_fitted_state(self) -> None:
        self._model = None
        self._ready = False
        self._preprocessing_fitted = False
        self._numeric_means = ()
        self._numeric_scales = ()
        self._category_vocabularies = {}
        self._category_training_counts = {}
        self._category_fallback_indices = {}
        self._category_fallback_values = {}
        self._training_metadata = {}
        self._determinism_settings = {}
        self._artifact_path = None
        self._peak_memory_measured = False

    @staticmethod
    def _training_fingerprint(
        normalized: pd.DataFrame,
        labels: NDArray[np.int64],
        features: tuple[str, ...],
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
    ) -> str:
        digest = hashlib.sha256()
        schema = {
            "version": 1,
            "features": list(features),
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "numeric_dtype": "<f8",
            "label_dtype": "<i8",
        }
        digest.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(len(normalized).to_bytes(8, "little"))
        if numeric_features:
            numeric = normalized.loc[:, numeric_features].to_numpy(dtype="<f8", copy=True)
            digest.update(numeric.tobytes(order="C"))
        for feature in categorical_features:
            encoded = json.dumps(
                normalized[feature].tolist(), ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
        return digest.hexdigest()

    @contextmanager
    def _backend_settings(self) -> Iterator[dict[str, Any]]:
        torch = self._torch
        backend = torch.backends
        saved = {
            "cuda_matmul_allow_tf32": backend.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": backend.cudnn.allow_tf32,
            "cudnn_benchmark": backend.cudnn.benchmark,
            "cudnn_deterministic": backend.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        settings = {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": False,
        }
        try:
            backend.cuda.matmul.allow_tf32 = False
            backend.cudnn.allow_tf32 = False
            backend.cudnn.benchmark = False
            backend.cudnn.deterministic = True
            torch.use_deterministic_algorithms(False)
            settings.update({
                "cuda_matmul_allow_tf32": bool(backend.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(backend.cudnn.allow_tf32),
                "cudnn_benchmark": bool(backend.cudnn.benchmark),
                "cudnn_deterministic": bool(backend.cudnn.deterministic),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            })
            yield settings
        finally:
            backend.cuda.matmul.allow_tf32 = saved["cuda_matmul_allow_tf32"]
            backend.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
            backend.cudnn.benchmark = saved["cudnn_benchmark"]
            backend.cudnn.deterministic = saved["cudnn_deterministic"]
            torch.use_deterministic_algorithms(
                saved["deterministic_algorithms"],
                warn_only=saved["deterministic_warn_only"],
            )

    @contextmanager
    def _training_environment(self) -> Iterator[dict[str, Any]]:
        torch = self._torch
        devices = [self._device.index] if self._device.type == "cuda" else []
        with self._backend_settings() as backend_settings:
            settings = {
                **backend_settings,
                "seed": self.seed,
                "shuffle_generator": "dedicated CPU torch.Generator seeded with seed + 1 modulo 2**32",
                "bitwise_cross_device_guarantee": False,
            }
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed)
                if self._device.type == "cuda":
                    with torch.cuda.device(self._device):
                        torch.cuda.manual_seed(self.seed)
                yield settings

    def _tensor_batch(
        self,
        numerical: NDArray[np.float32] | None,
        categorical: NDArray[np.int64] | None,
        labels: NDArray[np.int64] | None = None,
    ) -> tuple[Any | None, Any | None, Any | None]:
        x_cont, x_cat = self._to_tensors(numerical, categorical)
        y = None if labels is None else self._torch.as_tensor(
            labels, dtype=self._torch.float32, device=self._device,
        )
        return x_cont, x_cat, y

    def _train_epoch(
        self,
        model: Any,
        optimizer: Any,
        train_numeric: NDArray[np.float32] | None,
        train_categorical: NDArray[np.int64] | None,
        labels: NDArray[np.int64],
        generator: Any,
    ) -> float:
        torch = self._torch
        model.train()
        order = torch.randperm(len(labels), generator=generator).numpy()
        weighted_loss = 0.0
        batch_size = self.params["train_batch_size"]
        for start in range(0, len(labels), batch_size):
            indices = order[start:start + batch_size]
            x_cont, x_cat, y = self._tensor_batch(
                None if train_numeric is None else train_numeric[indices],
                None if train_categorical is None else train_categorical[indices],
                labels[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_cont, x_cat)
            if tuple(logits.shape) != (len(indices), 1) or not torch.isfinite(logits).all().item():
                raise RuntimeError("Training produced invalid or non-finite logits.")
            if y is None:
                raise RuntimeError("Training batch labels were not constructed.")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training produced a non-finite BCE-with-logits loss.")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not torch.stack([
                torch.isfinite(gradient).all() for gradient in gradients
            ]).all().item():
                raise RuntimeError("Training produced missing or non-finite gradients.")
            optimizer.step()
            parameters = list(model.parameters())
            if not torch.stack([
                torch.isfinite(parameter).all() for parameter in parameters
            ]).all().item():
                raise RuntimeError("Optimizer step produced non-finite model parameters.")
            weighted_loss += float(loss.detach().item()) * len(indices)
        return weighted_loss / len(labels)

    def _evaluate_validation(
        self,
        model: Any,
        numerical: NDArray[np.float32] | None,
        categorical: NDArray[np.int64] | None,
        labels: NDArray[np.int64],
    ) -> tuple[float, float]:
        torch = self._torch
        previous_modes = tuple((module, module.training) for module in model.modules())
        losses: list[float] = []
        probabilities: list[NDArray[np.float64]] = []
        try:
            model.eval()
            with torch.inference_mode():
                batch_size = self.params["validation_batch_size"]
                for start in range(0, len(labels), batch_size):
                    stop = min(start + batch_size, len(labels))
                    x_cont, x_cat, y = self._tensor_batch(
                        None if numerical is None else numerical[start:stop],
                        None if categorical is None else categorical[start:stop],
                        labels[start:stop],
                    )
                    logits = model(x_cont, x_cat)
                    if tuple(logits.shape) != (stop - start, 1) or not torch.isfinite(logits).all().item():
                        raise RuntimeError("Validation produced invalid or non-finite logits.")
                    if y is None:
                        raise RuntimeError("Validation batch labels were not constructed.")
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Validation produced a non-finite BCE-with-logits loss.")
                    losses.append(float(loss.item()) * (stop - start))
                    batch_logits = logits.detach().to("cpu", dtype=torch.float64).numpy()[:, 0]
                    probabilities.append(self._sigmoid_probabilities(batch_logits, stop - start))
        finally:
            for module, was_training in previous_modes:
                module.training = was_training
        validation_loss = sum(losses) / len(labels)
        scores = np.concatenate(probabilities)
        if scores.shape != (len(labels),) or not np.isfinite(scores).all():
            raise RuntimeError("Validation produced invalid probabilities.")
        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError as exc:
            raise RuntimeError("Validation ROC-AUC is undefined.") from exc
        if not math.isfinite(validation_loss) or not math.isfinite(auc):
            raise RuntimeError("Validation produced non-finite loss or ROC-AUC.")
        return validation_loss, auc

    @staticmethod
    def _state_snapshot(model: Any) -> dict[str, Any]:
        return {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in model.state_dict().items()
        }

    def _record_selection(
        self,
        state: dict[str, Any],
        *,
        epoch: int,
        auc: float,
        model: Any,
    ) -> bool:
        if not math.isfinite(auc):
            raise RuntimeError("Checkpoint selection requires a finite validation ROC-AUC.")
        if state["best_auc"] is None or auc > state["best_auc"]:
            state["best_auc"] = float(auc)
            state["best_epoch"] = epoch
            state["best_state"] = self._state_snapshot(model)
        reference = state["patience_reference"]
        if reference is None:
            state["patience_reference"] = float(auc)
            state["patience_counter"] = 0
        elif auc - reference >= self.params["min_delta"]:
            state["patience_reference"] = float(auc)
            state["patience_counter"] = 0
        else:
            state["patience_counter"] += 1
        return state["patience_counter"] >= self.params["patience"]

    def _fingerprint(
        self, normalized: pd.DataFrame, labels: NDArray[np.int64],
    ) -> str:
        return self._training_fingerprint(
            normalized,
            labels,
            self.features,
            self._numeric_features,
            self._ordered_categorical_features,
        )

    def _require_preprocessing(self) -> None:
        if not self._preprocessing_fitted:
            raise RuntimeError("FT-Transformer preprocessing must be fitted before this operation.")

    def _map_categories(
        self, normalized: pd.DataFrame,
    ) -> tuple[NDArray[np.int64] | None, dict[str, int]]:
        encoded_columns: list[NDArray[np.int64]] = []
        unseen_counts: dict[str, int] = {}
        for feature in self._ordered_categorical_features:
            vocabulary = self._category_vocabularies[feature]
            lookup = {value: index for index, value in enumerate(vocabulary)}
            fallback_index = self._category_fallback_indices[feature]
            values = normalized[feature].to_numpy(dtype=object)
            encoded = np.empty(len(values), dtype=np.int64)
            unseen_count = 0
            for index, value in enumerate(values):
                category_index = lookup.get(value)
                if category_index is None:
                    encoded[index] = fallback_index
                    unseen_count += 1
                else:
                    encoded[index] = category_index
            encoded_columns.append(encoded)
            unseen_counts[feature] = unseen_count
        if not encoded_columns:
            return None, unseen_counts
        return np.column_stack(encoded_columns).astype(np.int64, copy=False), unseen_counts

    def _transform_inputs(
        self, X: pd.DataFrame,
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.int64] | None]:
        self._require_preprocessing()
        return self._transform_normalized(self._inputs(X))

    def _transform_normalized(
        self, normalized: pd.DataFrame,
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.int64] | None]:
        numerical: NDArray[np.float32] | None = None
        if self._numeric_features:
            numeric = normalized.loc[:, self._numeric_features].to_numpy(dtype=np.float64)
            transformed = (numeric - np.asarray(self._numeric_means)) / np.asarray(self._numeric_scales)
            with np.errstate(over="ignore", invalid="ignore"):
                numerical = transformed.astype(np.float32)
            if not np.isfinite(numerical).all():
                raise ValueError("Numerical preprocessing produced non-finite float32 values.")
        categorical, _ = self._map_categories(normalized)
        return numerical, categorical

    def unseen_category_counts(self, X: pd.DataFrame) -> dict[str, int]:
        """Return fresh per-feature unknown counts without inference or state changes."""
        self._require_preprocessing()
        normalized = self._inputs(X)
        _, unseen_counts = self._map_categories(normalized)
        return unseen_counts

    def _construct_model(self) -> Any:
        self._require_preprocessing()
        defaults = self._ft_transformer.get_default_kwargs(self.params["n_blocks"])
        default_flag = bool(defaults.pop("_is_default"))
        architecture = {name: self.params[name] for name in _ARCHITECTURE_KEYS}
        kwargs = {**defaults, **architecture}
        if kwargs["ffn_d_hidden"] is not None:
            kwargs["ffn_d_hidden_multiplier"] = None
        kwargs["ffn_activation"] = self.params["ffn_activation"]
        is_default_architecture = (
            default_flag
            and self.params["ffn_activation"] == "ReGLU"
            and all(kwargs[name] == defaults[name] for name in defaults)
        )
        cardinalities = [
            len(self._category_vocabularies[name])
            for name in self._ordered_categorical_features
        ]
        model = self._ft_transformer(
            n_cont_features=len(self._numeric_features),
            cat_cardinalities=cardinalities,
            d_out=1,
            _is_default=is_default_architecture,
            **kwargs,
        )
        model = model.to(device=self._device, dtype=self._torch.float32)
        return model

    def _to_tensors(
        self,
        numerical: NDArray[np.float32] | None,
        categorical: NDArray[np.int64] | None,
    ) -> tuple[Any | None, Any | None]:
        x_cont = None if numerical is None else self._torch.as_tensor(
            numerical, dtype=self._torch.float32, device=self._device,
        )
        x_cat = None if categorical is None else self._torch.as_tensor(
            categorical, dtype=self._torch.int64, device=self._device,
        )
        return x_cont, x_cat

    def _predict_logits(
        self,
        X: pd.DataFrame,
        *,
        batch_size: int | None = None,
        model: Any | None = None,
    ) -> NDArray[np.float64]:
        self._require_preprocessing()
        inference_model = self._model if model is None else model
        if inference_model is None:
            raise RuntimeError("An FT-Transformer model must be constructed before inference.")
        size = self.prediction_batch_size if batch_size is None else batch_size
        if type(size) is not int or size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        numerical, categorical = self._transform_inputs(X)
        training_states = tuple(
            (module, module.training) for module in inference_model.modules()
        )
        outputs: list[NDArray[np.float64]] = []
        try:
            inference_model.eval()
            with self._backend_settings(), self._torch.inference_mode():
                for start in range(0, len(X), size):
                    stop = min(start + size, len(X))
                    x_cont, x_cat = self._to_tensors(
                        None if numerical is None else numerical[start:stop],
                        None if categorical is None else categorical[start:stop],
                    )
                    logits = inference_model(x_cont, x_cat)
                    if tuple(logits.shape) != (stop - start, 1):
                        raise ValueError(
                            "FT-Transformer must return exactly one logit per input row."
                        )
                    if not self._torch.isfinite(logits).all().item():
                        raise ValueError("FT-Transformer returned non-finite logits.")
                    outputs.append(logits.detach().to("cpu", dtype=self._torch.float64).numpy()[:, 0])
        finally:
            for module, was_training in training_states:
                module.training = was_training
        if not outputs:
            raise ValueError("Predictors must be nonempty.")
        return np.concatenate(outputs)

    @staticmethod
    def _sigmoid_probabilities(logits: NDArray[np.float64], n_rows: int) -> NDArray[np.float64]:
        if logits.shape != (n_rows,) or not np.isfinite(logits).all():
            raise ValueError("FT-Transformer returned invalid raw logits.")
        probabilities = np.empty(n_rows, dtype=np.float64)
        positive = logits >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        negative_exp = np.exp(logits[~positive])
        probabilities[~positive] = negative_exp / (1.0 + negative_exp)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("FT-Transformer produced invalid positive-class probabilities.")
        return probabilities

    def _require_ready(self) -> Any:
        if not self._ready or not self._preprocessing_fitted or self._model is None:
            raise RuntimeError("FT-Transformer must complete training before prediction or saving.")
        return self._model

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        if self._training_started or self._preprocessing_fitted or self._ready:
            raise RuntimeError("FT-Transformer refitting is unsupported; create a new adapter.")
        self._training_started = True
        self._ready = False
        self._model = None
        try:
            train = self._inputs(X_train)
            validation = self._inputs(X_val)
            train_labels = self._validate_labels(y_train, train, "training")
            validation_labels = self._validate_labels(y_val, validation, "validation")
            self._fit_preprocessing_normalized(train)
            train_numeric, train_categorical = self._transform_normalized(train)
            validation_numeric, validation_categorical = self._transform_normalized(validation)

            fingerprint = self._fingerprint(train, train_labels)
            validation_unseen = self.unseen_category_counts(validation)
            selection: dict[str, Any] = {
                "best_auc": None,
                "best_epoch": None,
                "best_state": None,
                "patience_reference": None,
                "patience_counter": 0,
            }
            history: list[dict[str, Any]] = []
            print(
                f"FT-Transformer training start: {len(train)} train / "
                f"{len(validation)} validation rows on {self._device} "
                f"for at most {self.params['max_epochs']} epochs.",
                flush=True,
            )
            self.reset_peak_memory()
            with self._training_environment() as determinism:
                model = self._construct_model()
                optimizer = self._torch.optim.AdamW(
                    model.make_parameter_groups(),
                    lr=self.params["learning_rate"],
                    weight_decay=self.params["weight_decay"],
                )
                generator = self._torch.Generator(device="cpu")
                generator.manual_seed((self.seed + 1) % 2**32)
                for epoch in range(1, self.params["max_epochs"] + 1):
                    self.synchronize()
                    train_start = time.perf_counter()
                    training_loss = self._train_epoch(
                        model,
                        optimizer,
                        train_numeric,
                        train_categorical,
                        train_labels,
                        generator,
                    )
                    self.synchronize()
                    training_duration = time.perf_counter() - train_start

                    self.synchronize()
                    validation_start = time.perf_counter()
                    validation_loss, validation_auc = self._evaluate_validation(
                        model,
                        validation_numeric,
                        validation_categorical,
                        validation_labels,
                    )
                    self.synchronize()
                    validation_duration = time.perf_counter() - validation_start
                    should_stop = self._record_selection(
                        selection,
                        epoch=epoch,
                        auc=validation_auc,
                        model=model,
                    )
                    history.append({
                        "epoch": epoch,
                        "training_loss": float(training_loss),
                        "validation_loss": float(validation_loss),
                        "validation_roc_auc": float(validation_auc),
                        "training_duration_seconds": float(training_duration),
                        "validation_duration_seconds": float(validation_duration),
                    })
                    if epoch == 1 or epoch % 10 == 0 or should_stop or epoch == self.params["max_epochs"]:
                        print(
                            f"FT-Transformer epoch {epoch}: "
                            f"train_loss={training_loss:.6f}, "
                            f"val_loss={validation_loss:.6f}, "
                            f"val_auc={validation_auc:.6f}",
                            flush=True,
                        )
                    if should_stop:
                        stopping_reason = "patience"
                        break
                else:
                    stopping_reason = "max_epochs"

                if selection["best_state"] is None or selection["best_epoch"] is None:
                    raise RuntimeError("Training completed without a selectable finite checkpoint.")
                model.load_state_dict(selection["best_state"], strict=True)
                model.eval()
                self._model = model
                self._determinism_settings = dict(determinism)

            self._training_metadata = {
                "training_population_size": int(len(train)),
                "validation_population_size": int(len(validation)),
                "training_class_counts": {
                    "0": int(np.count_nonzero(train_labels == 0)),
                    "1": int(np.count_nonzero(train_labels == 1)),
                },
                "validation_class_counts": {
                    "0": int(np.count_nonzero(validation_labels == 0)),
                    "1": int(np.count_nonzero(validation_labels == 1)),
                },
                "validation_unseen_category_counts": dict(validation_unseen),
                "training_fingerprint": fingerprint,
                "training_fingerprint_format": "sha256 over schema JSON, ordered raw normalized numeric/category values, and int64 labels; see adapter implementation",
                "training_history": deepcopy(history),
                "best_epoch": int(selection["best_epoch"]),
                "best_validation_roc_auc": float(selection["best_auc"]),
                "epochs_executed": len(history),
                "patience_reference": float(selection["patience_reference"]),
                "patience_counter": int(selection["patience_counter"]),
                "stopping_reason": stopping_reason,
                "validation_population_policy": "complete fixed same-domain validation split, evaluated every epoch in batches",
                "validation_selection_policy": SELECTION_POLICY,
            }
            self._ready = True
            print(
                f"FT-Transformer selected epoch {selection['best_epoch']} "
                f"(AUC={selection['best_auc']:.6f}); stopped by {stopping_reason} "
                f"after {len(history)} epochs.",
                flush=True,
            )
        except Exception:
            self._clear_fitted_state()
            raise

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        model = self._require_ready()
        logits = self._predict_logits(X, model=model)
        return self._sigmoid_probabilities(logits, len(X))

    def save(self, output_path: Path) -> None:
        model = self._require_ready()
        output_path = Path(output_path)
        if output_path.name != self.artifact_name:
            raise ValueError(f"FT-Transformer artifacts must be named {self.artifact_name}.")
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"Artifact parent directory does not exist: {output_path.parent}")
        state_dict = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in model.state_dict().items()
        }
        payload = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "adapter": type(self).__name__,
            "constructor": {
                "features": list(self.features),
                "categorical_features": list(self.categorical_features),
                "target": self.target,
                "seed": self.seed,
                "prediction_batch_size": self.prediction_batch_size,
                "model_params": deepcopy(self.params),
            },
            "schema": {
                "features": list(self.features),
                "numeric_features": list(self._numeric_features),
                "categorical_features": list(self._ordered_categorical_features),
            },
            "preprocessing": self._preprocessing_metadata(),
            "class_meaning": {"positive_label": 1, "probability": "P(y=1)"},
            "versions": self._versions(),
            "device": str(self._device),
            "precision": "float32",
            "determinism": deepcopy(self._determinism_settings),
            "training": deepcopy(self._training_metadata),
            "state_dict": state_dict,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("wb") as handle:
                self._torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._artifact_path = str(output_path)

    @classmethod
    def load(cls, output_path: Path, *, device: str) -> FTTransformerAdapter:
        torch, _ = _runtime()
        output_path = Path(output_path)
        if output_path.name != cls.artifact_name:
            raise ValueError(f"FT-Transformer artifacts must be named {cls.artifact_name}.")
        if not output_path.is_file():
            raise FileNotFoundError(f"FT-Transformer artifact does not exist: {output_path}")
        if not isinstance(device, str) or not (
            device == "cpu"
            or (device.startswith("cuda:") and device[5:].isascii() and device[5:].isdigit())
        ):
            raise ValueError("device must be explicit: 'cpu' or 'cuda:<index>'.")
        try:
            payload = torch.load(output_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Unable to read FT-Transformer artifact {output_path}.") from exc
        required = {
            "format_version", "adapter", "constructor", "schema", "preprocessing",
            "class_meaning", "versions", "device", "precision", "determinism",
            "training", "state_dict",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("FT-Transformer artifact is incomplete or contains unsupported fields.")
        if type(payload["format_version"]) is not int or payload["format_version"] != ARTIFACT_FORMAT_VERSION:
            raise ValueError("Unsupported FT-Transformer artifact format version.")
        if payload["adapter"] != cls.__name__:
            raise ValueError("Artifact adapter type is incompatible.")
        versions = payload["versions"]
        if not isinstance(versions, dict):
            raise ValueError("Artifact version metadata is invalid.")
        current_versions = cls._versions()
        for name in ("rtdl-revisiting-models", "torch"):
            if versions.get(name) != current_versions[name]:
                raise RuntimeError(
                    f"Artifact {name} version {versions.get(name)!r} is incompatible "
                    f"with installed {current_versions[name]!r}."
                )
        if not isinstance(payload["device"], str):
            raise ValueError("Artifact device metadata is invalid.")
        if payload["device"] != device:
            raise RuntimeError(
                f"Artifact was trained on {payload['device']!r}; requested {device!r}. "
                "Cross-device reload is unsupported."
            )
        if payload["precision"] != "float32":
            raise ValueError("Artifact precision is unsupported.")
        constructor = payload["constructor"]
        if not isinstance(constructor, dict) or set(constructor) != {
            "features", "categorical_features", "target", "seed",
            "prediction_batch_size", "model_params",
        }:
            raise ValueError("Artifact constructor metadata is incomplete.")
        features = constructor["features"]
        categorical_features = constructor["categorical_features"]
        if (
            not isinstance(features, list)
            or not isinstance(categorical_features, list)
            or any(not isinstance(name, str) for name in features + categorical_features)
            or not isinstance(constructor["target"], str)
            or type(constructor["seed"]) is not int
            or type(constructor["prediction_batch_size"]) is not int
        ):
            raise ValueError("Artifact feature schema is invalid.")
        schema = payload["schema"]
        if not isinstance(schema, dict) or set(schema) != {
            "features", "numeric_features", "categorical_features",
        }:
            raise ValueError("Artifact schema metadata is incomplete.")
        if schema != {
            "features": features,
            "numeric_features": [name for name in features if name not in categorical_features],
            "categorical_features": [name for name in features if name in categorical_features],
        } or categorical_features != schema["categorical_features"]:
            raise ValueError("Artifact feature schema/order is inconsistent.")
        if payload["class_meaning"] != {"positive_label": 1, "probability": "P(y=1)"}:
            raise ValueError("Artifact positive-class meaning is unsupported.")
        determinism = payload["determinism"]
        training = payload["training"]
        state_dict = payload["state_dict"]
        if not isinstance(determinism, dict) or not isinstance(training, dict):
            raise ValueError("Artifact training metadata is invalid.")
        if set(determinism) != {
            "seed", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
            "cudnn_benchmark", "cudnn_deterministic",
            "deterministic_algorithms", "shuffle_generator",
            "bitwise_cross_device_guarantee",
        } or (
            determinism["seed"] != constructor["seed"]
            or determinism["cuda_matmul_allow_tf32"] is not False
            or determinism["cudnn_allow_tf32"] is not False
            or determinism["cudnn_benchmark"] is not False
            or determinism["cudnn_deterministic"] is not True
            or determinism["deterministic_algorithms"] is not False
            or determinism["bitwise_cross_device_guarantee"] is not False
            or determinism["shuffle_generator"]
            != "dedicated CPU torch.Generator seeded with seed + 1 modulo 2**32"
        ):
            raise ValueError("Artifact determinism metadata is incomplete or inconsistent.")
        training_fields = {
            "training_population_size", "validation_population_size",
            "training_class_counts", "validation_class_counts",
            "validation_unseen_category_counts", "training_fingerprint",
            "training_fingerprint_format", "training_history", "best_epoch",
            "best_validation_roc_auc", "epochs_executed", "patience_reference",
            "patience_counter", "stopping_reason",
            "validation_population_policy", "validation_selection_policy",
        }
        if set(training) != training_fields:
            raise ValueError("Artifact training metadata is incomplete or unsupported.")
        train_size = training["training_population_size"]
        validation_size = training["validation_population_size"]
        epochs = training["epochs_executed"]
        best_epoch = training["best_epoch"]
        history = training["training_history"]
        saved_params = constructor["model_params"]
        if not isinstance(saved_params, dict) or set(saved_params) != FT_SUPPORTED_PARAMS:
            raise ValueError("Artifact model parameters are incomplete or unsupported.")
        try:
            validate_ft_transformer_params(saved_params)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Artifact model parameters are invalid.") from exc
        if (
            type(train_size) is not int or train_size <= 0
            or type(validation_size) is not int or validation_size <= 0
            or type(epochs) is not int or not 1 <= epochs <= saved_params["max_epochs"]
            or type(best_epoch) is not int or not 1 <= best_epoch <= epochs
            or not isinstance(history, list) or len(history) != epochs
            or not isinstance(training["training_fingerprint"], str)
            or len(training["training_fingerprint"]) != 64
            or any(character not in "0123456789abcdef" for character in training["training_fingerprint"])
            or training["stopping_reason"] not in ("patience", "max_epochs")
            or training["validation_selection_policy"] != SELECTION_POLICY
            or training["validation_population_policy"]
            != "complete fixed same-domain validation split, evaluated every epoch in batches"
            or training["training_fingerprint_format"]
            != "sha256 over schema JSON, ordered raw normalized numeric/category values, and int64 labels; see adapter implementation"
        ):
            raise ValueError("Artifact training population, history, or checkpoint metadata is invalid.")
        for size_name, class_counts in (
            ("training", training["training_class_counts"]),
            ("validation", training["validation_class_counts"]),
        ):
            if (
                not isinstance(class_counts, dict)
                or set(class_counts) != {"0", "1"}
                or any(type(count) is not int or count <= 0 for count in class_counts.values())
                or sum(class_counts.values()) != training[f"{size_name}_population_size"]
            ):
                raise ValueError(f"Artifact {size_name} class counts are invalid.")
        unseen_counts = training["validation_unseen_category_counts"]
        if (
            not isinstance(unseen_counts, dict)
            or set(unseen_counts) != set(schema["categorical_features"])
            or any(
                type(count) is not int or not 0 <= count <= validation_size
                for count in unseen_counts.values()
            )
        ):
            raise ValueError("Artifact validation unseen-category diagnostics are invalid.")
        history_fields = {
            "epoch", "training_loss", "validation_loss", "validation_roc_auc",
            "training_duration_seconds", "validation_duration_seconds",
        }
        for index, record in enumerate(history, start=1):
            if (
                not isinstance(record, dict)
                or set(record) != history_fields
                or type(record["epoch"]) is not int
                or record["epoch"] != index
                or any(
                    type(record[name]) not in (int, float) or not math.isfinite(record[name])
                    for name in history_fields - {"epoch"}
                )
                or not 0 <= record["validation_roc_auc"] <= 1
                or record["training_duration_seconds"] < 0
                or record["validation_duration_seconds"] < 0
            ):
                raise ValueError("Artifact epoch history is invalid.")
        if (
            type(training["best_validation_roc_auc"]) not in (int, float)
            or not math.isfinite(training["best_validation_roc_auc"])
            or not 0 <= training["best_validation_roc_auc"] <= 1
            or type(training["patience_reference"]) not in (int, float)
            or not math.isfinite(training["patience_reference"])
            or type(training["patience_counter"]) is not int
            or not 0 <= training["patience_counter"] <= epochs
        ):
            raise ValueError("Artifact checkpoint-selection metadata is invalid.")
        best_history_epoch = min(
            record["epoch"] for record in history
            if record["validation_roc_auc"] == training["best_validation_roc_auc"]
        ) if history else None
        expected_reference: float | None = None
        expected_patience_counter = 0
        for record in history:
            auc = record["validation_roc_auc"]
            if expected_reference is None:
                expected_reference = auc
            elif auc - expected_reference >= saved_params["min_delta"]:
                expected_reference = auc
                expected_patience_counter = 0
            else:
                expected_patience_counter += 1
        expected_stopping_reason = (
            "patience"
            if expected_patience_counter >= saved_params["patience"]
            else "max_epochs"
        )
        if (
            best_history_epoch != best_epoch
            or max(record["validation_roc_auc"] for record in history)
            != training["best_validation_roc_auc"]
            or expected_reference != training["patience_reference"]
            or expected_patience_counter != training["patience_counter"]
            or training["stopping_reason"] != expected_stopping_reason
            or (expected_stopping_reason == "max_epochs" and epochs != saved_params["max_epochs"])
        ):
            raise ValueError("Artifact checkpoint selection and stopping metadata are inconsistent.")
        if not isinstance(state_dict, dict) or not state_dict or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state_dict.items()
        ):
            raise ValueError("Artifact state_dict is invalid.")
        if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in state_dict.values()):
            raise ValueError("Artifact state_dict tensors must be CPU float32 tensors.")
        if any(not torch.isfinite(value).all().item() for value in state_dict.values()):
            raise ValueError("Artifact state_dict contains non-finite tensors.")
        adapter = cls(
            tuple(features),
            tuple(categorical_features),
            constructor["target"],
            seed=constructor["seed"],
            model_params=constructor["model_params"],
            prediction_batch_size=constructor["prediction_batch_size"],
        )
        adapter._restore_preprocessing(payload["preprocessing"])
        with torch.random.fork_rng(devices=[adapter._device.index] if adapter._device.type == "cuda" else []):
            torch.random.default_generator.manual_seed(adapter.seed)
            if adapter._device.type == "cuda":
                with torch.cuda.device(adapter._device):
                    torch.cuda.manual_seed(adapter.seed)
            model = adapter._construct_model()
        try:
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, KeyError) as exc:
            raise ValueError("Artifact model state does not match its architecture.") from exc
        model.eval()
        adapter._model = model
        adapter._determinism_settings = deepcopy(determinism)
        adapter._training_metadata = deepcopy(training)
        adapter._ready = True
        adapter._artifact_path = str(output_path)
        return adapter

    @staticmethod
    def _versions() -> dict[str, str]:
        return {
            "rtdl-revisiting-models": version("rtdl-revisiting-models"),
            "torch": str(_runtime()[0].__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": version("scikit-learn"),
            "python": sys.version.split()[0],
        }

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    def reset_peak_memory(self) -> None:
        if self._device.type == "cuda":
            self.synchronize()
            self._torch.cuda.reset_peak_memory_stats(self._device)
            self._peak_memory_measured = True

    def resource_metadata(self) -> dict[str, Any]:
        if self._device.type != "cuda" or not self._peak_memory_measured:
            return {"peak_gpu_memory": None}
        free_bytes, total_bytes = self._torch.cuda.mem_get_info(self._device)
        return {"peak_gpu_memory": {
            "device": str(self._device),
            "allocated_bytes": int(self._torch.cuda.max_memory_allocated(self._device)),
            "reserved_bytes": int(self._torch.cuda.max_memory_reserved(self._device)),
            "available_bytes_at_query": int(free_bytes),
            "device_total_bytes": int(total_bytes),
            "scope": "PyTorch allocator peaks since adapter reset; available memory is a point-in-time free/total query.",
        }}

    def _preprocessing_metadata(self) -> dict[str, Any]:
        self._require_preprocessing()
        return {
            "numeric_features": list(self._numeric_features),
            "numeric_means": list(self._numeric_means),
            "numeric_scales": list(self._numeric_scales),
            "categorical_features": list(self._ordered_categorical_features),
            "category_vocabularies": {
                feature: list(self._category_vocabularies[feature])
                for feature in self._ordered_categorical_features
            },
            "category_training_counts": deepcopy(self._category_training_counts),
            "category_cardinalities": [
                len(self._category_vocabularies[feature])
                for feature in self._ordered_categorical_features
            ],
            "category_fallback_indices": dict(self._category_fallback_indices),
            "category_fallback_values": dict(self._category_fallback_values),
            "unknown_category_policy": "training_mode",
        }

    def metadata(self) -> dict[str, Any]:
        torch = self._torch
        metadata: dict[str, Any] = {
            "adapter": type(self).__name__,
            "ready": bool(self._ready),
            "model_type": "task-trained",
            "validation_used": bool(self._training_metadata),
            "validation_policy": (
                "Complete same-domain validation evaluated each epoch; validation ROC-AUC selects "
                "the best checkpoint and controls patience."
            ),
            "features": list(self.features),
            "categorical_features": list(self._ordered_categorical_features),
            "target": self.target,
            "seed": self.seed,
            "device": str(self._device),
            "precision": "float32",
            "prediction_batch_size": self.prediction_batch_size,
            "model_params": deepcopy(self.params),
            "compute_settings": {
                "device": str(self._device),
                "device_name": (
                    torch.cuda.get_device_name(self._device)
                    if self._device.type == "cuda" else "CPU"
                ),
                "seed": self.seed,
                "precision": "float32",
                "architecture": {
                    name: self.params[name] for name in _ARCHITECTURE_KEYS
                },
                "determinism": deepcopy(self._determinism_settings),
            },
            "versions": self._versions(),
            "determinism": deepcopy(self._determinism_settings),
            "training": deepcopy(self._training_metadata),
            "resource": self.resource_metadata(),
            "artifact_path": self._artifact_path,
        }
        if self._preprocessing_fitted:
            metadata["preprocessing"] = self._preprocessing_metadata()
        return metadata
