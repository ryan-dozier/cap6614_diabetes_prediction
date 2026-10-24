from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.experiments.config import validate_model_params
from src.experiments.metrics import binary_labels
from src.experiments.results import write_json
from src.models.base import ModelAdapter
from src.paths import PROJECT_ROOT

SUPPORTED_VERSION = "9.2.0"
CHECKPOINT_FILENAME = "tabpfn-v3.5-20260909.safetensors"
MODEL_GENERATION = "TabPFN v3.5 (20260909)"


def _runtime() -> tuple[Any, Any]:
    installed = version("tabpfn")
    if installed != SUPPORTED_VERSION:
        raise RuntimeError(f"This adapter supports TabPFN {SUPPORTED_VERSION}; installed {installed}. No automatic upgrade is performed.")
    import torch
    from tabpfn import TabPFNClassifier

    required = {
        "n_estimators", "categorical_features_indices", "model_path", "device",
        "ignore_pretraining_limits", "inference_precision", "fit_mode",
        "memory_saving_mode", "random_state",
    }
    if not required <= set(inspect.signature(TabPFNClassifier).parameters):
        raise RuntimeError("Installed TabPFNClassifier signature is incompatible with the verified 9.2.0 API.")
    if not all(hasattr(TabPFNClassifier, name) for name in ("save_fit_state", "load_from_fit_state")):
        raise RuntimeError("TabPFN fitted-state persistence API is unavailable.")
    return torch, TabPFNClassifier


def _checkpoint_path(configured: str | None) -> Path:
    if configured is None:
        from tabpfn.model_loading import get_cache_dir

        path = get_cache_dir() / CHECKPOINT_FILENAME
    else:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    path = path.resolve()
    if path.name != CHECKPOINT_FILENAME:
        raise ValueError(f"Only the verified {MODEL_GENERATION} checkpoint is supported: {CHECKPOINT_FILENAME}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Local TabPFN checkpoint missing: {path}. Manually obtain the licensed "
            "Prior-Labs/tabpfn_3_5 checkpoint first. This adapter never downloads weights."
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TabPFNAdapter(ModelAdapter):
    artifact_name = "model.tabpfn_fit"

    def __init__(
        self, features: tuple[str, ...], categorical_features: tuple[str, ...],
        target: str, *, seed: int, model_params: dict[str, Any],
    ) -> None:
        super().__init__(features, categorical_features, target)
        validate_model_params(model_params, "tabpfn")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        self.target = target
        self.seed = seed
        self.params = {
            "n_estimators": 4, "device": "cuda:0", "fit_mode": "fit_preprocessors",
            "inference_precision": "autocast", "memory_saving_mode": True,
            **model_params,
        }
        self._torch, self._classifier = _runtime()
        self._device = self._torch.device(self.params["device"])
        if self._device.type == "cuda":
            if not self._torch.cuda.is_available() or self._device.index >= self._torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {self._device} requested but unavailable; CPU fallback is disabled.")
        checkpoint = _checkpoint_path(self.params.get("model_path"))
        self.params["model_path"] = str(checkpoint)
        self._checkpoint = {
            "generation": MODEL_GENERATION, "repository": "Prior-Labs/tabpfn_3_5",
            "path": str(checkpoint), "sha256": _sha256(checkpoint),
        }
        self._model: Any = None
        self._peak_memory_measured = False

    def _estimator_params(self) -> dict[str, Any]:
        params = {
            **self.params,
            "random_state": self.seed,
            "categorical_features_indices": [
                index for index, name in enumerate(self.features)
                if name in self.categorical_features
            ],
            "ignore_pretraining_limits": False,
        }
        if params["inference_precision"] == "float32":
            params["inference_precision"] = self._torch.float32
        return params

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> None:
        if self._model is not None:
            raise RuntimeError("This adapter is already fitted; create a new adapter for a new context.")
        train = self._inputs(X_train)
        labels = binary_labels(y_train)
        if len(train) != len(labels) or len(np.unique(labels)) != 2:
            raise ValueError("Training X/y lengths must match and labels must contain both classes.")
        # Validation is intentionally not inspected, encoded, or passed to TabPFN.
        model = self._classifier(**self._estimator_params())
        model.fit(train, labels.copy())
        if set(model.classes_) != {0, 1}:
            raise ValueError("TabPFN fitted classes must be 0 and 1.")
        if model.n_estimators_ != self.params["n_estimators"]:
            raise RuntimeError("TabPFN changed the explicitly configured ensemble count.")
        self._model = model

    def _require_fitted(self) -> Any:
        if self._model is None:
            raise RuntimeError("TabPFN must be fitted before prediction, saving, or metadata access.")
        return self._model

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        model = self._require_fitted()
        raw = np.asarray(model.predict_proba(self._inputs(X)))
        classes = np.asarray(model.classes_)
        return self._positive_probability(raw, classes, len(X))

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    def reset_peak_memory(self) -> None:
        if self._device.type == "cuda":
            self.synchronize()
            self._torch.cuda.reset_peak_memory_stats(self._device)
            self._peak_memory_measured = True

    def resource_metadata(self) -> dict[str, Any]:
        if not self._peak_memory_measured:
            return {"peak_gpu_memory": None}
        return {"peak_gpu_memory": {
            "device": str(self._device),
            "allocated_bytes": int(self._torch.cuda.max_memory_allocated(self._device)),
            "reserved_bytes": int(self._torch.cuda.max_memory_reserved(self._device)),
            "scope": "Process PyTorch allocator peaks since adapter reset, not total GPU usage.",
        }}

    def save(self, output_path: Path) -> None:
        if output_path.suffix != ".tabpfn_fit":
            raise ValueError("TabPFN fitted state must use .tabpfn_fit.")
        self._require_fitted().save_fit_state(output_path)
        write_json(output_path.with_suffix(".adapter.json"), {
            "features": list(self.features),
            "categorical_features": list(self.categorical_features),
            "target": self.target, "seed": self.seed, "model_params": self.params,
            "checkpoint": self._checkpoint, "tabpfn_version": SUPPORTED_VERSION,
            "preprocessing": "Stateless categorical astype(str); internal fitted encoder is in .tabpfn_fit.",
        })

    @classmethod
    def load(cls, output_path: Path, *, device: str) -> TabPFNAdapter:
        """Restore trusted fitted state; original checkpoint path must still exist."""
        saved = json.loads(output_path.with_suffix(".adapter.json").read_text(encoding="utf-8"))
        if saved["tabpfn_version"] != SUPPORTED_VERSION:
            raise RuntimeError("Saved TabPFN version is incompatible.")
        adapter = cls(
            tuple(saved["features"]), tuple(saved["categorical_features"]), saved["target"],
            seed=saved["seed"], model_params={**saved["model_params"], "device": device},
        )
        if adapter._checkpoint != saved["checkpoint"]:
            raise ValueError("Checkpoint identity differs from the saved fitted-state dependency.")
        adapter._model = adapter._classifier.load_from_fit_state(output_path, device=device)
        return adapter

    def metadata(self) -> dict[str, Any]:
        model = self._require_fitted()
        precision = getattr(model, "forced_inference_dtype_", None)
        init_params = model.get_params(deep=False)
        init_params["inference_precision"] = str(init_params["inference_precision"])
        return {
            "adapter": type(self).__name__, "tabpfn_version": SUPPORTED_VERSION,
            "torch_version": version("torch"), "checkpoint": self._checkpoint,
            "validation_used": False,
            "validation_policy": "Unused; no early stopping, calibration, tuning, or validation context.",
            "preprocessing": {
                "external": "Categorical astype(str) on copies, no external fitted encoder, scaling or one-hot encoding.",
                "internal": "TabPFN 9.2.0 clean_data fits its ordinal encoder on training context only; reused across all queries.",
                "unseen_categories": "Internal OrdinalEncoder handle_unknown=use_encoded_value, unknown_value=-1.",
                "missing_nonfinite": "Canonical missing/nonfinite inputs rejected before TabPFN.",
                "inference_config": asdict(model.inference_config_),
            },
            "categorical_features": list(self.categorical_features),
            "categorical_indices_supplied": self._estimator_params()["categorical_features_indices"],
            "categorical_indices_resolved": list(model.categorical_features_indices_),
            "classes": model.classes_.tolist(),
            "requested_parameters": self.params,
            "effective_init_parameters": init_params,
            "effective_ensemble_count": int(model.n_estimators_),
            "compute_settings": {
                "device": str(self._device), "resolved_devices": [str(device) for device in model.devices_],
                "device_name": self._torch.cuda.get_device_name(self._device) if self._device.type == "cuda" else "CPU",
                "seed": self.seed, "n_estimators": int(model.n_estimators_),
                "fit_mode": self.params["fit_mode"],
                "inference_precision": self.params["inference_precision"],
                "forced_dtype": str(precision) if precision is not None else None,
                "autocast_enabled": bool(model.use_autocast_),
                "memory_saving_mode": self.params["memory_saving_mode"],
            },
            "ignore_pretraining_limits": False,
            "query_policy": (
                "Fixed training context and cached training-fitted preprocessing. "
                "v3.5 uses train-only scaling/ECDF and train-key attention; numerical "
                "query-batch equivalence is not assumed. Use the local smoke test to measure it."
            ),
            "timing_note": "Substantial inference work can occur during predict_proba; fit time is not comparable training cost.",
            "persistence": {
                "method": "TabPFN save_fit_state / load_from_fit_state plus adapter JSON sidecar.",
                "self_contained": False,
                "dependencies": "Same package version and original local checkpoint path/hash; weights are not embedded. Load only trusted artifacts.",
            },
        }
