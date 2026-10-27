from __future__ import annotations

import hashlib
import inspect
import io
import json
import logging
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.experiments.config import validate_model_params
from src.experiments.metrics import binary_labels
from src.experiments.results import write_json
from src.models.base import ModelAdapter
from src.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
SUPPORTED_VERSION = "2.2.0"
CHECKPOINT_FILENAME = "tabicl-classifier-v2-20260212.ckpt"
CHECKPOINT_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
CHECKPOINT_SIZE = 110368038
PROBABILITY_SUM_ATOL = 32 * float(np.finfo(np.float32).eps)
SAVE_FLAGS = {"save_model_weights": True, "save_training_data": True, "save_kv_cache": False}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime() -> tuple[Any, Any]:
    installed = version("tabicl")
    if installed != SUPPORTED_VERSION:
        raise RuntimeError(f"TabICL {SUPPORTED_VERSION} required; installed {installed}. No automatic upgrade.")
    import torch
    from tabicl import TabICLClassifier

    required = {
        "n_estimators", "norm_methods", "batch_size", "kv_cache", "model_path",
        "allow_auto_download", "checkpoint_version", "device", "use_amp", "use_fa3",
        "offload_mode", "random_state", "n_jobs", "inference_config",
    }
    if not required <= set(inspect.signature(TabICLClassifier).parameters):
        raise RuntimeError("Installed TabICL constructor is incompatible with the verified API.")
    for name, parameters in (
        ("fit", {"X", "y"}), ("predict_proba", {"X"}),
        ("save", {"path", *SAVE_FLAGS}), ("load", {"path", "device"}),
    ):
        if not hasattr(TabICLClassifier, name) or not parameters <= set(
            inspect.signature(getattr(TabICLClassifier, name)).parameters
        ):
            raise RuntimeError(f"Installed TabICL {name} API is incompatible.")
    return torch, TabICLClassifier


def _versions() -> dict[str, str]:
    return {
        **{name: version(name) for name in ("tabicl", "torch", "numpy", "pandas", "scikit-learn", "scipy")},
        "python": sys.version,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    raise TypeError(f"Unsupported TabICL metadata value: {type(value).__name__}")


class TabICLAdapter(ModelAdapter):
    """Explicit none-normalization experiment, not TabICL's default protocol."""

    artifact_name = "model.tabicl_fit"

    def __init__(
        self, features: tuple[str, ...], categorical_features: tuple[str, ...],
        target: str, *, seed: int, model_params: dict[str, Any],
    ) -> None:
        self._configure(features, categorical_features, target, seed, model_params)
        self._checkpoint = self._verify_checkpoint()

    def _configure(
        self, features: tuple[str, ...], categorical_features: tuple[str, ...],
        target: str, seed: int, model_params: dict[str, Any],
    ) -> None:
        super().__init__(features, categorical_features, target)
        validate_model_params(model_params, "tabicl")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        self.target, self.seed = target, seed
        self.params = {
            "n_estimators": 8, "batch_size": 1, "device": "cuda:0",
            "use_amp": False, "use_fa3": False, "kv_cache": False,
            "offload_mode": "cpu", "n_jobs": 4, **deepcopy(model_params),
        }
        path = Path(self.params["model_path"]).expanduser()
        self.params["model_path"] = str((PROJECT_ROOT / path).resolve())
        self._torch, self._classifier = _runtime()
        self._device = self._check_device(self.params["device"])
        self._model: Any = None
        self._diagnostic_path: Path | None = None
        self._diagnostic_records: list[dict[str, Any]] = []
        self._diagnostic_text: list[str] = []
        self._artifact_sha256: str | None = None
        self._peak_memory_measured = False

    def _check_device(self, device: str) -> Any:
        resolved = self._torch.device(device)
        if resolved.type == "cuda" and (
            not self._torch.cuda.is_available() or resolved.index >= self._torch.cuda.device_count()
        ):
            raise RuntimeError(f"CUDA device {device} unavailable; CPU fallback is disabled.")
        return resolved

    def _verify_checkpoint(self) -> dict[str, Any]:
        path = Path(self.params["model_path"])
        if path.name != CHECKPOINT_FILENAME:
            raise ValueError(f"TabICLv2 checkpoint must be named {CHECKPOINT_FILENAME}.")
        if not path.is_file():
            raise FileNotFoundError(f"Local TabICL checkpoint missing: {path}. This adapter never downloads weights.")
        if path.stat().st_size != CHECKPOINT_SIZE or _sha256(path) != CHECKPOINT_SHA256:
            raise ValueError("TabICL checkpoint size/SHA-256 does not match the fixed official revision.")
        return {
            "generation": "TabICLv2 classifier", "repository": "jingang/TabICL",
            "revision": CHECKPOINT_REVISION, "filename": CHECKPOINT_FILENAME,
            "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "path": str(path),
        }

    def _estimator_params(self) -> dict[str, Any]:
        return {
            **deepcopy(self.params), "random_state": self.seed, "allow_auto_download": False,
            "checkpoint_version": CHECKPOINT_FILENAME, "feat_shuffle_method": "latin",
            "class_shuffle_method": "shift", "outlier_threshold": 4.0,
            "softmax_temperature": 0.9, "average_logits": True, "support_many_classes": False,
            "disk_offload_dir": None, "verbose": True, "inference_config": None,
        }

    def set_diagnostic_path(self, path: Path) -> None:
        """Bind a fresh run log; capture remains in memory until a run binds it."""
        if self._diagnostic_path is not None:
            raise RuntimeError("TabICL diagnostic log is already bound.")
        with path.open("x", encoding="utf-8") as handle:
            handle.write("".join(self._diagnostic_text))
        self._diagnostic_path = path

    @contextmanager
    def _native_diagnostics(self, operation: str) -> Generator[None, None, None]:
        capture = io.StringIO()
        caught: list[Any] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with redirect_stdout(capture), redirect_stderr(capture):
                    yield
        finally:
            original = sys.exception()
            try:
                for warning in caught:
                    message = f"{warning.category.__name__}: {warning.message}"
                    capture.write(message + "\n")
                    LOGGER.warning("TabICL %s: %s", operation, message)
                text = capture.getvalue()
                lower = text.lower()
                recovery = "oom with batch_size=" in lower or "reducing to" in lower
                swap = "(swap)" in lower or "cpu_swap" in lower
                self._diagnostic_records.append({
                    "operation": operation, "warning_count": len(caught),
                    "recovery_detected": recovery, "swap_fallback_detected": swap,
                })
                entry = f"\n--- {operation} ---\n{text}"
                if self._diagnostic_path is None:
                    self._diagnostic_text.append(entry)
                else:
                    with self._diagnostic_path.open("a", encoding="utf-8") as handle:
                        handle.write(entry)
                if recovery or swap:
                    LOGGER.warning("TabICL native recovery/swap detected during %s; inspect native diagnostics.", operation)
            except (OSError, ValueError) as exc:
                if original is None:
                    raise
                LOGGER.exception("Could not persist TabICL diagnostics.")
                original.add_note(f"Could not persist native diagnostics: {exc}")

    def diagnostic_metadata(self) -> dict[str, Any]:
        return {
            "log_path": str(self._diagnostic_path) if self._diagnostic_path else None,
            "operations": deepcopy(self._diagnostic_records),
            "recovery_detection": (
                "recovery_detected" if any(row["recovery_detected"] for row in self._diagnostic_records)
                else "no_recovery_detected"
            ),
            "swap_fallback_detected": any(row["swap_fallback_detected"] for row in self._diagnostic_records),
            "limitations": (
                "Native stdout/stderr go to the diagnostic log, not the console; warnings remain visible. "
                "Captured output is not a complete scheduling trace. "
                "No recovery detected is not proof of unchanged internal scheduling. "
                "Native memory estimates and storage-tier choices may vary between calls."
            ),
        }

    def _validate_fitted(self, model: Any) -> None:
        if np.asarray(model.classes_).shape != (2,) or set(model.classes_) != {0, 1}:
            raise ValueError("TabICL fitted classes must be 0 and 1.")
        if model.n_features_in_ != len(self.features) or tuple(model.feature_names_in_) != self.features:
            raise ValueError("TabICL fitted feature schema differs from the adapter sidecar.")
        categorical_positions = [
            index for index, feature in enumerate(self.features) if feature in self.categorical_features
        ]
        fitted_positions = {
            name: list(positions) for name, _transformer, positions in model.X_encoder_.tfm_.transformers_
        }
        if fitted_positions["categorical"] != categorical_positions:
            raise ValueError("TabICL fitted categorical mapping differs from the adapter schema.")
        if model.model_.max_classes < 2:
            raise RuntimeError("TabICL checkpoint does not support native binary classification.")
        generator = model.ensemble_generator_
        count = sum(len(configs) for configs in generator.ensemble_configs_.values())
        if count != self.params["n_estimators"]:
            raise RuntimeError("TabICL changed the requested effective ensemble count.")
        if set(generator.preprocessors_) != {"none"} or model.model_kv_cache_ is not None:
            raise RuntimeError("TabICL fitted state differs from the explicit none/uncached profile.")
        if _json_value(model.get_params(deep=False)) != _json_value(self._estimator_params()):
            raise RuntimeError("TabICL effective constructor settings differ from the bounded profile.")

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> None:
        if self._model is not None:
            raise RuntimeError("TabICL is already fitted; create a new adapter for a new context.")
        train, labels = self._inputs(X_train), binary_labels(y_train)
        if len(train) != len(labels) or len(np.unique(labels)) != 2:
            raise ValueError("Training X/y lengths must match and labels must contain both classes.")
        # Recheck identity immediately before native fit loads the checkpoint.
        if self._verify_checkpoint() != self._checkpoint:
            raise ValueError("TabICL checkpoint identity changed before fit.")
        with self._native_diagnostics("fit"):
            model = self._classifier(**self._estimator_params())
            model.fit(train, labels.copy())
        self._validate_fitted(model)
        self._model = model

    def _require_fitted(self) -> Any:
        if self._model is None:
            raise RuntimeError("TabICL must be fitted before prediction, saving, or metadata access.")
        return self._model

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        model = self._require_fitted()
        inputs = self._inputs(X)
        threads = self._torch.get_num_threads()
        try:
            with self._native_diagnostics("predict_proba"):
                raw = np.asarray(model.predict_proba(inputs))
        finally:
            original = sys.exception()
            try:
                self._torch.set_num_threads(threads)
            except RuntimeError as exc:
                if original is None:
                    raise
                LOGGER.exception("Could not restore torch thread count; preserving native prediction error.")
                original.add_note(f"Could not restore torch thread count: {exc}")
        probability = self._positive_probability(raw, np.asarray(model.classes_), len(X))
        if not np.allclose(raw.sum(axis=1), 1, rtol=0, atol=PROBABILITY_SUM_ATOL):
            raise ValueError("TabICL probability row sums exceed the fixed FP32 engineering acceptance criterion.")
        return probability

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    def reset_peak_memory(self) -> None:
        if self._device.type == "cuda":
            self.synchronize()
            self._torch.cuda.reset_peak_memory_stats(self._device)
            self._peak_memory_measured = True

    def resource_metadata(self) -> dict[str, Any]:
        peak = None
        if self._peak_memory_measured:
            peak = {
                "device": str(self._device),
                "allocated_bytes": int(self._torch.cuda.max_memory_allocated(self._device)),
                "reserved_bytes": int(self._torch.cuda.max_memory_reserved(self._device)),
                "scope": "Process PyTorch allocator peaks since reset, not total device VRAM.",
            }
        return {"peak_gpu_memory": peak, "native_diagnostics": self.diagnostic_metadata()}

    def save(self, output_path: Path) -> None:
        model = self._require_fitted()
        if output_path.suffix != ".tabicl_fit":
            raise ValueError("TabICL fitted state must use .tabicl_fit.")
        with self._native_diagnostics("save"):
            model.save(output_path, **SAVE_FLAGS)
        self._artifact_sha256 = _sha256(output_path)
        write_json(output_path.with_suffix(".adapter.json"), {
            "format_version": 1, "features": list(self.features),
            "categorical_features": list(self.categorical_features), "target": self.target,
            "seed": self.seed, "model_params": self.params, "checkpoint": self._checkpoint,
            "profile": self.profile, "versions": _versions(), "save_flags": SAVE_FLAGS,
            "artifact_sha256": self._artifact_sha256, "saved_device": str(self._device),
        })

    @classmethod
    def load(
        cls, output_path: Path, *, device: str, diagnostic_path: Path | None = None,
    ) -> TabICLAdapter:
        """Restore only trusted native artifacts on their saved device."""
        if output_path.suffix != ".tabicl_fit":
            raise ValueError("TabICL fitted state must use .tabicl_fit.")
        saved = json.loads(output_path.with_suffix(".adapter.json").read_text(encoding="utf-8"))
        if saved["format_version"] != 1 or saved["versions"] != _versions() or saved["save_flags"] != SAVE_FLAGS:
            raise RuntimeError("TabICL saved format, dependency versions, or persistence flags are incompatible.")
        if saved["saved_device"] != device or saved["model_params"]["device"] != device:
            raise ValueError("TabICL native deserialization uses the saved device before override; only same-device restore is certified.")
        if _sha256(output_path) != saved["artifact_sha256"]:
            raise ValueError("TabICL artifact SHA-256 differs from its trusted sidecar.")
        checkpoint = saved["checkpoint"]
        if (
            checkpoint["sha256"] != CHECKPOINT_SHA256 or checkpoint["revision"] != CHECKPOINT_REVISION
            or checkpoint["filename"] != CHECKPOINT_FILENAME or checkpoint["size_bytes"] != CHECKPOINT_SIZE
            or checkpoint["repository"] != "jingang/TabICL"
        ):
            raise ValueError("Saved TabICL checkpoint provenance is incompatible.")
        adapter = cls.__new__(cls)
        adapter._configure(
            tuple(saved["features"]), tuple(saved["categorical_features"]),
            saved["target"], saved["seed"], saved["model_params"],
        )
        if adapter.profile != saved["profile"]:
            raise ValueError("Saved TabICL profile differs from its parameters.")
        adapter._checkpoint = checkpoint
        adapter._artifact_sha256 = saved["artifact_sha256"]
        if diagnostic_path is not None:
            adapter.set_diagnostic_path(diagnostic_path)
        with adapter._native_diagnostics("load"):
            model = adapter._classifier.load(output_path, device=device)
        adapter._validate_fitted(model)
        adapter._model = model
        return adapter

    @property
    def profile(self) -> str:
        precision = "amp" if self.params["use_amp"] else "fp32"
        return f"tabicl_v2_none_{precision}_uncached"

    def metadata(self) -> dict[str, Any]:
        model = self._require_fitted()
        generator = model.ensemble_generator_
        encoder = model.X_encoder_.tfm_
        mapping = [
            {"transformer": name, "input_positions": list(positions)}
            for name, _transformer, positions in encoder.transformers_
            if name != "remainder"
        ]
        config = {
            name: dict(getattr(model.inference_config_, name).items())
            for name in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
        }
        return _json_value({
            "adapter": type(self).__name__, "tabicl_version": SUPPORTED_VERSION,
            "experimental_profile": self.profile,
            "profile_scope": "Explicit none-normalization experiment; not a universal TabICL requirement or vendor-default benchmark.",
            "checkpoint": self._checkpoint, "artifact_sha256": self._artifact_sha256,
            "model_configuration": model.model_config_,
            "validation_used": False, "validation_policy": "Unused; train-only context/preprocessing, no tuning or calibration.",
            "feature_order": self.features, "categorical_features": self.categorical_features,
            "classes": model.classes_, "requested_parameters": self._estimator_params(),
            "effective_init_parameters": model.get_params(deep=False),
            "effective_ensemble_count": sum(len(items) for items in generator.ensemble_configs_.values()),
            "ensemble_configurations": generator.ensemble_configs_,
            "preprocessing": {
                "external": "Categorical astype(str) on copies; canonical inputs unchanged.",
                "internal": "Train-fitted ordinal encoder, mean imputer, constant-feature filter, standard scaling and outlier clipping.",
                "normalization_views": list(generator.preprocessors_),
                "normalization_policy": "Explicit experimental ['none']; native default is ['none', 'power'].",
                "transformed_column_mapping": mapping,
                "retained_feature_mask": generator.unique_filter_.features_to_keep_,
                "unseen_categories": "OrdinalEncoder unknown_value=-1; no query refit.",
                "missing_nonfinite": "Rejected by shared input contract; native NaN masking is outside this profile.",
                "calibration": "None.",
            },
            "compute_settings": {
                "device": str(self._device), "seed": self.seed,
                "device_name": self._torch.cuda.get_device_name(self._device) if self._device.type == "cuda" else "CPU",
                "n_estimators": self.params["n_estimators"], "ensemble_batch_size": self.params["batch_size"],
                "use_amp": self.params["use_amp"], "use_fa3": False, "kv_cache": False,
                "torch_threads_outside_prediction": self._torch.get_num_threads(),
                "n_jobs": self.params["n_jobs"], "inference_config": config,
                "cuda_matmul_allow_tf32": self._torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": self._torch.backends.cudnn.allow_tf32,
                "cuda_autocast_dtype": str(self._torch.get_autocast_dtype("cuda")),
            },
            "native_diagnostics": self.diagnostic_metadata(),
            "query_policy": (
                "Train-only summaries/attention. External chunks are query rows; native batch_size is ensemble members. "
                "Bounded local checks passed for the documented none/FP32 profile and both n2000 contexts; "
                "arbitrary inputs, devices, versions, and profiles are not certified. "
                "This is historical evidence, not certification of this run. See docs/results/tabicl_stage3_results.md."
            ),
            "persistence": {
                "method": "Official TabICL save/load, trusted pickle-based native format plus adapter sidecar.",
                "self_contained": True, "save_flags": SAVE_FLAGS,
                "dependencies": _versions(), "device_policy": "Same saved device only; no cross-device portability claim.",
            },
            "timing_note": "Fit loads weights and prepares context; uncached prediction repeats substantial context computation.",
        })
