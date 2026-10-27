from __future__ import annotations

import json
import math
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

from src.data import Domain, FeatureSet, VALID_DOMAINS, VALID_FEATURE_SETS
from src.models.ft_transformer_config import resolve_ft_transformer_params
from src.paths import PROJECT_ROOT, RESULTS_DIR


def positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_model_params(params: object, model: str = "catboost") -> None:
    """Use a deliberately small allowlist; aliases and fit controls are forbidden."""
    if not isinstance(params, dict):
        raise ValueError("model_params must be a JSON object.")
    if model == "tabpfn":
        validate_tabpfn_params(params)
        return
    if model == "tabicl":
        validate_tabicl_params(params)
        return
    if model == "ft_transformer":
        resolve_ft_transformer_params(params)
        return
    supported = {
        "iterations", "depth", "learning_rate", "loss_function", "eval_metric",
        "task_type", "thread_count", "verbose", "l2_leaf_reg",
    }
    unknown = set(params) - supported
    if unknown:
        raise ValueError(f"Unsupported or framework-controlled model_params: {sorted(unknown)}")
    for key, value in params.items():
        if key in {"iterations", "depth"}:
            positive_integer(value, key)
        elif key == "thread_count":
            if type(value) is not int or (value != -1 and value <= 0):
                raise ValueError("thread_count must be -1 or a positive integer.")
        elif key == "verbose":
            if type(value) is not bool and (type(value) is not int or value <= 0):
                raise ValueError("verbose must be a boolean or a positive integer.")
        elif key in {"learning_rate", "l2_leaf_reg"}:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
            if value < 0 or (key == "learning_rate" and value == 0):
                raise ValueError(f"{key} must be {'positive' if key == 'learning_rate' else 'nonnegative'}.")
        elif key in {"loss_function", "eval_metric"}:
            allowed = {"Logloss"} if key == "loss_function" else {"AUC", "Logloss"}
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{key} must be one of {sorted(allowed)}.")
        elif key == "task_type" and value not in ("CPU", "GPU"):
            raise ValueError("task_type must be CPU or GPU.")


def validate_tabicl_params(params: dict[str, Any]) -> None:
    supported = {
        "model_path", "device", "n_estimators", "batch_size", "norm_methods",
        "use_amp", "use_fa3", "kv_cache", "offload_mode", "n_jobs",
    }
    unknown = set(params) - supported
    if unknown:
        raise ValueError(f"Unsupported or framework-controlled TabICL model_params: {sorted(unknown)}")
    path = params.get("model_path")
    if not isinstance(path, str) or not path.strip() or path == "auto":
        raise ValueError("TabICL model_path must name an explicit local checkpoint.")
    if params.get("norm_methods") != ["none"]:
        raise ValueError(
            "This experimental TabICL adapter profile requires explicit norm_methods=['none']; "
            "this is not a general TabICL requirement."
        )
    for key in ("n_estimators", "batch_size", "n_jobs"):
        if key in params:
            positive_integer(params[key], key)
    if "device" in params:
        device = params["device"]
        if not isinstance(device, str) or not (
            device == "cpu" or (device.startswith("cuda:") and device[5:].isascii() and device[5:].isdigit())
        ):
            raise ValueError("TabICL device must be explicit: cpu or cuda:<index>.")
    if "use_amp" in params and type(params["use_amp"]) is not bool:
        raise ValueError("TabICL use_amp must be an explicit boolean, not 'auto'.")
    for key in ("use_fa3", "kv_cache"):
        if key in params and params[key] is not False:
            raise ValueError(f"This TabICL profile requires {key}=false.")
    if "offload_mode" in params and params["offload_mode"] != "cpu":
        raise ValueError("This TabICL profile requires offload_mode='cpu' (column outputs).")


def validate_tabpfn_params(params: dict[str, Any]) -> None:
    supported = {
        "n_estimators", "device", "fit_mode", "inference_precision",
        "memory_saving_mode", "model_path",
    }
    unknown = set(params) - supported
    if unknown:
        raise ValueError(f"Unsupported or framework-controlled TabPFN model_params: {sorted(unknown)}")
    if "n_estimators" in params:
        positive_integer(params["n_estimators"], "n_estimators")
    if "device" in params:
        device = params["device"]
        if not isinstance(device, str) or not (
            device == "cpu" or (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("TabPFN device must be explicit: cpu or cuda:<index>.")
    if "fit_mode" in params and params["fit_mode"] not in ("fit_preprocessors", "low_memory", "fit_with_cache"):
        raise ValueError("Unsupported TabPFN fit_mode; batched/fine-tuning mode is not supported.")
    if "inference_precision" in params and params["inference_precision"] not in ("auto", "autocast", "float32"):
        raise ValueError("TabPFN inference_precision must be auto, autocast, or float32.")
    if "memory_saving_mode" in params:
        value = params["memory_saving_mode"]
        if type(value) is not bool and value != "auto":
            raise ValueError("TabPFN memory_saving_mode must be a boolean or 'auto'.")
    if "model_path" in params:
        value = params["model_path"]
        if not isinstance(value, str) or not value.strip() or value == "auto":
            raise ValueError("model_path must name a local checkpoint, not 'auto'.")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model: str
    train_domain: Domain
    validation_domain: Domain
    test_domains: tuple[Domain, ...]
    feature_set: FeatureSet
    seed: int
    threshold: float
    calibration_bins: int
    model_params: dict[str, Any]
    output_dir: str
    early_stopping_rounds: int | None = None
    train_sample_size: int | None = None
    sampling_seed: int | None = None
    prediction_batch_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a nonempty string.")
        if self.model not in ("catboost", "tabpfn", "tabicl", "ft_transformer"):
            raise ValueError("model must be 'catboost', 'tabpfn', 'tabicl', or 'ft_transformer'.")
        for name, domain in (
            ("train_domain", self.train_domain),
            ("validation_domain", self.validation_domain),
        ):
            if not isinstance(domain, str) or domain not in VALID_DOMAINS:
                raise ValueError(f"{name} must be one of {VALID_DOMAINS}.")
        if self.validation_domain != self.train_domain:
            raise ValueError("validation_domain must equal train_domain.")
        if not isinstance(self.test_domains, tuple) or not self.test_domains:
            raise ValueError("test_domains must be a nonempty list of unique domains.")
        if any(not isinstance(domain, str) or domain not in VALID_DOMAINS for domain in self.test_domains):
            raise ValueError(f"test_domains must contain only {VALID_DOMAINS}.")
        if len(set(self.test_domains)) != len(self.test_domains):
            raise ValueError("test_domains must be unique.")
        if not isinstance(self.feature_set, str) or self.feature_set not in VALID_FEATURE_SETS:
            raise ValueError(f"feature_set must be explicitly one of {VALID_FEATURE_SETS}.")
        if "nhanes" in (self.train_domain, *self.test_domains) and self.feature_set == "full_24":
            raise ValueError("NHANES cannot use full_24; select an NHANES-compatible feature set.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold must be a finite number strictly between 0 and 1.")
        positive_integer(self.calibration_bins, "calibration_bins")
        if self.model == "catboost":
            positive_integer(self.early_stopping_rounds, "early_stopping_rounds")
        elif self.model == "ft_transformer" and self.early_stopping_rounds is not None:
            raise ValueError("ft_transformer uses patience/max_epochs/min_delta; early_stopping_rounds must be null.")
        elif self.early_stopping_rounds is not None:
            raise ValueError("Pretrained predictors do not use early_stopping_rounds; omit it or use null.")
        validate_model_params(self.model_params, self.model)
        if self.train_sample_size is not None:
            positive_integer(self.train_sample_size, "train_sample_size")
            if self.sampling_seed is None:
                raise ValueError("sampling_seed must be explicit when train_sample_size is set.")
        if self.sampling_seed is not None:
            if type(self.sampling_seed) is not int or not 0 <= self.sampling_seed < 2**32:
                raise ValueError("sampling_seed must be an integer in [0, 2**32).")
            if self.train_sample_size is None:
                raise ValueError("sampling_seed requires train_sample_size.")
        if self.prediction_batch_size is not None:
            positive_integer(self.prediction_batch_size, "prediction_batch_size")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path string.")
        self.resolved_output_dir

    @property
    def resolved_output_dir(self) -> Path:
        path = (PROJECT_ROOT / self.output_dir).resolve()
        results = RESULTS_DIR.resolve()
        if path == results or not path.is_relative_to(results):
            raise ValueError("output_dir must resolve to a run directory inside results/.")
        return path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["test_domains"] = list(self.test_domains)
        result["output_dir"] = str(self.resolved_output_dir)
        return result

    @classmethod
    def from_dict(cls, value: object) -> ExperimentConfig:
        if not isinstance(value, dict):
            raise ValueError("Experiment configuration must be a JSON object.")
        expected = {field.name for field in fields(cls)}
        unknown = set(value) - expected
        required = {field.name for field in fields(cls) if field.default is MISSING}
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(f"Configuration fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
        if not isinstance(value["test_domains"], list):
            raise ValueError("test_domains must be a JSON list.")
        values = dict(value)
        values["test_domains"] = tuple(values["test_domains"])
        return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return ExperimentConfig.from_dict(json.load(handle))
