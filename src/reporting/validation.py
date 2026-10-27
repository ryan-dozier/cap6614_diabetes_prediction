from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain_labels import display_domain_name
from src.experiments.metrics import evaluate_binary
from src.experiments.sampling import identity_fingerprints
from src.paths import PROJECT_ROOT, RESULTS_DIR

METRIC_TOLERANCE = 1e-8
IDENTITY_COLUMNS = {"source": "row_index", "kaggle": "row_index", "nhanes": "seqn"}
CATEGORICAL_FEATURES = ["ethnicity", "gender", "smoking_status"]
METRICS = ("roc_auc", "average_precision", "f1", "brier_score")
ARTIFACTS = (
    "config.json",
    "metadata.json",
    "metrics.json",
    "training_sample.csv.gz",
    "calibration_bins.csv",
)


class ReportValidationError(ValueError):
    """A saved artifact does not support a complete, comparable report."""


@dataclass
class RunArtifacts:
    spec: dict[str, Any]
    run_dir: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    predictions: dict[str, dict[int, dict[str, float | int]]]
    prediction_order: dict[str, list[int]]
    training_identities: list[int]
    artifact_hashes: dict[str, str]
    display_names: dict[str, str]


def _fail(run: str, artifact: str, field: str, message: str) -> None:
    raise ReportValidationError(f"run {run!r}, artifact {artifact!r}, field {field}: {message}")


def _json(path: Path, run: str, artifact: str) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _fail(run, artifact, "JSON", f"cannot read valid JSON ({exc})")


def _number(value: Any, run: str, artifact: str, field: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(run, artifact, field, "expected a finite numeric value")
    return float(value)


def _integer(value: Any, run: str, artifact: str, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(run, artifact, field, f"expected an integer >= {minimum}")
    return value


def _read_csv(path: Path, run: str, artifact: str) -> list[dict[str, str]]:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                _fail(run, artifact, "header", "missing or duplicate column names")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail(run, artifact, "CSV", f"cannot read valid CSV ({exc})")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        _fail(run, artifact, "rows", "inconsistent number of columns")
    return rows


def _csv_number(value: str, run: str, artifact: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(run, artifact, field, "expected a finite numeric value")
    if not math.isfinite(number):
        _fail(run, artifact, field, "expected a finite numeric value")
    return number


def _csv_int(value: str, run: str, artifact: str, field: str) -> int:
    if not value or value.strip() != value:
        _fail(run, artifact, field, "expected an integer identity/label")
    try:
        number = int(value)
    except ValueError:
        _fail(run, artifact, field, "expected an integer identity/label")
    if str(number) != value:
        _fail(run, artifact, field, "expected a canonical integer identity/label")
    return number


def _equal_metric(saved: Any, expected: float | None, run: str, field: str) -> None:
    artifact = "metrics.json"
    if expected is None:
        if saved is not None:
            _fail(run, artifact, field, "must be null because the prediction-derived value is undefined")
        return
    actual = _number(saved, run, artifact, field)
    if not math.isclose(actual, expected, rel_tol=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE):
        _fail(
            run, artifact, field,
            f"saved value {actual:.12g} disagrees with prediction-derived value {expected:.12g} "
            f"(tolerance {METRIC_TOLERANCE:g})",
        )


def _check_summary(
    actual: dict[str, Any], recorded: Any, run: str, artifact: str, field: str,
) -> None:
    if not isinstance(recorded, dict):
        _fail(run, artifact, field, "missing recorded split summary")
    _validate_recorded_summary(recorded, run, artifact, field)
    for key in ("domain", "split", "rows", "class_counts", "identity_column",
                "identity_sha256", "ordered_identity_sha256"):
        if recorded.get(key) != actual[key]:
            _fail(run, artifact, f"{field}.{key}", f"recorded value disagrees with artifact contents")


def _validate_recorded_summary(recorded: dict[str, Any], run: str, artifact: str, field: str) -> None:
    counts = recorded.get("class_counts")
    if not isinstance(counts, dict) or set(counts) != {"0", "1"}:
        _fail(run, artifact, f"{field}.class_counts", "must record both binary class counts")
    rows = _integer(recorded.get("rows"), run, artifact, f"{field}.rows", minimum=1)
    count_zero = _integer(counts["0"], run, artifact, f"{field}.class_counts.0")
    count_one = _integer(counts["1"], run, artifact, f"{field}.class_counts.1")
    if count_zero + count_one != rows:
        _fail(run, artifact, f"{field}.class_counts", "class counts do not sum to row count")
    for key in ("identity_sha256", "ordered_identity_sha256"):
        digest = recorded.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail(run, artifact, f"{field}.{key}", "expected a lowercase SHA-256 hex digest")


def _summary(domain: str, split: str, identities: list[int], labels: list[int]) -> dict[str, Any]:
    identity_column = IDENTITY_COLUMNS[domain]
    fingerprints = identity_fingerprints(domain, identity_column, identities)
    return {
        "domain": domain,
        "split": split,
        "rows": len(labels),
        "class_counts": {
            "0": labels.count(0),
            "1": labels.count(1),
        },
        "identity_column": identity_column,
        **fingerprints,
    }


def _check_manifest_v1(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ReportValidationError("manifest.json must contain a JSON object")
    if manifest.get("report_schema_version") != 1:
        raise ReportValidationError("manifest.json field report_schema_version must be 1")
    comparison = manifest.get("comparison")
    if not isinstance(comparison, dict):
        raise ReportValidationError("manifest.json field comparison must be an object")
    expected = {
        "models": ["catboost", "tabpfn"],
        "training_domains": ["source", "kaggle"],
        "evaluation_domains": ["source", "kaggle", "nhanes"],
        "feature_set": "strict_common_16",
        "training_budget": 2000,
        "model_seed": 42,
        "sampling_seed": 42,
        "threshold": 0.5,
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise ReportValidationError(f"manifest.json field comparison.{key} must be {value!r}")
    if comparison.get("target") != "diagnosed_diabetes":
        raise ReportValidationError("manifest.json field comparison.target must be 'diagnosed_diabetes'")
    feature_order = comparison.get("feature_order")
    if not isinstance(feature_order, list) or len(feature_order) != 16:
        raise ReportValidationError("manifest.json field comparison.feature_order must list 16 unique features")
    if not all(isinstance(feature, str) and feature for feature in feature_order) or len(set(feature_order)) != 16:
        raise ReportValidationError("manifest.json field comparison.feature_order must contain feature names")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise ReportValidationError("manifest.json field runs must contain exactly four explicit run entries")
    seen_roles: set[str] = set()
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    required_roles = {
        f"{domain}_{model}"
        for domain in expected["training_domains"]
        for model in expected["models"]
    }
    for index, spec in enumerate(runs):
        if not isinstance(spec, dict):
            raise ReportValidationError(f"manifest.json field runs[{index}] must be an object")
        for key in ("role", "run_id", "artifact_dir", "model", "train_domain",
                    "prediction_batch_size", "expected_device"):
            if key not in spec:
                raise ReportValidationError(f"manifest.json field runs[{index}].{key} is required")
        if not all(isinstance(spec[key], str) and spec[key] for key in ("role", "run_id", "artifact_dir")):
            raise ReportValidationError(f"manifest.json runs[{index}] identifiers and path must be nonempty strings")
        role = spec["role"]
        if role in seen_roles:
            raise ReportValidationError(f"manifest.json has duplicate comparison role {role!r}")
        if spec["run_id"] in seen_ids:
            raise ReportValidationError(f"manifest.json has duplicate run_id {spec['run_id']!r}")
        if spec["artifact_dir"] in seen_paths:
            raise ReportValidationError(f"manifest.json has duplicate artifact_dir {spec['artifact_dir']!r}")
        seen_roles.add(role)
        seen_ids.add(spec["run_id"])
        seen_paths.add(spec["artifact_dir"])
        if role not in required_roles or role != f"{spec['train_domain']}_{spec['model']}":
            raise ReportValidationError(
                f"manifest.json runs[{index}] has ambiguous or unexpected comparison role {role!r}"
            )
        if spec["model"] not in expected["models"] or spec["train_domain"] not in expected["training_domains"]:
            raise ReportValidationError(f"manifest.json runs[{index}] has unexpected model or train_domain")
        expected_batch = 256 if role == "source_tabpfn" else (
            1024 if role == "kaggle_tabpfn" else 256
        )
        expected_device = "CPU" if spec["model"] == "catboost" else "cuda"
        if spec["prediction_batch_size"] != expected_batch:
            raise ReportValidationError(
                f"manifest.json runs[{index}].prediction_batch_size must be {expected_batch}"
            )
        if spec["expected_device"] != expected_device:
            raise ReportValidationError(f"manifest.json runs[{index}].expected_device must be {expected_device!r}")
    if seen_roles != required_roles:
        raise ReportValidationError(f"manifest.json runs must cover roles {sorted(required_roles)}")
    differences = manifest.get("expected_methodological_differences")
    if not isinstance(differences, list) or not differences or not all(
        isinstance(item, str) and item.strip() for item in differences
    ):
        raise ReportValidationError(
            "manifest.json field expected_methodological_differences must contain explanatory strings"
        )
    return runs


def _check_manifest_v2(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if type(manifest.get("output_schema_version")) is not int or manifest["output_schema_version"] != 2:
        raise ReportValidationError("manifest.json field output_schema_version must be 2 for report schema 2")
    comparison = manifest.get("comparison")
    if not isinstance(comparison, dict):
        raise ReportValidationError("manifest.json field comparison must be an object")
    models = comparison.get("models")
    if models not in (["catboost", "tabpfn", "tabicl"],
                      ["catboost", "tabpfn", "tabicl", "ft_transformer"]):
        raise ReportValidationError("comparison.models must declare the supported three- or four-model group")
    expected_comparison = {
        "models": models,
        "training_domains": ["source", "kaggle"],
        "evaluation_domains": ["source", "kaggle", "nhanes"],
        "feature_set": "strict_common_16",
        "target": "diagnosed_diabetes",
        "training_budget": 2000,
        "model_seed": 42,
        "sampling_seed": 42,
        "threshold": 0.5,
    }
    for key, expected in expected_comparison.items():
        actual = comparison.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ReportValidationError(f"manifest.json field comparison.{key} must be {expected!r}")
    feature_order = comparison.get("feature_order")
    if (
        not isinstance(feature_order, list)
        or len(feature_order) != 16
        or not all(isinstance(feature, str) and feature for feature in feature_order)
        or len(set(feature_order)) != 16
    ):
        raise ReportValidationError(
            "manifest.json field comparison.feature_order must contain 16 unique feature names"
        )
    if comparison.get("categorical_features") != CATEGORICAL_FEATURES:
        raise ReportValidationError(
            f"manifest.json field comparison.categorical_features must be {CATEGORICAL_FEATURES!r}"
        )

    profiles = manifest.get("model_profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(expected_comparison["models"]):
        raise ReportValidationError("manifest.json field model_profiles must define all declared models")
    expected_profiles = {
        "catboost": {
            "generation": "CatBoost CPU",
            "validation_policy": "same_domain_early_stopping",
            "device": "CPU",
            "prediction_batch_size": 256,
        },
        "tabpfn": {
            "generation": "TabPFN v3.5 (20260909)",
            "validation_policy": "training_context_only",
            "device": "cuda",
            "inference_precision": "float32",
            "n_estimators": 4,
        },
        "tabicl": {
            "generation": "TabICLv2 classifier",
            "checkpoint_revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
            "checkpoint_filename": "tabicl-classifier-v2-20260212.ckpt",
            "package_version": "2.2.0",
            "profile_id": "tabicl_v2_none_fp32_uncached",
            "validation_policy": "training_context_only",
            "device": "cuda",
            "precision": "FP32",
            "normalization": ["none"],
            "n_estimators": 8,
            "native_ensemble_batch_size": 1,
            "column_output_storage": "cpu",
            "kv_cache": False,
            "use_amp": False,
            "use_fa3": False,
        },
    }
    if "ft_transformer" in models:
        expected_profiles["ft_transformer"] = {
            "generation": "FT-Transformer (task-trained)",
            "package_version": "0.0.2",
            "profile_id": "ft_transformer_n2000_float32",
            "validation_policy": "same_domain_early_stopping",
            "device": "cuda:0",
            "precision": "float32",
        }
    for model, required in expected_profiles.items():
        actual = profiles[model]
        if not isinstance(actual, dict):
            raise ReportValidationError(f"manifest.json field model_profiles.{model} must be an object")
        for key, expected in required.items():
            value = actual.get(key)
            if type(value) is not type(expected) or value != expected:
                raise ReportValidationError(
                    f"manifest.json field model_profiles.{model}.{key} must be {expected!r}"
                )

    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 2 * len(models):
        raise ReportValidationError("manifest.json field runs must contain exactly two explicit runs per declared model")
    required_roles = {
        f"{domain}_{model}"
        for domain in expected_comparison["training_domains"]
        for model in expected_comparison["models"]
    }
    seen_roles: set[str] = set()
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    expected_batches = {
        ("source", "catboost"): 256,
        ("source", "tabpfn"): 256,
        ("source", "tabicl"): 128,
        ("source", "ft_transformer"): 256,
        ("kaggle", "catboost"): 256,
        ("kaggle", "tabpfn"): 1024,
        ("kaggle", "tabicl"): 128,
        ("kaggle", "ft_transformer"): 256,
    }
    for index, spec in enumerate(runs):
        if not isinstance(spec, dict):
            raise ReportValidationError(f"manifest.json field runs[{index}] must be an object")
        for key in ("role", "run_id", "artifact_dir", "model", "train_domain",
                    "prediction_batch_size", "expected_device"):
            if key not in spec:
                raise ReportValidationError(f"manifest.json field runs[{index}].{key} is required")
        if not all(
            isinstance(spec[key], str) and spec[key]
            for key in ("role", "run_id", "artifact_dir", "model", "train_domain", "expected_device")
        ):
            raise ReportValidationError(f"manifest.json runs[{index}] identifiers and path must be nonempty strings")
        role = spec["role"]
        if role in seen_roles:
            raise ReportValidationError(f"manifest.json has duplicate comparison role {role!r}")
        if spec["run_id"] in seen_ids:
            raise ReportValidationError(f"manifest.json has duplicate run_id {spec['run_id']!r}")
        if spec["artifact_dir"] in seen_paths:
            raise ReportValidationError(f"manifest.json has duplicate artifact_dir {spec['artifact_dir']!r}")
        seen_roles.add(role)
        seen_ids.add(spec["run_id"])
        seen_paths.add(spec["artifact_dir"])
        if (
            role not in required_roles
            or role != f"{spec['train_domain']}_{spec['model']}"
            or spec["model"] not in expected_comparison["models"]
            or spec["train_domain"] not in expected_comparison["training_domains"]
        ):
            raise ReportValidationError(f"manifest.json runs[{index}] has unexpected model, domain, or role")
        expected_device = profiles[spec["model"]]["device"]
        if spec["expected_device"] != expected_device:
            raise ReportValidationError(
                f"manifest.json runs[{index}].expected_device must be {expected_device!r}"
            )
        batch = spec["prediction_batch_size"]
        if type(batch) is not int or batch != expected_batches[(spec["train_domain"], spec["model"])]:
            raise ReportValidationError(
                f"manifest.json runs[{index}].prediction_batch_size must be "
                f"{expected_batches[(spec['train_domain'], spec['model'])]}"
            )
    if seen_roles != required_roles:
        raise ReportValidationError(f"manifest.json runs must cover roles {sorted(required_roles)}")

    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 2 * (len(models) - 1):
        raise ReportValidationError("manifest.json field comparisons must contain " + ("four" if len(models) == 3 else "six") + " explicit comparisons")
    spec_by_role = {spec["role"]: spec for spec in runs}
    seen_comparison_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    expected_pairs = {
        (domain, f"{domain}_catboost", f"{domain}_{candidate}")
        for domain in expected_comparison["training_domains"]
        for candidate in models[1:]
    }
    actual_pairs: set[tuple[str, str, str]] = set()
    for index, item in enumerate(comparisons):
        if not isinstance(item, dict):
            raise ReportValidationError(f"manifest.json field comparisons[{index}] must be an object")
        if not all(
            isinstance(item.get(key), str) and item[key]
            for key in ("id", "training_domain", "baseline_role", "candidate_role")
        ):
            raise ReportValidationError(f"manifest.json comparisons[{index}] requires string identifiers")
        if item["id"] in seen_comparison_ids:
            raise ReportValidationError(f"manifest.json has duplicate comparison id {item['id']!r}")
        seen_comparison_ids.add(item["id"])
        baseline_role = item["baseline_role"]
        candidate_role = item["candidate_role"]
        if baseline_role == candidate_role:
            raise ReportValidationError(f"manifest.json comparisons[{index}] cannot compare a role with itself")
        if baseline_role not in spec_by_role or candidate_role not in spec_by_role:
            raise ReportValidationError(f"manifest.json comparisons[{index}] references an unknown run role")
        baseline = spec_by_role[baseline_role]
        candidate = spec_by_role[candidate_role]
        if (
            baseline["train_domain"] != item["training_domain"]
            or candidate["train_domain"] != item["training_domain"]
        ):
            raise ReportValidationError(
                f"manifest.json comparisons[{index}] contains a cross-training-domain pair"
            )
        if baseline["model"] != "catboost" or candidate["model"] not in models[1:]:
            raise ReportValidationError(
                f"manifest.json comparisons[{index}] must use CatBoost baseline and a declared candidate"
            )
        pair = (baseline_role, candidate_role)
        if pair in seen_pairs:
            raise ReportValidationError(f"manifest.json has duplicate comparison {baseline_role!r} vs {candidate_role!r}")
        seen_pairs.add(pair)
        actual_pairs.add((item["training_domain"], baseline_role, candidate_role))
    if actual_pairs != expected_pairs:
        raise ReportValidationError("manifest.json comparisons must declare all candidates against CatBoost per domain")

    differences = manifest.get("expected_methodological_differences")
    if not isinstance(differences, list) or not differences or not all(
        isinstance(item, str) and item.strip() for item in differences
    ):
        raise ReportValidationError(
            "manifest.json field expected_methodological_differences must contain explanatory strings"
        )
    return runs


def _check_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ReportValidationError("manifest.json must contain a JSON object")
    schema_version = manifest.get("report_schema_version")
    if type(schema_version) is not int:
        raise ReportValidationError("manifest.json field report_schema_version must be a supported integer")
    if schema_version == 1:
        return _check_manifest_v1(manifest)
    if schema_version == 2:
        return _check_manifest_v2(manifest)
    raise ReportValidationError(f"unsupported report_schema_version {schema_version!r}")


def _input_run_dir(spec: dict[str, Any], runs_root: Path) -> Path:
    configured = Path(spec["artifact_dir"])
    candidate = configured if configured.is_absolute() else PROJECT_ROOT / configured
    resolved = candidate.resolve()
    if not resolved.is_relative_to(runs_root) or resolved == runs_root:
        raise ReportValidationError(
            f"run {spec['run_id']!r}, artifact_dir resolves outside {runs_root}: {resolved}"
        )
    if not resolved.is_dir():
        raise ReportValidationError(f"run {spec['run_id']!r}, artifact_dir does not exist: {resolved}")
    return resolved


def _validate_v2_model_profile(
    spec: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    model_name = spec["model"]
    model = metadata.get("model")
    if not isinstance(model, dict):
        _fail(spec["run_id"], "metadata.json", "model", "missing recorded model profile")
    params = config.get("model_params")
    compute = metadata.get("compute_settings")
    if model_name == "catboost":
        if not isinstance(params, dict) or not isinstance(compute, dict):
            _fail(spec["run_id"], "metadata.json/config.json", "model profile", "required CatBoost profile is missing")
        if model.get("validation_used") is not True:
            _fail(spec["run_id"], "metadata.json", "model.validation_used", "CatBoost must use same-domain validation")
        if (
            spec["expected_device"] != "CPU"
            or params.get("task_type") != "CPU"
            or compute.get("task_type") != "CPU"
        ):
            _fail(spec["run_id"], "config.json", "model_params.task_type", "expected CatBoost CPU execution")
        return

    if model_name == "ft_transformer":
        _validate_ft_profile(spec, config, metadata)
        return

    effective = model.get("effective_init_parameters")
    requested = model.get("requested_parameters")
    if (
        not isinstance(params, dict)
        or not isinstance(effective, dict)
        or not isinstance(requested, dict)
        or not isinstance(compute, dict)
    ):
        _fail(spec["run_id"], "metadata.json/config.json", "model profile", "required effective profile is missing")

    if model.get("validation_used") is not False:
        _fail(spec["run_id"], "metadata.json", "model.validation_used", "foundation models must not use validation")
    policy = model.get("validation_policy")
    if not isinstance(policy, str) or "Unused" not in policy:
        _fail(spec["run_id"], "metadata.json", "model.validation_policy", "expected an explicit unused-validation policy")
    if not str(compute.get("device", "")).startswith("cuda:"):
        _fail(spec["run_id"], "metadata.json", "compute_settings.device", "expected recorded CUDA execution")

    profile = manifest["model_profiles"][model_name]
    if model_name == "tabpfn":
        checkpoint = model.get("checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("generation") != profile["generation"]:
            _fail(spec["run_id"], "metadata.json", "model.checkpoint.generation", "does not match declared TabPFN generation")
        if params.get("inference_precision") != profile["inference_precision"]:
            _fail(spec["run_id"], "config.json", "model_params.inference_precision", "does not match declared FP32 profile")
        if params.get("n_estimators") != profile["n_estimators"]:
            _fail(spec["run_id"], "config.json", "model_params.n_estimators", "does not match declared ensemble")
        if (
            model.get("effective_ensemble_count") != profile["n_estimators"]
            or requested.get("n_estimators") != profile["n_estimators"]
            or effective.get("inference_precision") != "torch.float32"
            or requested.get("inference_precision") != profile["inference_precision"]
            or compute.get("forced_dtype") != "torch.float32"
            or compute.get("autocast_enabled") is not False
            or compute.get("n_estimators") != profile["n_estimators"]
        ):
            _fail(spec["run_id"], "metadata.json", "model effective precision/ensemble", "does not match declared TabPFN profile")
        return

    checkpoint = model.get("checkpoint")
    if not isinstance(checkpoint, dict):
        _fail(spec["run_id"], "metadata.json", "model.checkpoint", "missing recorded TabICL checkpoint metadata")
    try:
        configured_checkpoint = (PROJECT_ROOT / params["model_path"]).resolve()
        recorded_checkpoint = Path(checkpoint.get("path", "")).resolve()
    except (KeyError, OSError, TypeError, ValueError):
        _fail(spec["run_id"], "config.json/metadata.json", "TabICL checkpoint path", "is invalid")
    if configured_checkpoint != recorded_checkpoint:
        _fail(spec["run_id"], "metadata.json", "model.checkpoint.path", "disagrees with config.json")
    if (
        checkpoint.get("generation") != profile["generation"]
        or checkpoint.get("revision") != profile["checkpoint_revision"]
        or checkpoint.get("filename") != profile["checkpoint_filename"]
        or model.get("tabicl_version") != profile["package_version"]
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint["sha256"])
    ):
        _fail(spec["run_id"], "metadata.json", "model.checkpoint", "does not match declared TabICLv2 checkpoint")
    expected_config = {
        "norm_methods": profile["normalization"],
        "n_estimators": profile["n_estimators"],
        "batch_size": profile["native_ensemble_batch_size"],
        "kv_cache": profile["kv_cache"],
        "use_amp": profile["use_amp"],
        "use_fa3": profile["use_fa3"],
        "offload_mode": profile["column_output_storage"],
    }
    for key, expected in expected_config.items():
        if params.get(key) != expected or requested.get(key) != expected or effective.get(key) != expected:
            _fail(
                spec["run_id"], "config.json/metadata.json",
                f"TabICL {key}", f"does not match declared experimental profile {expected!r}",
            )
    preprocessing = model.get("preprocessing")
    diagnostics = model.get("native_diagnostics")
    if not isinstance(preprocessing, dict) or not isinstance(diagnostics, dict):
        _fail(spec["run_id"], "metadata.json", "model preprocessing/native_diagnostics", "required TabICL evidence is missing")
    recovery_detection = diagnostics.get("recovery_detection")
    limitations = diagnostics.get("limitations")
    if (
        not isinstance(recovery_detection, str)
        or not recovery_detection
        or type(diagnostics.get("swap_fallback_detected")) is not bool
        or not isinstance(diagnostics.get("operations"), list)
        or not isinstance(limitations, str)
    ):
        _fail(spec["run_id"], "metadata.json", "model.native_diagnostics", "invalid diagnostic evidence")
    if (
        model.get("experimental_profile") != profile["profile_id"]
        or model.get("effective_ensemble_count") != profile["n_estimators"]
        or compute.get("n_estimators") != profile["n_estimators"]
        or compute.get("ensemble_batch_size") != profile["native_ensemble_batch_size"]
        or compute.get("use_amp") is not False
        or compute.get("use_fa3") is not False
        or compute.get("kv_cache") is not False
        or compute.get("cuda_matmul_allow_tf32") is not False
        or preprocessing.get("normalization_views") != profile["normalization"]
    ):
        _fail(spec["run_id"], "metadata.json", "model/compute profile", "does not match declared TabICL profile")
    inference_config = compute.get("inference_config")
    column_config = inference_config.get("COL_CONFIG") if isinstance(inference_config, dict) else None
    if (
        not isinstance(column_config, dict)
        or column_config.get("offload") != profile["column_output_storage"]
        or column_config.get("use_amp") is not profile["use_amp"]
        or column_config.get("use_fa3") is not profile["use_fa3"]
    ):
        _fail(spec["run_id"], "metadata.json", "compute_settings.inference_config.COL_CONFIG", "does not match declared TabICL profile")
    profile_name = model.get("experimental_profile")
    if profile_name != "tabicl_v2_none_fp32_uncached":
        _fail(spec["run_id"], "metadata.json", "model.experimental_profile", "expected explicit none/FP32/uncached profile")



def _validate_ft_profile(spec: dict, config: dict, metadata: dict) -> None:
    # Pure config import: reporting never imports Torch or loads model weights.
    from src.models.ft_transformer_config import FT_DEFAULT_PARAMS
    model = metadata["model"]
    compute = metadata["compute_settings"]
    def require(condition: bool, field: str) -> None:
        if not condition:
            _fail(spec["run_id"], "metadata.json/config.json", field,
                  "does not match the frozen FT-Transformer profile")
    require(config.get("model_params") == FT_DEFAULT_PARAMS, "model_params")
    require(model.get("model_params") == FT_DEFAULT_PARAMS, "model.model_params")
    require(config.get("validation_domain") == config.get("train_domain"), "validation_domain")
    require(model.get("validation_used") is True and model.get("model_type") == "task-trained",
            "model.validation_used/model_type")
    require(model.get("ready") is True and model.get("adapter") == "FTTransformerAdapter", "model.adapter")
    require(model.get("precision") == "float32" and compute.get("precision") == "float32",
            "precision")
    require(model.get("device") == compute.get("device") == "cuda:0", "device")
    require(model.get("versions", {}).get("rtdl-revisiting-models") == "0.0.2", "package_version")
    require(compute.get("architecture") == {k: FT_DEFAULT_PARAMS[k] for k in
            ("n_blocks", "d_block", "attention_n_heads", "attention_dropout",
             "ffn_d_hidden", "ffn_d_hidden_multiplier", "ffn_activation", "ffn_dropout",
             "residual_dropout")}, "compute_settings.architecture")
    det = compute.get("determinism", {})
    require(det.get("cuda_matmul_allow_tf32") is False and det.get("cudnn_allow_tf32") is False
            and det.get("cudnn_benchmark") is False and det.get("cudnn_deterministic") is True,
            "compute_settings.determinism")
    training = model.get("training", {})
    history = training.get("training_history")
    require(isinstance(history, list) and 0 < len(history) <= 200, "training_history")
    best, reference, counter = None, None, 0
    for epoch, row in enumerate(history, 1):
        require(isinstance(row, dict) and row.get("epoch") == epoch, "training_history.epoch")
        auc = _number(row.get("validation_roc_auc"), spec["run_id"], "metadata.json", "validation_roc_auc")
        require(auc is not None and 0 <= auc <= 1, "validation_roc_auc")
        for field in ("training_loss", "validation_loss", "training_duration_seconds", "validation_duration_seconds"):
            value = _number(row.get(field), spec["run_id"], "metadata.json", field)
            require(value is not None and value >= 0, field)
        if best is None or auc > best[1]:
            best = (epoch, auc)
        if reference is None or auc - reference >= 0.001:
            reference, counter = auc, 0
        else:
            counter += 1
        require(counter < 20 or epoch == len(history), "first patience exhaustion")
    require(training.get("best_epoch") == best[0] and
            training.get("best_validation_roc_auc") == best[1], "selected checkpoint")
    require(training.get("epochs_executed") == len(history) and
            training.get("patience_counter") == counter and
            training.get("patience_reference") == reference, "training stop counters")
    require((counter == 20 and training.get("stopping_reason") == "patience") or
            (counter < 20 and len(history) == 200 and training.get("stopping_reason") == "max_epochs"),
            "stopping_reason")
    require(training.get("training_population_size") == metadata["training_context"]["rows"],
            "training_population_size")
    val = metadata["splits"][config["train_domain"] + "/val"]
    require(training.get("validation_population_size") == val["rows"] and
            training.get("validation_class_counts") == val["class_counts"], "validation population")


def validate_run(spec: dict[str, Any], run_dir: Path, manifest: dict[str, Any]) -> RunArtifacts:
    run = spec["run_id"]
    domains = manifest["comparison"]["evaluation_domains"]
    required = [*ARTIFACTS, *(f"predictions_{domain}_test.csv.gz" for domain in domains)]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        _fail(run, missing[0], "file", "required artifact is missing")

    hashes: dict[str, str] = {}
    for name in required:
        artifact_path = run_dir / name
        resolved_artifact = artifact_path.resolve()
        if not resolved_artifact.is_relative_to(run_dir.resolve()):
            _fail(run, name, "path", "artifact resolves outside its selected run directory")
        try:
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            relative_path = resolved_artifact.relative_to(PROJECT_ROOT.resolve())
        except (OSError, ValueError) as exc:
            _fail(run, name, "file", f"cannot read artifact safely ({exc})")
        hashes[str(relative_path)] = digest

    config = _json(run_dir / "config.json", run, "config.json")
    metadata = _json(run_dir / "metadata.json", run, "metadata.json")
    saved_metrics = _json(run_dir / "metrics.json", run, "metrics.json")
    if not isinstance(config, dict) or not isinstance(metadata, dict) or not isinstance(saved_metrics, dict):
        _fail(run, "config.json/metadata.json/metrics.json", "root", "expected JSON objects")
    if metadata.get("status") != "completed":
        _fail(run, "metadata.json", "status", f"expected 'completed', got {metadata.get('status')!r}")
    if metadata.get("configuration") != config:
        _fail(run, "metadata.json", "configuration", "does not exactly match config.json")
    if metadata.get("seed") != config.get("seed"):
        _fail(run, "metadata.json", "seed", "disagrees with config.json")
    if metadata.get("prediction_batch_size") != config.get("prediction_batch_size"):
        _fail(run, "metadata.json", "prediction_batch_size", "disagrees with config.json")
    try:
        stored_output = Path(config.get("output_dir", "")).resolve()
        if stored_output != run_dir.resolve():
            _fail(run, "config.json", "output_dir", "does not resolve to this artifact directory")
    except (OSError, TypeError, ValueError):
        _fail(run, "config.json", "output_dir", "invalid output directory")

    comparison = manifest["comparison"]
    expected_config = {
        "experiment_name": run,
        "model": spec["model"],
        "train_domain": spec["train_domain"],
        "validation_domain": spec["train_domain"],
        "test_domains": domains,
        "feature_set": comparison["feature_set"],
        "seed": comparison["model_seed"],
        "threshold": comparison["threshold"],
        "train_sample_size": comparison["training_budget"],
        "sampling_seed": comparison["sampling_seed"],
        "prediction_batch_size": spec["prediction_batch_size"],
    }
    for key, expected_value in expected_config.items():
        if config.get(key) != expected_value:
            _fail(run, "config.json", key, f"expected {expected_value!r}, got {config.get(key)!r}")
    threshold = _number(config.get("threshold"), run, "config.json", "threshold")
    if threshold is None or not 0 < threshold < 1:
        _fail(run, "config.json", "threshold", "must be strictly between zero and one")
    _integer(config.get("seed"), run, "config.json", "seed")
    _integer(config.get("train_sample_size"), run, "config.json", "train_sample_size", minimum=2)
    _integer(config.get("sampling_seed"), run, "config.json", "sampling_seed")
    _integer(config.get("prediction_batch_size"), run, "config.json", "prediction_batch_size", minimum=1)
    if config.get("model") == "catboost":
        if config.get("early_stopping_rounds") is None or config.get("validation_domain") != config.get("train_domain"):
            _fail(run, "config.json", "early_stopping_rounds", "CatBoost requires recorded same-domain validation")
        _integer(config["early_stopping_rounds"], run, "config.json", "early_stopping_rounds", minimum=1)
        params = config.get("model_params")
        if not isinstance(params, dict) or params.get("task_type") != "CPU":
            _fail(run, "config.json", "model_params.task_type", "expected CatBoost CPU execution")
        if spec["expected_device"] != "CPU":
            _fail(run, "manifest.json", "expected_device", "CatBoost run must declare CPU")
    elif config.get("model") == "tabpfn":
        if config.get("early_stopping_rounds") is not None:
            _fail(run, "config.json", "early_stopping_rounds", "TabPFN must not use early stopping")
        params = config.get("model_params")
        if not isinstance(params, dict) or not str(params.get("device", "")).startswith("cuda:"):
            _fail(run, "config.json", "model_params.device", "expected recorded CUDA execution")
    else:
        if config.get("early_stopping_rounds") is not None:
            _fail(run, "config.json", "early_stopping_rounds", "generic early_stopping_rounds must be null for this model")
        params = config.get("model_params")
        if not isinstance(params, dict) or not str(params.get("device", "")).startswith("cuda:"):
            _fail(run, "config.json", "model_params.device", "expected recorded CUDA execution")

    schema = metadata.get("schema")
    if not isinstance(schema, dict):
        _fail(run, "metadata.json", "schema", "missing feature/target schema")
    if schema.get("target") != comparison["target"]:
        _fail(run, "metadata.json", "schema.target", f"expected {comparison['target']!r}")
    if schema.get("feature_order") != comparison["feature_order"]:
        _fail(run, "metadata.json", "schema.feature_order", "does not match the manifest feature order")
    categorical_features = comparison.get("categorical_features", CATEGORICAL_FEATURES)
    numeric_features = [
        feature for feature in comparison["feature_order"]
        if feature not in categorical_features
    ]
    if (
        schema.get("categorical_features") != categorical_features
        or schema.get("numeric_features") != numeric_features
    ):
        _fail(run, "metadata.json", "schema categorical/numeric features", "does not match the fixed feature definitions")
    model_metadata = metadata.get("model")
    if isinstance(model_metadata, dict):
        recorded_categorical = model_metadata.get("categorical_features")
        if (
            (manifest["report_schema_version"] == 2 and recorded_categorical != categorical_features)
            or (
                "categorical_features" in model_metadata
                and recorded_categorical != categorical_features
            )
        ):
            _fail(run, "metadata.json", "model.categorical_features", "does not match the fixed categorical features")

    sampling = metadata.get("sampling")
    training = metadata.get("training_context")
    if not isinstance(sampling, dict) or not isinstance(training, dict):
        _fail(run, "metadata.json", "sampling/training_context", "required provenance metadata is missing")
    full_training = metadata.get("training_partition_full")
    split_metadata = metadata.get("splits")
    if not isinstance(full_training, dict) or not isinstance(split_metadata, dict):
        _fail(run, "metadata.json", "training_partition_full/splits", "required split provenance is missing")
    required_split_keys = {
        f"{spec['train_domain']}/train",
        f"{spec['train_domain']}/val",
        *(f"{domain}/test" for domain in domains),
    }
    if set(split_metadata) != required_split_keys:
        _fail(run, "metadata.json", "splits", f"expected exactly {sorted(required_split_keys)}")
    for split_name, recorded in (
        (f"{spec['train_domain']}/train", full_training),
        (f"{spec['train_domain']}/train", split_metadata.get(f"{spec['train_domain']}/train")),
        (f"{spec['train_domain']}/val", split_metadata.get(f"{spec['train_domain']}/val")),
    ):
        if not isinstance(recorded, dict) or recorded.get("domain") != split_name.split("/")[0] or recorded.get("split") != split_name.split("/")[1]:
            _fail(run, "metadata.json", f"splits.{split_name}", "invalid or inconsistent split summary")
        _validate_recorded_summary(recorded, run, "metadata.json", f"splits.{split_name}")
        if recorded.get("identity_column") != IDENTITY_COLUMNS[spec["train_domain"]]:
            _fail(run, "metadata.json", f"splits.{split_name}.identity_column", "unexpected authoritative identity column")
    if full_training != split_metadata.get(f"{spec['train_domain']}/train"):
        _fail(run, "metadata.json", "training_partition_full", "disagrees with splits train summary")
    sample_path = run_dir / "training_sample.csv.gz"
    sample_rows = _read_csv(sample_path, run, "training_sample.csv.gz")
    identity_column = IDENTITY_COLUMNS[spec["train_domain"]]
    if not sample_rows:
        _fail(run, "training_sample.csv.gz", "rows", "training sample is empty")
    required_sample_columns = {"domain", "split", "sample_order", identity_column}
    if not required_sample_columns.issubset(sample_rows[0]):
        _fail(run, "training_sample.csv.gz", "header", f"requires columns {sorted(required_sample_columns)}")
    sample_identities: list[int] = []
    for position, row in enumerate(sample_rows):
        if row["domain"] != spec["train_domain"] or row["split"] != "train":
            _fail(run, "training_sample.csv.gz", f"rows[{position}]", "domain/split disagree with the run role")
        if _csv_int(row["sample_order"], run, "training_sample.csv.gz", f"rows[{position}].sample_order") != position:
            _fail(run, "training_sample.csv.gz", f"rows[{position}].sample_order", "must be contiguous and ordered from zero")
        sample_identities.append(_csv_int(
            row[identity_column], run, "training_sample.csv.gz", f"rows[{position}].{identity_column}",
        ))
    if len(set(sample_identities)) != len(sample_identities):
        _fail(run, "training_sample.csv.gz", identity_column, "duplicate training identities")
    if len(sample_identities) != comparison["training_budget"]:
        _fail(run, "training_sample.csv.gz", "rows", f"expected {comparison['training_budget']} training identities")
    train_summary = {
        "domain": spec["train_domain"],
        "split": "train",
        "rows": len(sample_identities),
        "class_counts": sampling.get("selected_class_counts"),
        "identity_column": identity_column,
        **identity_fingerprints(spec["train_domain"], identity_column, sample_identities),
    }
    for field in ("domain", "split", "rows", "identity_column", "identity_sha256", "ordered_identity_sha256"):
        if training.get(field) != train_summary[field]:
            _fail(run, "metadata.json", f"training_context.{field}", "disagrees with training_sample.csv.gz")
    actual_fingerprints = identity_fingerprints(spec["train_domain"], identity_column, sample_identities)
    for record, field_name in ((training, "training_context"), (sampling, "sampling")):
        for fingerprint_name, value in actual_fingerprints.items():
            if record.get(fingerprint_name) != value:
                _fail(run, "metadata.json", f"{field_name}.{fingerprint_name}", "does not match saved identities")
    if sampling.get("requested_budget") != comparison["training_budget"]:
        _fail(run, "metadata.json", "sampling.requested_budget", "disagrees with requested training budget")
    if sampling.get("selected_training_rows") != len(sample_identities):
        _fail(run, "metadata.json", "sampling.selected_training_rows", "disagrees with training sample row count")
    if sampling.get("sampling_seed") != comparison["sampling_seed"]:
        _fail(run, "metadata.json", "sampling.sampling_seed", "disagrees with manifest")
    counts = sampling.get("selected_class_counts")
    if not isinstance(counts, dict) or set(counts) != {"0", "1"}:
        _fail(run, "metadata.json", "sampling.selected_class_counts", "must record both binary class counts")
    count_zero = _integer(counts["0"], run, "metadata.json", "sampling.selected_class_counts.0")
    count_one = _integer(counts["1"], run, "metadata.json", "sampling.selected_class_counts.1")
    if count_zero + count_one != len(sample_identities) or training.get("class_counts") != counts:
        _fail(run, "metadata.json", "training_context.class_counts", "inconsistent selected class counts")

    prepared = metadata.get("prepared_file_sha256")
    if not isinstance(prepared, dict) or not prepared:
        _fail(run, "metadata.json", "prepared_file_sha256", "missing prepared-data fingerprints")
    if not all(
        isinstance(key, str)
        and isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for key, value in prepared.items()
    ):
        _fail(run, "metadata.json", "prepared_file_sha256", "expected file names and SHA-256 fingerprints")

    if not isinstance(saved_metrics, dict) or set(saved_metrics) != set(domains):
        _fail(run, "metrics.json", "domains", f"expected exactly {domains}")
    predictions: dict[str, dict[int, dict[str, float | int]]] = {}
    prediction_order: dict[str, list[int]] = {}
    normalized_metrics: dict[str, dict[str, Any]] = {}
    for domain in domains:
        filename = f"predictions_{domain}_test.csv.gz"
        rows = _read_csv(run_dir / filename, run, filename)
        id_column = IDENTITY_COLUMNS[domain]
        required_columns = {"domain", "split", "row_index", id_column, "y_true", "y_probability", "y_pred"}
        if not rows or not required_columns.issubset(rows[0]):
            _fail(run, filename, "header/rows", f"requires columns {sorted(required_columns)} and at least one row")
        identities: list[int] = []
        labels: list[int] = []
        probabilities: list[float] = []
        predicted_labels: list[int] = []
        values: dict[int, dict[str, float | int]] = {}
        for position, row in enumerate(rows):
            if row["domain"] != domain or row["split"] != "test":
                _fail(run, filename, f"rows[{position}].domain/split", "does not identify the expected test partition")
            identity = _csv_int(row[id_column], run, filename, f"rows[{position}].{id_column}")
            row_index = _csv_int(row["row_index"], run, filename, f"rows[{position}].row_index")
            if domain != "nhanes" and row_index != identity:
                _fail(run, filename, f"rows[{position}].row_index", "must be the authoritative identity")
            if identity in values:
                _fail(run, filename, id_column, f"duplicate test identity {identity}")
            label = _csv_int(row["y_true"], run, filename, f"rows[{position}].y_true")
            predicted = _csv_int(row["y_pred"], run, filename, f"rows[{position}].y_pred")
            if label not in (0, 1) or predicted not in (0, 1):
                _fail(run, filename, f"rows[{position}].labels", "labels must be binary integers")
            probability = _csv_number(row["y_probability"], run, filename, f"rows[{position}].y_probability")
            if not 0 <= probability <= 1:
                _fail(run, filename, f"rows[{position}].y_probability", "must lie in [0, 1]")
            expected_predicted = int(probability >= threshold)
            if predicted != expected_predicted:
                _fail(run, filename, f"rows[{position}].y_pred", "does not match probability and saved threshold")
            identities.append(identity)
            labels.append(label)
            probabilities.append(probability)
            predicted_labels.append(predicted)
            values[identity] = {
                "y_true": label,
                "y_probability": probability,
                "y_pred": predicted,
            }
        predictions[domain] = values
        prediction_order[domain] = identities
        summary = _summary(domain, "test", identities, labels)
        _check_summary(summary, split_metadata.get(f"{domain}/test"), run, "metadata.json", f"splits.{domain}/test")

        calibration_bins = _integer(
            config.get("calibration_bins"), run, "config.json", "calibration_bins", minimum=1,
        )
        calculated = evaluate_binary(labels, probabilities, threshold, calibration_bins)
        recorded = saved_metrics[domain]
        if not isinstance(recorded, dict):
            _fail(run, "metrics.json", domain, "expected an object")
        for key in ("n_rows", "positive_count"):
            value = _integer(recorded.get(key), run, "metrics.json", f"{domain}.{key}")
            if value != calculated[key]:
                _fail(run, "metrics.json", f"{domain}.{key}", f"expected prediction-derived value {calculated[key]}")
        for key in (
            "positive_prevalence", "roc_auc", "average_precision", "f1", "precision",
            "recall", "log_loss", "brier_score", "ece", "threshold",
        ):
            _equal_metric(recorded.get(key), calculated[key], run, f"{domain}.{key}")
        confusion = recorded.get("confusion_matrix")
        if not isinstance(confusion, dict) or set(confusion) != {"tn", "fp", "fn", "tp"}:
            _fail(run, "metrics.json", f"{domain}.confusion_matrix", "expected tn/fp/fn/tp counts")
        for key, expected_count in calculated["confusion_matrix"].items():
            value = _integer(confusion[key], run, "metrics.json", f"{domain}.confusion_matrix.{key}")
            if value != expected_count:
                _fail(run, "metrics.json", f"{domain}.confusion_matrix.{key}", "disagrees with saved predictions")
        if recorded.get("warnings") != calculated["warnings"]:
            _fail(run, "metrics.json", f"{domain}.warnings", "disagrees with prediction-derived metric availability")
        calibration = recorded.get("calibration_bins")
        if not isinstance(calibration, list) or len(calibration) != calibration_bins:
            _fail(run, "metrics.json", f"{domain}.calibration_bins", "expected one saved row per configured bin")
        for bin_index, (saved_bin, expected_bin) in enumerate(zip(calibration, calculated["calibration_bins"])):
            if not isinstance(saved_bin, dict):
                _fail(run, "metrics.json", f"{domain}.calibration_bins[{bin_index}]", "expected an object")
            for key, expected_value in expected_bin.items():
                if key in {"bin", "count"}:
                    value = _integer(saved_bin.get(key), run, "metrics.json", f"{domain}.calibration_bins[{bin_index}].{key}")
                    if value != expected_value:
                        _fail(run, "metrics.json", f"{domain}.calibration_bins[{bin_index}].{key}", "disagrees with predictions")
                else:
                    _equal_metric(
                        saved_bin.get(key), expected_value, run,
                        f"{domain}.calibration_bins[{bin_index}].{key}",
                    )
        normalized_metrics[domain] = calculated

    calibration_path = run_dir / "calibration_bins.csv"
    calibration_rows = _read_csv(calibration_path, run, "calibration_bins.csv")
    calibration_columns = {
        "domain", "split", "bin", "lower_bound", "upper_bound", "count",
        "mean_predicted_probability", "observed_positive_rate", "absolute_gap",
    }
    if calibration_rows and not calibration_columns.issubset(calibration_rows[0]):
        _fail(run, "calibration_bins.csv", "header", f"requires columns {sorted(calibration_columns)}")
    expected_calibration_rows = len(domains) * calibration_bins
    if len(calibration_rows) != expected_calibration_rows:
        _fail(run, "calibration_bins.csv", "rows", f"expected {expected_calibration_rows} rows")
    calibration_seen: set[tuple[str, int]] = set()
    for index, row in enumerate(calibration_rows):
        domain = row["domain"]
        if domain not in domains or row["split"] != "test":
            _fail(run, "calibration_bins.csv", f"rows[{index}].domain/split", "unexpected domain or split")
        bin_index = _csv_int(row["bin"], run, "calibration_bins.csv", f"rows[{index}].bin")
        key = (domain, bin_index)
        if key in calibration_seen or not 0 <= bin_index < calibration_bins:
            _fail(run, "calibration_bins.csv", f"rows[{index}].bin", "duplicate or out-of-range calibration bin")
        calibration_seen.add(key)
        expected_bin = normalized_metrics[domain]["calibration_bins"][bin_index]
        if _csv_int(row["count"], run, "calibration_bins.csv", f"rows[{index}].count") != expected_bin["count"]:
            _fail(run, "calibration_bins.csv", f"rows[{index}].count", "disagrees with saved predictions")
        for key_name in (
            "lower_bound", "upper_bound", "mean_predicted_probability",
            "observed_positive_rate", "absolute_gap",
        ):
            value = row[key_name]
            expected_value = expected_bin[key_name]
            if value == "" and expected_value is None:
                continue
            actual = _csv_number(value, run, "calibration_bins.csv", f"rows[{index}].{key_name}")
            if expected_value is None or not math.isclose(
                actual, expected_value, rel_tol=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE,
            ):
                _fail(run, "calibration_bins.csv", f"rows[{index}].{key_name}", "disagrees with saved predictions")

    prediction_durations = metadata.get("prediction_duration_seconds")
    if (
        not isinstance(prediction_durations, dict)
        or set(prediction_durations) - set(domains)
    ):
        _fail(run, "metadata.json", "prediction_duration_seconds", f"expected optional durations for {domains}")
    for domain, duration in prediction_durations.items():
        value = _number(
            duration, run, "metadata.json", f"prediction_duration_seconds.{domain}",
            nullable=True,
        )
        if value is not None and value < 0:
            _fail(run, "metadata.json", f"prediction_duration_seconds.{domain}", "must be nonnegative or null")
    for field in ("fit_duration_seconds", "total_duration_seconds"):
        value = _number(metadata.get(field), run, "metadata.json", field)
        if value is None or value < 0:
            _fail(run, "metadata.json", field, "must be a nonnegative duration")
    if not isinstance(metadata.get("compute_settings"), dict):
        _fail(run, "metadata.json", "compute_settings", "missing recorded compute settings")
    compute = metadata["compute_settings"]
    if config["model"] == "catboost":
        if compute.get("task_type") != "CPU":
            _fail(run, "metadata.json", "compute_settings.task_type", "disagrees with expected CatBoost CPU execution")
    elif not str(compute.get("device", "")).startswith("cuda:"):
        _fail(run, "metadata.json", "compute_settings.device", "disagrees with expected TabPFN GPU execution")
    if isinstance(metadata.get("model"), dict) and metadata["model"].get("compute_settings") != compute:
        _fail(run, "metadata.json", "model.compute_settings", "disagrees with top-level compute settings")
    resources = metadata.get("resources")
    if not isinstance(resources, dict):
        _fail(run, "metadata.json", "resources", "missing runtime resource measurements")
    for field in ("cpu_peak_memory_bytes", "total_gpu_usage_bytes"):
        if resources.get(field) is not None:
            _integer(resources[field], run, "metadata.json", f"resources.{field}")
    gpu_memory = resources.get("peak_gpu_memory")
    if gpu_memory is not None:
        if not isinstance(gpu_memory, dict):
            _fail(run, "metadata.json", "resources.peak_gpu_memory", "expected an object or null")
        for field in ("allocated_bytes", "reserved_bytes"):
            _integer(gpu_memory.get(field), run, "metadata.json", f"resources.peak_gpu_memory.{field}")
        if not isinstance(gpu_memory.get("device"), str) or not isinstance(gpu_memory.get("scope"), str):
            _fail(run, "metadata.json", "resources.peak_gpu_memory", "device and scope descriptions are required")
    if manifest["report_schema_version"] == 2:
        _validate_v2_model_profile(spec, config, metadata, manifest)
    model = metadata.get("model")
    if not isinstance(model, dict):
        model = {}
    checkpoint_metadata = model.get("checkpoint")
    profile_id = {
        "catboost": "catboost_cpu",
        "tabpfn": "tabpfn_v35_float32",
        "tabicl": model.get("experimental_profile"),
        "ft_transformer": "ft_transformer_n2000_float32",
    }[spec["model"]]
    generation = (
        manifest.get("model_profiles", {}).get("catboost", {}).get("generation", "CatBoost")
        if spec["model"] == "catboost"
        else (
            checkpoint_metadata.get("generation")
            if isinstance(checkpoint_metadata, dict)
            else None
        )
    )
    if spec["model"] == "ft_transformer":
        generation = manifest["model_profiles"]["ft_transformer"]["generation"]
    return RunArtifacts(
        spec=spec,
        run_dir=run_dir,
        config=config,
        metadata=metadata,
        metrics=normalized_metrics,
        predictions=predictions,
        prediction_order=prediction_order,
        training_identities=sample_identities,
        artifact_hashes=hashes,
        display_names={
            "train_domain": display_domain_name(spec["train_domain"]),
            **{domain: display_domain_name(domain) for domain in domains},
            "model_profile_id": profile_id,
            "model_generation": generation,
        },
    )


def validate_comparison(manifest: dict[str, Any], manifest_path: Path) -> tuple[list[RunArtifacts], dict[str, Any]]:
    specs = _check_manifest(manifest)
    runs_root = (RESULTS_DIR / "predictive").resolve()
    loaded: list[RunArtifacts] = []
    resolved_dirs: set[Path] = set()
    for spec in specs:
        run_dir = _input_run_dir(spec, runs_root)
        if run_dir in resolved_dirs:
            raise ReportValidationError(f"manifest selects the same resolved run directory more than once: {run_dir}")
        resolved_dirs.add(run_dir)
        loaded.append(validate_run(spec, run_dir, manifest))

    by_role = {run.spec["role"]: run for run in loaded}
    calibration_bin_counts = {
        run.config["calibration_bins"] for run in loaded
    }
    if len(calibration_bin_counts) != 1:
        raise ReportValidationError(
            "selected runs use different calibration bin counts; one consistent descriptive "
            "calibration definition is required"
        )
    _check_calibration_boundary_compatibility(
        loaded,
        manifest["comparison"]["evaluation_domains"],
    )
    alignments: list[dict[str, Any]] = []
    domains = manifest["comparison"]["evaluation_domains"]
    for train_domain in manifest["comparison"]["training_domains"]:
        domain_runs = [
            by_role[f"{train_domain}_{model}"]
            for model in manifest["comparison"]["models"]
        ]
        reference_run = domain_runs[0]
        for compared_run in domain_runs[1:]:
            if reference_run.training_identities != compared_run.training_identities:
                raise ReportValidationError(
                    f"training identity mismatch for {train_domain}: "
                    f"{reference_run.spec['run_id']} vs {compared_run.spec['run_id']}"
                )
            reference_sampling = reference_run.metadata["sampling"]
            compared_sampling = compared_run.metadata["sampling"]
            for field in ("requested_budget", "sampling_seed", "algorithm", "available_training_rows",
                          "selected_training_rows", "selected_class_counts"):
                if reference_sampling.get(field) != compared_sampling.get(field):
                    raise ReportValidationError(
                        f"training sampling mismatch for {train_domain}, field sampling.{field}: "
                        f"{reference_run.spec['run_id']} vs {compared_run.spec['run_id']}"
                    )
            if reference_run.metadata["schema"] != compared_run.metadata["schema"]:
                raise ReportValidationError(f"feature order or target mismatch for training domain {train_domain}")
            for split_name in (f"{train_domain}/train", f"{train_domain}/val"):
                if reference_run.metadata["splits"][split_name] != compared_run.metadata["splits"][split_name]:
                    raise ReportValidationError(
                        f"prepared split fingerprints mismatch for training domain {train_domain}, split {split_name}"
                    )
            if reference_run.metadata["prepared_file_sha256"] != compared_run.metadata["prepared_file_sha256"]:
                raise ReportValidationError(f"prepared-data fingerprints mismatch for training domain {train_domain}")
            for domain in domains:
                reference_values = reference_run.predictions[domain]
                compared_values = compared_run.predictions[domain]
                if reference_values.keys() != compared_values.keys():
                    raise ReportValidationError(
                        f"test identity mismatch for {train_domain}-trained pair on evaluation domain {domain}"
                    )
                for identity in reference_values:
                    if reference_values[identity]["y_true"] != compared_values[identity]["y_true"]:
                        raise ReportValidationError(
                            f"test label mismatch for {train_domain}-trained pair on {domain}, identity {identity}"
                        )
                alignments.append({
                    "training_domain": train_domain,
                    "reference_run_id": reference_run.spec["run_id"],
                    "compared_run_id": compared_run.spec["run_id"],
                    "evaluation_domain": domain,
                    "identities_match": True,
                    "labels_match": True,
                    "prediction_order_differs": (
                        reference_run.prediction_order[domain] != compared_run.prediction_order[domain]
                    ),
                    "alignment_key": IDENTITY_COLUMNS[domain],
                })

    cross_run_population_checks: list[dict[str, Any]] = []
    reference = loaded[0]
    for domain in domains:
        reference_values = reference.predictions[domain]
        checks = []
        for run in loaded[1:]:
            values = run.predictions[domain]
            identities_match = reference_values.keys() == values.keys()
            labels_match = identities_match and all(
                reference_values[identity]["y_true"] == values[identity]["y_true"]
                for identity in reference_values
            )
            if not identities_match:
                raise ReportValidationError(
                    f"cross-run evaluation identity mismatch on {domain}: "
                    f"{reference.spec['run_id']} vs {run.spec['run_id']}"
                )
            if not labels_match:
                mismatched = next(
                    identity for identity in reference_values
                    if reference_values[identity]["y_true"] != values[identity]["y_true"]
                )
                raise ReportValidationError(
                    f"cross-run evaluation label mismatch on {domain}, identity {mismatched}: "
                    f"{reference.spec['run_id']} vs {run.spec['run_id']}"
                )
            check = {
                "evaluation_domain": domain,
                "reference_run_id": reference.spec["run_id"],
                "compared_run_id": run.spec["run_id"],
                "identities_match": identities_match,
                "labels_match": labels_match,
                "reference_rows": len(reference_values),
                "compared_rows": len(values),
                "alignment_key": IDENTITY_COLUMNS[domain],
                "prediction_order_differs": (
                    reference.prediction_order[domain] != run.prediction_order[domain]
                ),
            }
            checks.append(check)
            cross_run_population_checks.append(check)

    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    native_diagnostic_evidence = []
    if manifest["report_schema_version"] == 2:
        for run in loaded:
            if run.spec["model"] != "tabicl":
                continue
            diagnostics = run.metadata["model"]["native_diagnostics"]
            native_diagnostic_evidence.append({
                "run_id": run.spec["run_id"],
                "profile_id": run.metadata["model"]["experimental_profile"],
                "recovery_detection": diagnostics["recovery_detection"],
                "swap_fallback_detected": diagnostics["swap_fallback_detected"],
                "operation_count": len(diagnostics["operations"]),
                "limitations": diagnostics["limitations"],
            })
    return loaded, {
        "status": "passed",
        "report_schema_version": manifest["report_schema_version"],
        "output_schema_version": manifest.get("output_schema_version", 1),
        "claims": (
            "Saved-artifact agreement verifies recorded experiment consistency; it does not independently "
            "prove that every earlier data-preparation step was correct."
        ),
        "metric_absolute_and_relative_tolerance": METRIC_TOLERANCE,
        "identity_alignments": alignments,
        "cross_run_population_checks": cross_run_population_checks,
        "expected_methodological_differences": manifest["expected_methodological_differences"],
        "native_diagnostic_evidence": native_diagnostic_evidence,
        "validated_runs": [run.spec["run_id"] for run in loaded],
        "input_manifest_sha256": manifest_hash,
        "input_artifact_sha256": {
            run.spec["run_id"]: run.artifact_hashes for run in loaded
        },
    }


def _check_calibration_boundary_compatibility(
    runs: list[RunArtifacts],
    domains: list[str],
) -> None:
    if not runs:
        raise ReportValidationError("calibration boundary validation requires at least one run")
    reference = runs[0]
    for run in runs[1:]:
        for domain in domains:
            reference_boundaries = [
                (entry["lower_bound"], entry["upper_bound"])
                for entry in reference.metrics[domain]["calibration_bins"]
            ]
            compared_boundaries = [
                (entry["lower_bound"], entry["upper_bound"])
                for entry in run.metrics[domain]["calibration_bins"]
            ]
            if reference_boundaries != compared_boundaries:
                raise ReportValidationError(
                    "calibration bin boundaries differ across selected runs for "
                    f"evaluation domain {domain}: {reference.spec['run_id']} vs {run.spec['run_id']}"
                )
