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


def _check_manifest(manifest: Any) -> list[dict[str, Any]]:
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
    else:
        if config.get("early_stopping_rounds") is not None:
            _fail(run, "config.json", "early_stopping_rounds", "TabPFN must not use early stopping")
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
    alignments: list[dict[str, Any]] = []
    domains = manifest["comparison"]["evaluation_domains"]
    for train_domain in manifest["comparison"]["training_domains"]:
        catboost = by_role[f"{train_domain}_catboost"]
        tabpfn = by_role[f"{train_domain}_tabpfn"]
        if catboost.training_identities != tabpfn.training_identities:
            raise ReportValidationError(
                f"training identity mismatch for {train_domain}: "
                f"{catboost.spec['run_id']} vs {tabpfn.spec['run_id']}"
            )
        cat_sampling = catboost.metadata["sampling"]
        tab_sampling = tabpfn.metadata["sampling"]
        for field in ("requested_budget", "sampling_seed", "algorithm", "available_training_rows",
                      "selected_training_rows", "selected_class_counts"):
            if cat_sampling.get(field) != tab_sampling.get(field):
                raise ReportValidationError(
                    f"training sampling mismatch for {train_domain}, field sampling.{field}: "
                    f"{catboost.spec['run_id']} vs {tabpfn.spec['run_id']}"
                )
        if catboost.metadata["schema"] != tabpfn.metadata["schema"]:
            raise ReportValidationError(f"feature order or target mismatch for training domain {train_domain}")
        for split_name in (f"{train_domain}/train", f"{train_domain}/val"):
            if catboost.metadata["splits"][split_name] != tabpfn.metadata["splits"][split_name]:
                raise ReportValidationError(
                    f"prepared split fingerprints mismatch for training domain {train_domain}, split {split_name}"
                )
        if catboost.metadata["prepared_file_sha256"] != tabpfn.metadata["prepared_file_sha256"]:
            raise ReportValidationError(f"prepared-data fingerprints mismatch for training domain {train_domain}")
        for domain in domains:
            cat_map = catboost.predictions[domain]
            tab_map = tabpfn.predictions[domain]
            if cat_map.keys() != tab_map.keys():
                raise ReportValidationError(
                    f"test identity mismatch for {train_domain}-trained pair on evaluation domain {domain}"
                )
            for identity in cat_map:
                if cat_map[identity]["y_true"] != tab_map[identity]["y_true"]:
                    raise ReportValidationError(
                        f"test label mismatch for {train_domain}-trained pair on {domain}, identity {identity}"
                    )
            cat_order = catboost.prediction_order[domain]
            tab_order = tabpfn.prediction_order[domain]
            alignments.append({
                "training_domain": train_domain,
                "evaluation_domain": domain,
                "identities_match": True,
                "labels_match": True,
                "prediction_order_differs": cat_order != tab_order,
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
    return loaded, {
        "status": "passed",
        "claims": (
            "Saved-artifact agreement verifies recorded experiment consistency; it does not independently "
            "prove that every earlier data-preparation step was correct."
        ),
        "metric_absolute_and_relative_tolerance": METRIC_TOLERANCE,
        "identity_alignments": alignments,
        "cross_run_population_checks": cross_run_population_checks,
        "validated_runs": [run.spec["run_id"] for run in loaded],
        "input_manifest_sha256": manifest_hash,
        "input_artifact_sha256": {
            run.spec["run_id"]: run.artifact_hashes for run in loaded
        },
    }
