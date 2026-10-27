from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src import data
from src.domain_labels import display_domain_name
from src.experiments.config import ExperimentConfig, load_config
from src.experiments.results import create_run_directory, write_json, write_training_sample
from src.experiments.runner import environment_metadata, load_partition, require_disjoint, utc_now
from src.experiments.sampling import Partition, partition_summary, sampling_metadata, select_training
from src.models.ft_transformer import FTTransformerAdapter
from src.models.ft_transformer_config import FT_DEFAULT_PARAMS, resolve_ft_transformer_params
from src.paths import RESULTS_DIR

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
RTOL = 1e-5
ATOL = 1e-6
TRAIN_ROWS = 64
VALIDATION_ROWS = 32
QUERY_ROWS = 32
QUERY_BATCH_SIZES = (3, 7, 16)
RESOURCE_MAX_EPOCHS = 10
RESOURCE_DEFAULT_EPOCHS = 2
SMOKE_ROOT = "smoke"

_FUNCTIONAL_PARAMS = {
    "n_blocks": 1,
    "d_block": 16,
    "attention_n_heads": 4,
    "attention_dropout": 0.0,
    "ffn_d_hidden": 32,
    "ffn_d_hidden_multiplier": None,
    "ffn_activation": "ReGLU",
    "ffn_dropout": 0.0,
    "residual_dropout": 0.0,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "train_batch_size": 16,
    "max_epochs": 3,
    "validation_frequency": 1,
    "validation_batch_size": 32,
    "patience": 20,
    "min_delta": 0.001,
    "preprocessing": "standard_scaler",
    "unknown_category_policy": "training_mode",
    "precision": "float32",
    "device": "cpu",
    "class_weighting": "none",
}


def _validate_base_config(config: ExperimentConfig) -> dict[str, Any]:
    if config.model != "ft_transformer":
        raise ValueError("Smoke workflow requires an ft_transformer configuration.")
    if config.train_domain not in ("source", "kaggle"):
        raise ValueError("Smoke workflow only permits source or kaggle training.")
    if config.validation_domain != config.train_domain:
        raise ValueError("Smoke workflow requires same-domain validation.")
    if config.feature_set != "strict_common_16":
        raise ValueError("Smoke workflow requires feature_set='strict_common_16'.")
    if config.seed != 42 or config.sampling_seed != 42 or config.train_sample_size != 2000:
        raise ValueError("Smoke base configuration must use n2000 and seed 42 for training sampling.")
    if config.prediction_batch_size != 256 or config.threshold != 0.5:
        raise ValueError("Smoke base configuration must retain prediction_batch_size=256 and threshold=0.5.")
    if config.early_stopping_rounds is not None:
        raise ValueError("FT-Transformer smoke configurations require early_stopping_rounds=null.")
    resolved = resolve_ft_transformer_params(config.model_params)
    if resolved != FT_DEFAULT_PARAMS:
        raise ValueError("Smoke configuration must use the exact approved FT research profile as its base.")
    return resolved


def resolve_smoke_config(
    config: ExperimentConfig,
    *,
    output_dir: str,
    mode: str,
    resource_epochs: int | None = None,
    device_override: str | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Validate mode arguments and create the effective config before data access."""
    if mode not in ("functional", "resource", "feasibility"):
        raise ValueError("mode must be functional, resource, or feasibility.")
    base_params = _validate_base_config(config)
    if resource_epochs is not None:
        if mode != "resource":
            raise ValueError("--resource-epochs is only valid in resource mode.")
        if type(resource_epochs) is not int or not 2 <= resource_epochs <= RESOURCE_MAX_EPOCHS:
            raise ValueError("--resource-epochs must be an integer from 2 through 10.")
    if device_override is not None and not (
        device_override == "cpu"
        or (
            device_override.startswith("cuda:")
            and device_override[5:].isascii()
            and device_override[5:].isdigit()
        )
    ):
        raise ValueError("--device must be explicitly 'cpu' or 'cuda:<index>'.")

    params = dict(base_params)
    if mode == "functional":
        params.update(_FUNCTIONAL_PARAMS)
    elif mode == "resource":
        params["max_epochs"] = resource_epochs or RESOURCE_DEFAULT_EPOCHS
    if device_override is not None:
        params["device"] = device_override
    params = resolve_ft_transformer_params(params)

    try:
        output_path = Path(output_dir)
        config_output = str(output_path)
        effective = replace(
            config,
            output_dir=config_output,
            model_params=params,
            train_sample_size=TRAIN_ROWS if mode == "functional" else 2000,
            sampling_seed=42,
            prediction_batch_size=(
                QUERY_BATCH_SIZES[1] if mode == "functional" else 256
            ),
        )
        resolved_output = effective.resolved_output_dir
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid smoke output/configuration: {exc}") from exc
    smoke_root = (RESULTS_DIR / SMOKE_ROOT).resolve()
    if resolved_output == smoke_root or not resolved_output.is_relative_to(smoke_root):
        raise ValueError("Smoke output must be a fresh run directory below results/smoke/.")

    details = {
        "mode": mode,
        "purpose": {
            "functional": (
                "Small diagnostic profile; does not use the research architecture or population, "
                "and is not evidence of convergence or predictive superiority."
            ),
            "resource": (
                "Intentionally truncated resource diagnostic; not evidence of convergence or "
                "predictive superiority."
            ),
            "feasibility": (
                "Full approved training profile and complete validation only; no test evaluation. "
                "Completion does not complete test evaluation or M10."
            ),
        }[mode],
        "device_override": device_override,
        "base_device": base_params["device"],
        "effective_device": params["device"],
        "training_budget_truncated": mode == "resource",
        "epoch_cap": params["max_epochs"],
        "population_policy": (
            "Deterministic class-stratified functional subsets."
            if mode == "functional"
            else "Shared identity-sorted select_training sampler, n=2000, seed=42; full validation."
        ),
    }
    return effective, details


def _stratified_subset(partition: Partition, size: int, seed: int) -> Partition:
    if partition.summary["split"] not in ("train", "val"):
        raise ValueError("Functional smoke subsets may only use train or val partitions.")
    if type(size) is not int or size < 2 or size > len(partition.X):
        raise ValueError(f"Requested functional subset size {size} is unavailable.")
    labels = partition.y.to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Functional smoke selection requires binary labels.")
    by_class = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in by_class):
        raise ValueError("Functional smoke selection requires both classes.")
    allocation_positive = int(np.floor(size * len(by_class[1]) / len(labels) + 0.5))
    allocation_positive = min(max(allocation_positive, 1, size - len(by_class[0])), size - 1, len(by_class[1]))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(by_class[0], size - allocation_positive, replace=False),
        rng.choice(by_class[1], allocation_positive, replace=False),
    ])
    positions = np.sort(chosen)
    X = partition.X.iloc[positions].reset_index(drop=True).copy()
    y = partition.y.iloc[positions].reset_index(drop=True).copy()
    provenance = partition.provenance.iloc[positions].reset_index(drop=True).copy()
    domain = partition.summary["domain"]
    identity_column = partition.summary["identity_column"]
    return Partition(
        X=X,
        y=y,
        provenance=provenance,
        identities=frozenset((domain, int(value)) for value in provenance[identity_column]),
        summary=partition_summary(domain, partition.summary["split"], provenance, y),
    )


def select_smoke_populations(
    full_train: Partition,
    full_validation: Partition,
    *,
    mode: str,
    seed: int,
) -> tuple[Partition, Partition, dict[str, Any]]:
    if mode == "functional":
        training = _stratified_subset(full_train, TRAIN_ROWS, seed)
        validation = _stratified_subset(full_validation, VALIDATION_ROWS, seed + 1)
        sampling = {
            "algorithm": "Class-stratified sampling without replacement; selected rows restored to gateway order.",
            "training_seed": seed,
            "validation_seed": seed + 1,
            "training": training.summary,
            "validation": validation.summary,
        }
    elif mode in ("resource", "feasibility"):
        training = select_training(full_train, 2000, 42)
        validation = full_validation
        sampling = sampling_metadata(full_train, training, 2000, 42)
        sampling["algorithm"] = (
            "Existing predictive-runner select_training sampler; identity-sorted, n=2000, seed=42."
        )
    else:
        raise ValueError("mode must be functional, resource, or feasibility.")
    for name, partition in (("training", training), ("validation", validation)):
        if set(partition.y.to_numpy()) != {0, 1}:
            raise ValueError(f"{name} population must contain both binary classes.")
    require_disjoint(training, validation)
    return training, validation, sampling


def _write_population(path: Path, domain: str, split: str, provenance: pd.DataFrame) -> None:
    identity = provenance.copy()
    identity.insert(0, "sample_order", np.arange(len(identity)))
    identity.insert(0, "split", split)
    identity.insert(0, "domain", domain)
    identity.to_csv(
        path, index=False, compression={"method": "gzip", "mtime": 0},
    )


def _select_queries(validation: Partition, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(validation.X) < QUERY_ROWS:
        raise ValueError(f"Validation needs at least {QUERY_ROWS} rows for the fixed query checks.")
    positions = np.sort(
        np.random.default_rng(seed).choice(len(validation.X), QUERY_ROWS, replace=False)
    )
    queries = validation.X.iloc[positions].reset_index(drop=True).copy()
    provenance = validation.provenance.iloc[positions].reset_index(drop=True).copy()
    labels = validation.y.iloc[positions].reset_index(drop=True).copy()
    return queries, provenance, {
        "rows": QUERY_ROWS,
        "selection_policy": "Seeded NumPy default_rng sample without replacement; selected gateway positions restored to ascending order.",
        "selection_seed": seed,
        "partition_summary": partition_summary(
            validation.summary["domain"], validation.summary["split"], provenance, labels,
        ),
        "gateway_positions": positions.tolist(),
    }


def _validate_probabilities(values: NDArray[np.float64], rows: int) -> None:
    if (
        values.shape != (rows,)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Smoke predictions must be finite (n,) probabilities in [0,1].")


def _compare(
    actual: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    threshold: float,
    query_rows: int,
    batch_size: int | None,
) -> dict[str, Any]:
    _validate_probabilities(reference, len(reference))
    _validate_probabilities(actual, query_rows)
    if actual.shape != reference.shape:
        raise ValueError("Compared prediction vectors must have matching shapes.")
    difference = np.abs(actual - reference)
    disagreements = int(np.count_nonzero((actual >= threshold) != (reference >= threshold)))
    passed = bool(np.allclose(actual, reference, rtol=RTOL, atol=ATOL) and disagreements == 0)
    return {
        "status": "passed" if passed else "failed",
        "query_rows": query_rows,
        "batch_size": batch_size,
        "max_absolute_probability_difference": float(difference.max()),
        "threshold_disagreement_count": disagreements,
        "rtol": RTOL,
        "atol": ATOL,
    }


def _operation(
    name: str,
    action: Callable[[], T],
    *,
    adapter: FTTransformerAdapter | None,
    device: str,
    report: dict[str, Any],
    report_path: Path,
) -> T:
    record: dict[str, Any] = {"status": "running"}
    report.setdefault("operations", {})[name] = record
    started: float | None = None
    try:
        if adapter is not None:
            adapter.synchronize()
            adapter.reset_peak_memory()
        elif device.startswith("cuda:"):
            import torch

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        record["resource_scope"] = (
            "Adapter allocator peak reset immediately before this operation."
            if device.startswith("cuda:")
            else "CPU; GPU allocator metrics are not applicable."
        )
        write_json(report_path, report)
        started = time.perf_counter()
        result = action()
        if adapter is not None:
            adapter.synchronize()
            record["resources"] = adapter.resource_metadata()
        elif device.startswith("cuda:"):
            import torch

            torch.cuda.synchronize(device)
            record["resources"] = {
                "peak_gpu_memory": {
                    "device": device,
                    "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "scope": "PyTorch allocator peak during operation after explicit reset.",
                },
            }
        else:
            record["resources"] = {"peak_gpu_memory": None}
        record.update(status="completed", duration_seconds=time.perf_counter() - started)
        return result
    except BaseException as exc:
        record.update(status="failed", error={
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        original = sys.exception()
        if started is not None and "duration_seconds" not in record:
            record["duration_seconds"] = time.perf_counter() - started
        try:
            write_json(report_path, report)
        except BaseException as report_error:
            if original is None:
                raise
            LOGGER.exception("Could not write smoke operation diagnostics.")
            original.add_note(f"Could not persist smoke operation diagnostics: {report_error}")


def _run_comparison(
    name: str,
    action: Callable[[], NDArray[np.float64]],
    *,
    reference: NDArray[np.float64],
    threshold: float,
    batch_size: int | None,
    adapter: FTTransformerAdapter,
    device: str,
    report: dict[str, Any],
    report_path: Path,
) -> NDArray[np.float64]:
    values = _operation(
        name, action, adapter=adapter, device=device, report=report, report_path=report_path,
    )
    check = _compare(
        values, reference, threshold=threshold, query_rows=len(reference), batch_size=batch_size,
    )
    report.setdefault("checks", {})[name] = check
    write_json(report_path, report)
    if check["status"] != "passed":
        raise AssertionError(f"{name} prediction comparison failed: {check}")
    return values


def run(
    config: ExperimentConfig,
    *,
    output_dir: str,
    mode: str = "functional",
    resource_epochs: int | None = None,
    device_override: str | None = None,
) -> Path:
    effective, mode_details = resolve_smoke_config(
        config, output_dir=output_dir, mode=mode,
        resource_epochs=resource_epochs, device_override=device_override,
    )
    output = create_run_directory(effective)
    report_path = output / "smoke.json"
    requested_config = config.to_dict()
    effective_config = effective.to_dict()
    report: dict[str, Any] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "purpose": mode_details["purpose"],
        "mode": mode,
        "mode_details": mode_details,
        "requested_configuration": requested_config,
        "effective_configuration": effective_config,
        "requested_configuration_file": None,
        "training": {},
        "validation": {},
        "checks": {},
        "operations": {},
        "criteria": {"rtol": RTOL, "atol": ATOL, "threshold": config.threshold},
        "information_boundary": (
            "Only source/kaggle train and same-domain val partitions are permitted. "
            "No test partition or NHANES data is loaded."
        ),
        "status_scope": "Smoke status describes only this train/validation diagnostic, not M10 or research performance.",
    }
    adapter: FTTransformerAdapter | None = None
    started = time.perf_counter()
    try:
        write_json(output / "requested_config.json", requested_config)
        write_json(output / "effective_config.json", effective_config)
        write_json(report_path, report)

        schema = data.load_feature_schema(effective.feature_set)
        if (
            len(set(schema.features)) != len(schema.features)
            or set(schema.features) != set(schema.numeric_features) | set(schema.categorical_features)
            or set(schema.numeric_features) & set(schema.categorical_features)
        ):
            raise ValueError("Loaded feature schema is internally inconsistent.")
        report["schema"] = {
            "feature_order": list(schema.features),
            "numeric_features": list(schema.numeric_features),
            "categorical_features": list(schema.categorical_features),
            "target": schema.target,
        }
        report["environment"] = environment_metadata("ft_transformer")
        report["seed"] = effective.seed
        report["device"] = {
            "effective": effective.model_params["device"],
            "explicit_override": device_override,
            "intended_feasibility_device": "cuda:0",
            "matches_intended_feasibility_device": (
                effective.model_params["device"] == "cuda:0"
            ),
        }
        full_train = load_partition(effective.train_domain, "train", effective.feature_set, schema)
        full_validation = load_partition(
            effective.validation_domain, "val", effective.feature_set, schema,
        )
        if (
            full_train.summary["domain"] != effective.train_domain
            or full_train.summary["split"] != "train"
            or full_validation.summary["domain"] != effective.train_domain
            or full_validation.summary["split"] != "val"
        ):
            raise ValueError("Smoke loader returned a partition outside the permitted train/val boundary.")
        require_disjoint(full_train, full_validation)
        report["full_partition_sizes"] = {
            "training": len(full_train.X),
            "validation": len(full_validation.X),
        }

        training, validation, report["sampling"] = select_smoke_populations(
            full_train, full_validation, mode=mode, seed=effective.seed,
        )
        report["training"] = {
            "population_size": len(training.X),
            "class_counts": training.summary["class_counts"],
            "identities": training.summary,
            "validation_complete": mode != "functional",
            "epoch_cap": effective.model_params["max_epochs"],
            "truncated": mode == "resource",
        }
        report["validation"] = {
            "population_size": len(validation.X),
            "class_counts": validation.summary["class_counts"],
            "identities": validation.summary,
            "complete": mode != "functional",
            "selection_policy": "Deterministic class-stratified subset." if mode == "functional"
            else "Complete same-domain fixed validation population.",
        }
        _write_population(
            output / "training_identities.csv.gz",
            effective.train_domain,
            "train",
            training.provenance,
        )
        _write_population(
            output / "validation_identities.csv.gz",
            effective.train_domain,
            "val",
            validation.provenance,
        )
        if mode != "functional":
            write_training_sample(output, effective.train_domain, training.provenance)
        queries, query_provenance, query_summary = _select_queries(validation, effective.seed + 2)
        query_summary["identity_summary"] = partition_summary(
            effective.train_domain, "val", query_provenance,
            validation.y.iloc[query_summary["gateway_positions"]].reset_index(drop=True),
        )
        report["validation_queries"] = query_summary
        _write_population(
            output / "validation_query_identities.csv.gz",
            effective.train_domain,
            "val",
            query_provenance,
        )
        print(
            f"FT-Transformer {mode} smoke: {effective.train_domain}, "
            f"train={len(training.X)}, validation={len(validation.X)}, "
            f"device={effective.model_params['device']}, "
            f"epoch_cap={effective.model_params['max_epochs']}",
            flush=True,
        )
        adapter = FTTransformerAdapter(
            schema.features,
            schema.categorical_features,
            schema.target,
            seed=effective.seed,
            model_params=effective.model_params,
            prediction_batch_size=effective.prediction_batch_size or 256,
        )
        _operation(
            "fit", lambda: adapter.fit(training.X, training.y, validation.X, validation.y),
            adapter=adapter,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        fitted_metadata = adapter.metadata()
        report["model"] = fitted_metadata
        report["training"].update({
            "executed_epochs": fitted_metadata["training"]["epochs_executed"],
            "stopping_reason": fitted_metadata["training"]["stopping_reason"],
            "best_epoch": fitted_metadata["training"]["best_epoch"],
            "history": fitted_metadata["training"]["training_history"],
            "validation_unseen_category_counts": (
                fitted_metadata["training"]["validation_unseen_category_counts"]
            ),
        })
        history = fitted_metadata["training"]["training_history"]
        report["fit_timing"] = {
            "duration_seconds": report["operations"]["fit"]["duration_seconds"],
            "epoch_training_seconds": [item["training_duration_seconds"] for item in history],
            "epoch_validation_seconds": [item["validation_duration_seconds"] for item in history],
            "combined_fit_overhead_seconds": report["operations"]["fit"]["duration_seconds"] - sum(
                item["training_duration_seconds"] + item["validation_duration_seconds"]
                for item in history
            ),
            "combined_fit_overhead_note": (
                "Fit duration minus summed epoch durations; includes construction, checkpoint "
                "copies/restoration, and other overhead and is not pure preprocessing time."
            ),
        }
        report["validation"]["unseen_category_counts"] = (
            fitted_metadata["training"]["validation_unseen_category_counts"]
        )
        write_json(report_path, report)

        query_before = queries.copy(deep=True)
        reference = _operation(
            "reference_prediction",
            lambda: adapter.predict_proba_batched(queries, QUERY_BATCH_SIZES[1]),
            adapter=adapter,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        _validate_probabilities(reference, len(queries))
        report["checks"]["reference_prediction"] = {
            "status": "passed",
            "query_rows": len(queries),
            "batch_size": QUERY_BATCH_SIZES[1],
            "finite_and_in_range": True,
            "rtol": RTOL,
            "atol": ATOL,
            "threshold_disagreement_count": 0,
            "max_absolute_probability_difference": 0.0,
        }
        np.save(output / "reference_probabilities.npy", reference, allow_pickle=False)

        def compare_predictions(name: str, actual: NDArray[np.float64], batch: int | None) -> None:
            report["checks"][name] = _compare(
                actual,
                reference,
                threshold=effective.threshold,
                query_rows=len(queries),
                batch_size=batch,
            )
            write_json(report_path, report)
            if report["checks"][name]["status"] != "passed":
                raise AssertionError(f"{name} prediction check failed: {report['checks'][name]}")

        repeated = _operation(
            "repeated_prediction",
            lambda: adapter.predict_proba_batched(queries, QUERY_BATCH_SIZES[1]),
            adapter=adapter,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        compare_predictions("repeated_prediction", repeated, QUERY_BATCH_SIZES[1])
        permutation = np.random.default_rng(effective.seed + 3).permutation(len(queries))
        permuted = _operation(
            "permuted_prediction",
            lambda: adapter.predict_proba_batched(queries.iloc[permutation], QUERY_BATCH_SIZES[1]),
            adapter=adapter,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        restored_order = permuted[np.argsort(permutation)]
        report["validation_queries"]["permutation"] = permutation.tolist()
        compare_predictions("permuted_prediction_restored_order", restored_order, QUERY_BATCH_SIZES[1])
        for batch_size in QUERY_BATCH_SIZES:
            values = _operation(
                f"prediction_batch_{batch_size}",
                lambda batch_size=batch_size: adapter.predict_proba_batched(queries, batch_size),
                adapter=adapter,
                device=effective.model_params["device"],
                report=report,
                report_path=report_path,
            )
            compare_predictions(f"prediction_batch_{batch_size}", values, batch_size)
        pd.testing.assert_frame_equal(queries, query_before)
        report["checks"]["query_input_unchanged"] = {"status": "passed"}
        write_json(report_path, report)

        artifact = output / adapter.artifact_name
        _operation(
            "save",
            lambda: adapter.save(artifact),
            adapter=adapter,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        if not artifact.is_file():
            raise FileNotFoundError(f"Adapter did not create its artifact: {artifact}")
        report["model"] = adapter.metadata()
        report["artifact"] = str(artifact)
        write_json(report_path, report)
        del adapter
        adapter = None
        gc.collect()
        if effective.model_params["device"].startswith("cuda:"):
            import torch

            torch.cuda.empty_cache()

        restored_adapter = _operation(
            "reload",
            lambda: FTTransformerAdapter.load(
                artifact, device=effective.model_params["device"],
            ),
            adapter=None,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        reloaded = _operation(
            "reloaded_prediction",
            lambda: restored_adapter.predict_proba_batched(queries, QUERY_BATCH_SIZES[1]),
            adapter=restored_adapter,
            device=effective.model_params["device"],
            report=report,
            report_path=report_path,
        )
        report["checks"]["save_reload"] = _compare(
            reloaded,
            reference,
            threshold=effective.threshold,
            query_rows=len(queries),
            batch_size=QUERY_BATCH_SIZES[1],
        )
        write_json(report_path, report)
        if report["checks"]["save_reload"]["status"] != "passed":
            raise AssertionError(f"Save/reload prediction check failed: {report['checks']['save_reload']}")
        report["reloaded_model"] = restored_adapter.metadata()
        del restored_adapter
        gc.collect()
        if effective.model_params["device"].startswith("cuda:"):
            import torch

            torch.cuda.empty_cache()
        report.update(
            status="completed",
            ended_at_utc=utc_now(),
            total_duration_seconds=time.perf_counter() - started,
            training_budget_truncated=mode == "resource",
            completion_note=(
                "No test evaluation was performed; feasibility-mode completion does not complete M10."
            ),
        )
        write_json(report_path, report)
    except BaseException as exc:
        report.update(
            status="failed",
            ended_at_utc=utc_now(),
            total_duration_seconds=time.perf_counter() - started,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        try:
            write_json(report_path, report)
        except BaseException as report_error:
            LOGGER.exception("Could not persist smoke failure record; preserving original exception.")
            exc.add_note(f"Could not persist smoke failure record: {report_error}")
        raise

    print(
        f"FT-Transformer {mode} smoke completed for "
        f"{display_domain_name(effective.train_domain, include_id=True)}. Output: {output}",
        flush=True,
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bounded FT-Transformer train/validation-only smoke workflow; never loads test partitions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode", choices=("functional", "resource", "feasibility"),
        default="functional",
    )
    parser.add_argument(
        "--resource-epochs", type=int,
        help="Resource-mode epoch cap (2 by default; maximum 10).",
    )
    parser.add_argument(
        "--device",
        help="Explicit smoke-only device override: cpu or cuda:<index>; no automatic fallback.",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        effective, _details = resolve_smoke_config(
            config,
            output_dir=args.output_dir,
            mode=args.mode,
            resource_epochs=args.resource_epochs,
            device_override=args.device,
        )
        run(
            config,
            output_dir=args.output_dir,
            mode=args.mode,
            resource_epochs=args.resource_epochs,
            device_override=args.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
