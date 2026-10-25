from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src import data
from src.experiments.config import ExperimentConfig, load_config, positive_integer
from src.experiments.results import write_json, write_training_sample
from src.experiments.runner import (
    create_adapter, environment_metadata, load_partition, require_disjoint, utc_now,
)
from src.experiments.sampling import Partition, sampling_metadata, select_training
from src.models.base import ModelAdapter
from src.paths import PROJECT_ROOT, RESULTS_DIR
from src.domain_labels import display_domain_name

REFERENCE_BATCH_SIZE = 256
RTOL = 1e-5
ATOL = 1e-6
T = TypeVar("T")


def validate_request(
    config: ExperimentConfig, output_dir: str | Path, query_size: int,
    batch_sizes: list[int], repetitions: int,
) -> Path:
    if config.model != "tabpfn":
        raise ValueError("The benchmark requires a TabPFN configuration.")
    if config.model_params.get("inference_precision") != "float32":
        raise ValueError("The benchmark requires explicit inference_precision=float32.")
    device = config.model_params.get("device")
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("The benchmark requires an explicit CUDA device (cuda:<index>).")
    if config.train_sample_size is None or config.sampling_seed is None:
        raise ValueError("Configure train_sample_size and sampling_seed explicitly.")
    positive_integer(query_size, "query_size")
    positive_integer(repetitions, "repetitions")
    for size in batch_sizes:
        positive_integer(size, "batch_size")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch_sizes must be unique.")
    if REFERENCE_BATCH_SIZE not in batch_sizes:
        raise ValueError("batch_sizes must include reference batch size 256.")
    results = RESULTS_DIR.resolve()
    root = (results / "benchmarks").resolve()
    if root == results or not root.is_relative_to(results):
        raise ValueError("results/benchmarks/ must not resolve outside results/ or to its root.")
    output = (PROJECT_ROOT / output_dir).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("output_dir must resolve to a new directory under results/benchmarks/.")
    research = config.resolved_output_dir
    if output == research or output.is_relative_to(research) or research.is_relative_to(output):
        raise ValueError("Benchmark and research output directories must be separate.")
    return output


def select_queries(partition: Partition, query_size: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    positive_integer(query_size, "query_size")
    if partition.summary["split"] != "val":
        raise ValueError("Benchmark queries must come from validation.")
    if query_size > len(partition.X):
        raise ValueError("query_size exceeds the validation partition.")
    domain = partition.summary["domain"]
    identity = "seqn" if domain == "nhanes" else "row_index"
    values = partition.provenance[identity]
    if values.isna().any() or values.duplicated().any():
        raise ValueError("Validation identities must be nonmissing and unique.")
    positions = np.argsort(values.to_numpy(), kind="stable")[:query_size]
    queries = partition.X.iloc[positions].reset_index(drop=True).copy()
    identifiers = ["row_index", *(["seqn"] if domain == "nhanes" else [])]
    provenance = partition.provenance.iloc[positions][identifiers].reset_index(drop=True).copy()
    lines = [f"{domain}\t{identity}\t{int(value)}\n" for value in provenance[identity]]
    summary = {
        "domain": domain, "split": "val", "rows": query_size, "identity_column": identity,
        "ordered_identity_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "selection": "Sort validation by authoritative identity, then take first query_size rows; no label selection.",
    }
    provenance.insert(0, "query_order", np.arange(query_size))
    provenance.insert(0, "split", "val")
    provenance.insert(0, "domain", domain)
    return queries, provenance, summary


def is_cuda_oom(exc: BaseException) -> bool:
    # PyTorch CUDA OOM is a RuntimeError; do not classify CPU or arbitrary errors as CUDA OOM.
    return isinstance(exc, RuntimeError) and (
        "cuda out of memory" in str(exc).lower()
        or "cuda error: out of memory" in str(exc).lower()
        or type(exc).__name__ == "OutOfMemoryError"
    )


def error_details(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "notes": list(getattr(exc, "__notes__", []))}


def release_cuda(device: str) -> None:
    import torch

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def measure(
    adapter: ModelAdapter, action: Callable[[], T], record: dict[str, Any],
    query_size: int | None = None,
) -> T:
    record["status"] = "running"
    started = None
    try:
        adapter.synchronize()
        adapter.reset_peak_memory()
        started = time.perf_counter()
        result = action()
        adapter.synchronize()
        elapsed = time.perf_counter() - started
        record.update(status="completed", duration_seconds=elapsed)
        if query_size is not None:
            if elapsed <= 0:
                raise RuntimeError("Prediction duration must be positive to measure throughput.")
            record["rows_per_second"] = query_size / elapsed
        return result
    except BaseException as exc:
        record.update(
            status="out_of_memory" if is_cuda_oom(exc) else "failed",
            error=error_details(exc),
        )
        if started is not None:
            record["duration_seconds"] = time.perf_counter() - started
        raise
    finally:
        original = sys.exception()
        try:
            memory = adapter.resource_metadata().get("peak_gpu_memory")
            record["peak_allocated_bytes"] = memory["allocated_bytes"] if memory else None
            record["peak_reserved_bytes"] = memory["reserved_bytes"] if memory else None
        except BaseException as exc:
            if original is None:
                raise
            original.add_note(f"Could not collect memory diagnostics: {exc}")


def validate_probability(value: NDArray, query_size: int) -> None:
    if value.shape != (query_size,):
        raise ValueError(f"Expected exactly {query_size} one-dimensional probabilities; got {value.shape}.")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Prediction probabilities must be finite and in [0, 1].")


def comparison(value: NDArray, reference: NDArray, threshold: float) -> dict[str, Any]:
    difference = np.abs(value - reference)
    return {
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "allclose": bool(np.allclose(value, reference, rtol=RTOL, atol=ATOL)),
        "threshold_disagreements": int(np.count_nonzero((value >= threshold) != (reference >= threshold))),
    }


def summarize_case(case: dict[str, Any], repetitions: int) -> None:
    complete = [row for row in case["repetitions"] if row["status"] == "completed"]
    case["numerically_eligible"] = (
        case["execution_status"] == "completed" and len(complete) == repetitions
        and all(row["allclose"] for row in complete)
    )
    case["completed_repetitions"] = len(complete)
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        measured = [row[field] for row in case["repetitions"] if row.get(field) is not None]
        case[field] = max(measured) if measured else None
    if complete:
        durations = [row["duration_seconds"] for row in complete]
        case.update(
            median_duration_seconds=float(np.median(durations)),
            median_rows_per_second=float(np.median([row["rows_per_second"] for row in complete])),
            min_duration_seconds=min(durations), max_duration_seconds=max(durations),
        )


SUMMARY_FIELDS = (
    "batch_size", "status", "execution_status", "numerically_eligible", "completed_repetitions",
    "context_preparation_seconds", "warmup_seconds", "median_duration_seconds",
    "median_rows_per_second", "min_duration_seconds", "max_duration_seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "speedup_vs_256",
    "failing_operation", "failing_repetition",
)
REPETITION_FIELDS = (
    "batch_size", "repetition", "status", "duration_seconds", "rows_per_second",
    "peak_allocated_bytes", "peak_reserved_bytes", "max_absolute_difference",
    "mean_absolute_difference", "allclose", "threshold_disagreements", "probability_artifact",
    "failing_operation", "error_type", "error_message",
)


def persist(output: Path, report: dict[str, Any]) -> None:
    for case in report["cases"]:
        summarize_case(case, report["benchmark_settings"]["repetitions"])
    baseline = next(case for case in report["cases"] if case["batch_size"] == REFERENCE_BATCH_SIZE)
    if baseline["numerically_eligible"]:
        for case in report["cases"]:
            if case.get("median_duration_seconds"):
                case["speedup_vs_256"] = baseline["median_duration_seconds"] / case["median_duration_seconds"]
    write_json(output / "benchmark.json", report)
    summaries, repetitions = [], []
    for case in report["cases"]:
        row = {field: case.get(field) for field in SUMMARY_FIELDS}
        row["context_preparation_seconds"] = case["context_preparation"].get("duration_seconds")
        row["warmup_seconds"] = case["warmup"].get("duration_seconds")
        summaries.append(row)
        for repetition in case["repetitions"]:
            row = {field: repetition.get(field) for field in REPETITION_FIELDS}
            row.update(
                batch_size=case["batch_size"],
                error_type=repetition.get("error", {}).get("type"),
                error_message=repetition.get("error", {}).get("message"),
            )
            repetitions.append(row)
    for filename, rows, fields in (
        ("benchmark_summary.csv", summaries, SUMMARY_FIELDS),
        ("benchmark_repetitions.csv", repetitions, REPETITION_FIELDS),
    ):
        path = output / filename
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def run(
    config: ExperimentConfig, *, output_dir: str | Path, query_size: int = 2049,
    batch_sizes: list[int] | None = None, repetitions: int = 3,
    command_arguments: dict[str, Any] | None = None,
) -> Path:
    requested = [256, 512, 1024] if batch_sizes is None else list(batch_sizes)
    output = validate_request(config, output_dir, query_size, requested, repetitions)
    output.mkdir(parents=True, exist_ok=False)
    order = [REFERENCE_BATCH_SIZE, *(size for size in requested if size != REFERENCE_BATCH_SIZE)]
    report: dict[str, Any] = {
        "status": "running", "purpose": "Validation-only inference benchmark, not a predictive research result.",
        "started_at_utc": utc_now(), "ended_at_utc": None,
        "command_arguments": command_arguments,
        "configuration": config.to_dict(),
        "benchmark_settings": {
            "query_size": query_size, "requested_batch_sizes": requested, "execution_order": order,
            "repetitions": repetitions, "reference_batch_size": REFERENCE_BATCH_SIZE,
            "rtol": RTOL, "atol": ATOL, "threshold": config.threshold,
        },
        "memory_scope": "Process PyTorch allocator peaks reset per operation, not total GPU usage.",
        "timing_caveat": "Fixed case order, warm-up/cache state and laptop thermals can affect timings; small differences are not statistically established.",
        "cases": [{
            "batch_size": size, "status": "not_run", "execution_status": "not_run",
            "numerically_eligible": False, "completed_repetitions": 0,
            "context_preparation": {"status": "not_run", "duration_seconds": None},
            "warmup": {"status": "not_run", "duration_seconds": None}, "repetitions": [],
            "median_duration_seconds": None, "median_rows_per_second": None,
            "min_duration_seconds": None, "max_duration_seconds": None,
            "peak_allocated_bytes": None, "peak_reserved_bytes": None, "speedup_vs_256": None,
        } for size in requested],
        "artifacts": {
            "report": "benchmark.json", "summary": "benchmark_summary.csv",
            "repetitions": "benchmark_repetitions.csv",
            "training_sample": "training_sample.csv.gz", "validation_queries": "validation_queries.csv.gz",
        },
    }
    device = config.model_params["device"]
    reference = None
    active_case = None
    active_repetition = None
    operation = "data_loading"
    try:
        persist(output, report)
        report["environment"] = environment_metadata("tabpfn")
        schema = data.load_feature_schema(config.feature_set)
        full = load_partition(config.train_domain, "train", config.feature_set, schema)
        val = load_partition(config.validation_domain, "val", config.feature_set, schema)
        require_disjoint(full, val)
        selected = select_training(full, config.train_sample_size, config.sampling_seed)
        queries, provenance, query_summary = select_queries(val, query_size)
        report.update(
            sampling=sampling_metadata(full, selected, config.train_sample_size, config.sampling_seed),
            training_context=selected.summary, validation_queries=query_summary,
            feature_order=list(schema.features),
        )
        write_training_sample(output, config.train_domain, selected.provenance)
        provenance.to_csv(output / "validation_queries.csv.gz", index=False, compression="gzip")
        print(
            f"Selected {len(selected.X)} training context rows for "
            f"{display_domain_name(config.train_domain, include_id=True)} and "
            f"{len(queries)} validation queries; "
            f"requested device={device}, precision=float32. No test partitions.",
            flush=True,
        )
        persist(output, report)
        for size in order:
            active_case = next(case for case in report["cases"] if case["batch_size"] == size)
            case = active_case
            active_repetition = None
            case.update(status="running", execution_status="running")
            adapter = None
            handled_oom = None
            print(f"Batch {size}: starting fresh adapter.", flush=True)
            try:
                operation = "model_construction"
                persist(output, report)
                random.seed(config.seed)
                np.random.seed(config.seed)
                adapter = create_adapter(config, schema)
                operation = "context_preparation"
                persist(output, report)
                measure(
                    adapter, lambda: adapter.fit(selected.X, selected.y, val.X, val.y),
                    case["context_preparation"],
                )
                operation = "model_metadata"
                case["model"] = adapter.metadata()
                settings = case["model"]["compute_settings"]
                print(
                    f"Batch {size}: context prepared in {case['context_preparation']['duration_seconds']:.3f}s; "
                    f"resolved device={settings['device']}, precision={settings['forced_dtype']}, "
                    f"autocast={settings['autocast_enabled']}.",
                    flush=True,
                )
                persist(output, report)
                operation = "warmup"
                warmup = measure(
                    adapter, lambda: adapter.predict_proba_batched(queries, size), case["warmup"],
                )
                operation = "warmup_probability_validation"
                validate_probability(warmup, query_size)
                del warmup
                print(f"Batch {size}: warm-up complete in {case['warmup']['duration_seconds']:.3f}s.", flush=True)
                persist(output, report)
                for repetition in range(1, repetitions + 1):
                    row: dict[str, Any] = {
                        "repetition": repetition, "status": "running", "duration_seconds": None,
                        "rows_per_second": None, "peak_allocated_bytes": None,
                        "peak_reserved_bytes": None, "max_absolute_difference": None,
                        "mean_absolute_difference": None, "allclose": None,
                        "threshold_disagreements": None, "probability_artifact": None,
                    }
                    active_repetition = row
                    case["repetitions"].append(row)
                    operation = "prediction"
                    persist(output, report)
                    probability = measure(
                        adapter, lambda: adapter.predict_proba_batched(queries, size), row, query_size,
                    )
                    operation = "probability_serialization"
                    filename = f"probabilities_batch{size}_repetition{repetition}.npy"
                    np.save(output / filename, probability, allow_pickle=False)
                    row["probability_artifact"] = filename
                    operation = "probability_validation"
                    validate_probability(probability, query_size)
                    if reference is None:
                        reference = probability.copy()
                        report["reference_probability_artifact"] = filename
                    operation = "agreement"
                    row.update(comparison(probability, reference, config.threshold))
                    print(
                        f"Batch {size}, repetition {repetition}: {row['duration_seconds']:.3f}s, "
                        f"{row['rows_per_second']:.1f} rows/s, agreement={'passed' if row['allclose'] else 'FAILED'}.",
                        flush=True,
                    )
                    persist(output, report)
                    if size == REFERENCE_BATCH_SIZE and not row["allclose"]:
                        case.update(status="agreement_failed", execution_status="completed")
                        raise RuntimeError("Reference repetitions failed numerical agreement; baseline is unstable.")
                    active_repetition = None
                case.update(
                    status="completed" if all(row["allclose"] for row in case["repetitions"]) else "agreement_failed",
                    execution_status="completed",
                )
            except BaseException as exc:
                oom = is_cuda_oom(exc)
                if case["status"] != "agreement_failed":
                    case.update(status="out_of_memory" if oom else "failed", execution_status="out_of_memory" if oom else "failed")
                case.update(error=error_details(exc), failing_operation=operation)
                case["failing_repetition"] = active_repetition["repetition"] if active_repetition is not None else None
                if active_repetition is not None and not (
                    operation == "agreement" and case["status"] == "agreement_failed"
                ):
                    active_repetition.update(
                        status="out_of_memory" if oom else "failed",
                        error=error_details(exc), failing_operation=operation,
                    )
                try:
                    persist(output, report)
                except BaseException as diagnostic_error:
                    exc.add_note(f"Could not save benchmark diagnostics: {diagnostic_error}")
                    raise exc
                if not oom or size == REFERENCE_BATCH_SIZE:
                    raise
                print(f"Batch {size}: CUDA OOM during {operation}; no retry. Cleaning up before next case.", flush=True)
                # A handled OOM traceback can retain prediction-frame GPU tensors.
                traceback.clear_frames(exc.__traceback__)
                handled_oom = exc.with_traceback(None)
            finally:
                original = sys.exception()
                adapter = None
                try:
                    release_cuda(device)
                except BaseException as cleanup_error:
                    if original is not None:
                        original.add_note(f"CUDA cleanup also failed: {cleanup_error}")
                    elif handled_oom is not None:
                        handled_oom.add_note(f"CUDA cleanup also failed: {cleanup_error}")
                        raise handled_oom
                    else:
                        operation = "cleanup"
                        case.update(
                            status="failed", execution_status="failed", failing_operation=operation,
                            error=error_details(cleanup_error),
                        )
                        raise
            persist(output, report)
            print(f"Batch {size}: {case['status']}; numerically eligible={case['numerically_eligible']}.", flush=True)
        eligible = [case for case in report["cases"] if case["numerically_eligible"]]
        fastest = min(eligible, key=lambda case: case["median_duration_seconds"])
        report["fastest_eligible_batch_size"] = fastest["batch_size"]
        report["status"] = (
            "completed" if all(case["status"] == "completed" for case in report["cases"]) else "completed_with_issues"
        )
        report["ended_at_utc"] = utc_now()
        persist(output, report)
    except BaseException as exc:
        report.update(
            status="failed", ended_at_utc=utc_now(), error=error_details(exc),
            failing_operation=operation,
            failing_batch_size=active_case["batch_size"] if active_case is not None else None,
            failing_repetition=active_repetition["repetition"] if active_repetition is not None else None,
        )
        try:
            persist(output, report)
        except BaseException as diagnostic_error:
            exc.add_note(f"Could not save benchmark failure report: {diagnostic_error}")
        raise
    print(
        f"Benchmark {report['status']}; fastest eligible measured batch={report['fastest_eligible_batch_size']}. "
        f"Output: {output}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only float32 CUDA TabPFN throughput/agreement benchmark; no test data, downloads, or research runs.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-size", type=int, default=2049)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        validate_request(config, args.output_dir, args.query_size, args.batch_sizes, args.repetitions)
    except ValueError as exc:
        parser.error(str(exc))
    output = run(
        config, output_dir=args.output_dir, query_size=args.query_size,
        batch_sizes=args.batch_sizes, repetitions=args.repetitions, command_arguments=vars(args),
    )
    status = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))["status"]
    if status == "completed_with_issues":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
