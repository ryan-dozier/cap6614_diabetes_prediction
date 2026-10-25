from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from src import data
from src.experiments.config import load_config, positive_integer
from src.experiments.results import create_run_directory, write_json, write_training_sample
from src.experiments.runner import load_partition, require_disjoint, utc_now
from src.experiments.sampling import partition_summary, sampling_metadata, select_training
from src.models.tabpfn import TabPFNAdapter
from src.domain_labels import display_domain_name

T = TypeVar("T")


def measure_operation(
    name: str, action: Callable[[], T], device: str, report: dict, report_path: Path,
) -> T:
    """Record synchronized operation timing and process CUDA allocator peaks."""
    cuda = None
    if device.startswith("cuda:"):
        import torch

        cuda = torch.cuda
        cuda.synchronize(device)
        cuda.reset_peak_memory_stats(device)
    operation = {"status": "running"}
    report["operations"][name] = operation
    write_json(report_path, report)
    started = time.perf_counter()
    try:
        result = action()
        if cuda is not None:
            cuda.synchronize(device)
        operation["status"] = "completed"
        return result
    except BaseException as exc:
        operation.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        original = sys.exception()
        operation["duration_seconds"] = time.perf_counter() - started
        try:
            if cuda is not None:
                operation["peak_allocated_bytes"] = int(cuda.max_memory_allocated(device))
                operation["peak_reserved_bytes"] = int(cuda.max_memory_reserved(device))
            write_json(report_path, report)
        except BaseException as reporting_error:
            if original is not None:
                original.add_note(f"Could not save operation diagnostics: {reporting_error}")
            else:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional local TabPFN smoke check using TRAIN/VAL ONLY; never evaluates test labels or downloads weights.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", default="results/smoke/tabpfn_local")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--query-size", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--inference-precision", choices=("auto", "autocast", "float32"),
        help="Explicit smoke-only precision override; research configuration is not modified.",
    )
    args = parser.parse_args()
    for name in ("sample_size", "query_size", "batch_size"):
        positive_integer(getattr(args, name), name)
    config = load_config(args.config)
    if config.model != "tabpfn":
        parser.error("Use a TabPFN configuration.")
    config = replace(
        config, output_dir=args.output_dir, train_sample_size=args.sample_size,
        sampling_seed=config.sampling_seed if config.sampling_seed is not None else 42,
        prediction_batch_size=args.batch_size,
        model_params={
            **config.model_params,
            **({"inference_precision": args.inference_precision} if args.inference_precision else {}),
        },
    )
    output = create_run_directory(config)
    report = {
        "status": "running", "purpose": "Train/validation-only local smoke check, not a research result.",
        "started_at_utc": utc_now(), "config": config.to_dict(),
        "query_rows": args.query_size, "batch_size": args.batch_size,
        "operations": {}, "intended_chunked_path_status": "not_started",
        "memory_scope": "Process PyTorch allocator peaks reset before each operation; not total device VRAM.",
    }
    started = time.perf_counter()
    try:
        write_json(output / "smoke.json", report)
        schema = data.load_feature_schema(config.feature_set)
        full = load_partition(config.train_domain, "train", config.feature_set, schema)
        val = load_partition(config.validation_domain, "val", config.feature_set, schema)
        require_disjoint(full, val)
        if args.query_size > len(val.X):
            raise ValueError("query_size exceeds available validation rows.")
        selected = select_training(full, config.train_sample_size, config.sampling_seed)
        report["sampling"] = sampling_metadata(full, selected, config.train_sample_size, config.sampling_seed)
        report["training_context"] = selected.summary
        report["validation_queries"] = partition_summary(
            config.validation_domain, "val", val.provenance.iloc[:args.query_size],
            val.y.iloc[:args.query_size],
        )
        report["query_selection"] = "First query_size rows in deterministic data-gateway validation order."
        val.provenance.iloc[:args.query_size].to_csv(
            output / "validation_queries.csv.gz", index=False, compression="gzip",
        )
        write_training_sample(output, config.train_domain, selected.provenance)
        adapter = TabPFNAdapter(
            schema.features, schema.categorical_features, schema.target,
            seed=config.seed, model_params=config.model_params,
        )
        device = adapter.params["device"]
        report_path = output / "smoke.json"
        measure_operation(
            "context_preparation", lambda: adapter.fit(selected.X, selected.y, val.X, val.y),
            device, report, report_path,
        )
        report["model"] = adapter.metadata()
        write_json(report_path, report)
        queries = val.X.iloc[:args.query_size].copy()
        chunked = measure_operation(
            "chunked_prediction", lambda: adapter.predict_proba_batched(queries, args.batch_size),
            device, report, report_path,
        )
        report["intended_chunked_path_status"] = "completed"
        report["probability_validation"] = {
            "rows": len(chunked), "finite": bool(np.isfinite(chunked).all()),
            "in_unit_interval": bool(((chunked >= 0) & (chunked <= 1)).all()),
            "minimum": float(chunked.min()), "maximum": float(chunked.max()),
        }
        np.save(output / "chunked_probabilities.npy", chunked, allow_pickle=False)
        artifact = output / adapter.artifact_name
        measure_operation("save_fitted_state", lambda: adapter.save(artifact), device, report, report_path)
        del adapter
        gc.collect()
        if device.startswith("cuda:"):
            import torch

            torch.cuda.empty_cache()
        restored = measure_operation(
            "reload_fitted_state", lambda: TabPFNAdapter.load(artifact, device=device),
            device, report, report_path,
        )
        reloaded = measure_operation(
            "reloaded_chunked_prediction", lambda: restored.predict_proba_batched(queries, args.batch_size),
            device, report, report_path,
        )
        report["reload_max_absolute_difference"] = float(np.max(np.abs(chunked - reloaded)))
        np.testing.assert_allclose(chunked, reloaded, rtol=1e-5, atol=1e-6)
        report["reload_agreement"] = "passed"
        unchunked = None
        try:
            unchunked = measure_operation(
                "unchunked_comparison", lambda: restored.predict_proba(queries),
                device, report, report_path,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            report["unchunked_comparison_status"] = "out_of_memory"
            print("Unchunked comparison ran out of memory; intended chunked inference and reload passed. No retry.")
        if unchunked is not None:
            report["unchunked_comparison_status"] = "completed"
            report["chunking_max_absolute_difference"] = float(np.max(np.abs(unchunked - chunked)))
            report["chunking_mean_absolute_difference"] = float(np.mean(np.abs(unchunked - chunked)))
            report["chunking_threshold_disagreements"] = int(
                np.count_nonzero((unchunked >= config.threshold) != (chunked >= config.threshold))
            )
            report["chunking_allclose_rtol_1e_minus5_atol_1e_minus6"] = bool(
                np.allclose(unchunked, chunked, rtol=1e-5, atol=1e-6)
            )
            np.save(output / "unchunked_probabilities.npy", unchunked, allow_pickle=False)
        if unchunked is not None and not report["chunking_allclose_rtol_1e_minus5_atol_1e_minus6"]:
            raise RuntimeError(
                "Chunked and unchunked inference differ beyond smoke-test tolerance "
                f"(max absolute difference={report['chunking_max_absolute_difference']:.9g}, "
                f"precision={config.model_params.get('inference_precision', 'autocast')}); "
                "inspect smoke.json before research runs. No precision or tolerance fallback was applied."
            )
        report.update(status="completed", ended_at_utc=utc_now(), duration_seconds=time.perf_counter() - started)
        write_json(output / "smoke.json", report)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), ended_at_utc=utc_now())
        if report["intended_chunked_path_status"] == "not_started":
            report["intended_chunked_path_status"] = "failed_or_not_reached"
        try:
            write_json(output / "smoke.json", report)
        except BaseException as reporting_error:
            exc.add_note(f"Could not save smoke failure report: {reporting_error}")
        raise
    print(
        f"Train/val-only smoke check passed for "
        f"{display_domain_name(config.train_domain, include_id=True)}; "
        f"no test partitions used. Report: {output / 'smoke.json'}"
    )


if __name__ == "__main__":
    main()
