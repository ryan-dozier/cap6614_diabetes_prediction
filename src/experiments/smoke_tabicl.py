from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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
from src.experiments.benchmark_tabpfn import select_queries
from src.experiments.config import ExperimentConfig, load_config, positive_integer
from src.experiments.results import create_run_directory, write_json, write_training_sample
from src.experiments.runner import environment_metadata, load_partition, require_disjoint, utc_now
from src.experiments.sampling import sampling_metadata, select_training
from src.models.tabicl import CHECKPOINT_FILENAME, TabICLAdapter, _runtime
from src.paths import PROJECT_ROOT, RESULTS_DIR

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
FP32_ATOL = 64 * float(np.finfo(np.float32).eps)
HEADROOM_BYTES = 512 * 1024**2
FROZEN_CONTEXT_HASHES = {
    "source": "7d81147cce93cd5c56b2e66484f9f8902ae4ef1784bab270c32016e43f44d170",
    "kaggle": "612fcfb9dae65a36b591f44065baef3cf629b028435737db4a5480b09469fe3b",
}
CONTEXT_REFERENCES = {
    "source": PROJECT_ROOT / "results/predictive/catboost_source_strict_common_16_n2000_seed42/training_sample.csv.gz",
    "kaggle": PROJECT_ROOT / "results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/training_sample.csv.gz",
}


def _memory(torch: Any, device: str) -> dict[str, Any]:
    import psutil

    ram = psutil.virtual_memory()
    result = {
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "host_total_bytes": ram.total, "host_available_bytes": ram.available,
    }
    if device.startswith("cuda:"):
        free, total = torch.cuda.mem_get_info(device)
        result.update(device_free_bytes=int(free), device_total_bytes=int(total))
    return result


def measure_operation(
    name: str, action: Callable[[], T], torch: Any, device: str,
    report: dict[str, Any], report_path: Path,
) -> T:
    record: dict[str, Any] = {"status": "running"}
    report["operations"][name] = record
    started = None
    try:
        if device.startswith("cuda:"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        record["memory_before"] = _memory(torch, device)
        write_json(report_path, report)
        started = time.perf_counter()
        result = action()
        if device.startswith("cuda:"):
            torch.cuda.synchronize(device)
        record.update(status="completed", duration_seconds=time.perf_counter() - started)
        return result
    except BaseException as exc:
        record.update(status="failed", error={
            "type": type(exc).__name__, "message": str(exc),
            "notes": list(getattr(exc, "__notes__", [])), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        original = sys.exception()
        try:
            if started is not None and "duration_seconds" not in record:
                record["duration_seconds"] = time.perf_counter() - started
            record["memory_after"] = _memory(torch, device)
            if device.startswith("cuda:"):
                record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
                record["allocator_headroom_passed"] = (
                    record["peak_reserved_bytes"] <= record["memory_after"]["device_total_bytes"] - HEADROOM_BYTES
                )
            write_json(report_path, report)
        except Exception as exc:
            if original is None:
                raise
            LOGGER.exception("Could not save operation diagnostics; preserving original failure.")
            original.add_note(f"Could not save operation diagnostics: {exc}")


def validate_probability(value: NDArray, count: int) -> None:
    if (
        value.shape != (count,) or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any()
    ):
        raise ValueError("Smoke predictions must be finite real (n,) probabilities in [0,1].")


def comparison(actual: NDArray, reference: NDArray) -> dict[str, Any]:
    validate_probability(reference, len(reference))
    validate_probability(actual, len(reference))
    delta = np.abs(actual - reference)
    flips = (actual >= 0.5) != (reference >= 0.5)
    far = (np.abs(actual - 0.5) > FP32_ATOL) & (np.abs(reference - 0.5) > FP32_ATOL)
    close = bool((delta <= FP32_ATOL).all())
    return {
        "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "bitwise_equal": bool(np.array_equal(actual, reference)), "within_fixed_criterion": close,
        "threshold_disagreements": int(flips.sum()),
        "threshold_disagreements_outside_boundary_band": int((flips & far).sum()),
        "eligible": close and not bool(flips.any()),
        "interpretation": "Any threshold flip needs review; tolerance is an engineering gate, not a guaranteed error bound.",
    }


def _reload_worker(request_path: Path) -> None:
    import json

    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = request_path.parent
    report: dict[str, Any] = {
        "status": "running", "operations": {}, "environment": environment_metadata("tabicl"),
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
    }
    path = output / "reload.json"
    adapter = None
    try:
        if not report["offline"]:
            raise RuntimeError("Reload worker requires HF_HUB_OFFLINE=1.")
        sidecar_path = Path(request["artifact"]).with_suffix(".adapter.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if Path(sidecar["model_params"]["model_path"]).exists():
            raise RuntimeError("Isolated checkpoint must be unavailable before the embedded-weight reload check.")
        torch, _ = _runtime()
        device = request["device"]
        adapter = measure_operation(
            "reload", lambda: TabICLAdapter.load(
                Path(request["artifact"]), device=device, diagnostic_path=output / "native_reload.log",
            ), torch, device, report, path,
        )
        queries = pd.read_csv(request["queries"], dtype=request["query_dtypes"], float_precision="round_trip")
        probability = measure_operation(
            "reloaded_prediction", lambda: adapter.predict_proba_batched(queries, request["batch_size"]),
            torch, device, report, path,
        )
        validate_probability(probability, len(queries))
        np.save(output / "reloaded_probabilities.npy", probability, allow_pickle=False)
        report["model"] = adapter.metadata()
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error={
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        original = sys.exception()
        try:
            if adapter is not None:
                report["native_diagnostics"] = adapter.diagnostic_metadata()
            write_json(path, report)
        except Exception as exc:
            if original is None:
                raise
            LOGGER.exception("Could not save reload failure diagnostics.")
            original.add_note(f"Could not save reload diagnostics: {exc}")


def reload_in_subprocess(output: Path, artifact: Path, queries: pd.DataFrame, device: str, batch_size: int) -> dict[str, Any]:
    import json

    query_path = output / "reload_queries.csv"
    queries.to_csv(query_path, index=False)
    request = output / "reload_request.json"
    write_json(request, {
        "artifact": str(artifact), "queries": str(query_path),
        "device": device, "batch_size": batch_size,
        "query_dtypes": {column: str(dtype) for column, dtype in queries.dtypes.items()},
    })
    try:
        with (output / "reload_process.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "src.experiments.smoke_tabicl", "--reload-request", str(request)],
                cwd=PROJECT_ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"},
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Fresh-process TabICL reload failed ({result.returncode}); inspect reload.json and reload_process.log.")
        return json.loads((output / "reload.json").read_text(encoding="utf-8"))
    finally:
        original = sys.exception()
        for temporary in (query_path, request):
            try:
                temporary.unlink()
            except OSError as exc:
                if original is None:
                    raise
                LOGGER.exception("Could not remove reload temporary file %s.", temporary)
                original.add_note(f"Could not remove reload temporary file {temporary}: {exc}")


def _validate_request(
    config: ExperimentConfig, output_dir: str, sample_size: int, query_size: int, batch_size: int, suite: str,
) -> ExperimentConfig:
    if suite not in ("equivalence", "feasibility"):
        raise ValueError("suite must be equivalence or feasibility.")
    if config.model != "tabicl" or config.train_domain not in ("source", "kaggle"):
        raise ValueError("Use a TabICL source/kaggle configuration.")
    if config.feature_set != "strict_common_16" or config.seed != 42 or config.sampling_seed != 42 or config.threshold != 0.5:
        raise ValueError("Stage-3 protocol requires strict_common_16, both seeds 42, and threshold 0.5.")
    if config.model_params.get("use_amp", False) is not False:
        raise ValueError("The fixed FP32 engineering criteria do not certify AMP. Use an explicitly reviewed separate protocol.")
    if config.model_params.get("n_estimators", 8) != 8 or config.model_params.get("batch_size", 1) != 1:
        raise ValueError("This stage-3 acceptance protocol requires 8 ensemble members and native ensemble batch_size=1.")
    for name, value in (("sample_size", sample_size), ("query_size", query_size), ("batch_size", batch_size)):
        positive_integer(value, name)
    if sample_size not in (512, 2000) or query_size < (257 if sample_size == 2000 else 65):
        raise ValueError("Use n512 with at least 65 queries or n2000 with at least 257 queries.")
    if suite == "feasibility" and (sample_size != 2000 or batch_size != 128):
        raise ValueError("Feasibility requires the frozen n2000 context and 128-row query chunks.")
    if suite == "feasibility" and not config.model_params.get("device", "cuda:0").startswith("cuda:"):
        raise ValueError("Laptop feasibility certification requires an explicit CUDA device.")
    root = (RESULTS_DIR / "smoke").resolve()
    if root == RESULTS_DIR.resolve() or not root.is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError("Smoke root must be inside results/.")
    config = replace(
        config, output_dir=output_dir, train_sample_size=sample_size, prediction_batch_size=batch_size,
        experiment_name=f"tabicl_v2_{config.train_domain}_none_fp32_n{sample_size}_{suite}",
    )
    if config.resolved_output_dir == root or not config.resolved_output_dir.is_relative_to(root):
        raise ValueError("TabICL smoke output must be a NEW run directory under results/smoke/.")
    return config


def run(
    config: ExperimentConfig, *, output_dir: str, sample_size: int = 512,
    query_size: int = 65, batch_size: int = 128, suite: str = "equivalence",
) -> Path:
    config = _validate_request(config, output_dir, sample_size, query_size, batch_size, suite)
    output = create_run_directory(config)
    path = output / "smoke.json"
    report: dict[str, Any] = {
        "status": "running", "started_at_utc": utc_now(), "operations": {}, "checks": {},
        "purpose": "Train/validation-only engineering smoke check, not a predictive research result.",
        "domain": config.train_domain, "domain_display": display_domain_name(config.train_domain),
        "configuration": config.to_dict(), "suite": suite,
        "criteria": {"absolute_tolerance": FP32_ATOL, "relative_tolerance": 0, "headroom_bytes": HEADROOM_BYTES},
        "criterion_interpretation": "Fixed engineering acceptance criteria; not mathematically guaranteed error bounds.",
        "readiness_scope": "One domain only; M07 requires real feasibility for BOTH frozen n2000 contexts.",
        "numerically_eligible": False, "resource_eligible": False,
    }
    adapter = None
    started = time.perf_counter()
    try:
        write_json(path, report)
        report["environment"] = environment_metadata("tabicl")
        schema = data.load_feature_schema(config.feature_set)
        full = load_partition(config.train_domain, "train", config.feature_set, schema)
        val = load_partition(config.validation_domain, "val", config.feature_set, schema)
        require_disjoint(full, val)
        selected = select_training(full, sample_size, config.sampling_seed)
        report["sampling"] = sampling_metadata(full, selected, sample_size, config.sampling_seed)
        if sample_size == 2000:
            if selected.summary["ordered_identity_sha256"] != FROZEN_CONTEXT_HASHES[config.train_domain]:
                raise ValueError("n2000 ordered training identity hash differs from the frozen comparison.")
            reference = pd.read_csv(CONTEXT_REFERENCES[config.train_domain])
            expected = selected.provenance["row_index"].tolist()
            if (
                len(reference) != 2000 or reference["sample_order"].tolist() != list(range(2000))
                or set(reference["domain"]) != {config.train_domain} or set(reference["split"]) != {"train"}
                or reference["row_index"].tolist() != expected
            ):
                raise ValueError("n2000 identity sequence differs from the saved matched context.")
        write_training_sample(output, config.train_domain, selected.provenance)
        queries, provenance, summary = select_queries(val, query_size)
        provenance.to_csv(output / "validation_queries.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        report["queries"] = summary
        train_before, labels_before = selected.X.copy(deep=True), selected.y.copy(deep=True)
        query_before = queries.copy(deep=True)
        with tempfile.TemporaryDirectory(prefix="isolated_checkpoint_", dir=output) as temporary:
            original_path = (PROJECT_ROOT / Path(config.model_params["model_path"]).expanduser()).resolve()
            isolated_path = Path(temporary) / CHECKPOINT_FILENAME
            shutil.copyfile(original_path, isolated_path)
            report["checkpoint_source_path"] = str(original_path)
            adapter = TabICLAdapter(
                schema.features, schema.categorical_features, schema.target, seed=config.seed,
                model_params={**config.model_params, "model_path": str(isolated_path)},
            )
            adapter.set_diagnostic_path(output / "native_diagnostics.log")
            torch, device = adapter._torch, adapter.params["device"]

            def measure(name: str, action: Callable[[], T]) -> T:
                return measure_operation(name, action, torch, device, report, path)

            measure("preparation", lambda: adapter.fit(selected.X, selected.y, val.X, val.y))
            pd.testing.assert_frame_equal(selected.X, train_before)
            pd.testing.assert_series_equal(selected.y, labels_before)
            report["model"] = adapter.metadata()
            report["checks"]["fit_input_nonmutation"] = True

            def predict(name: str, frame: pd.DataFrame, chunk: int | None = None) -> NDArray:
                before = frame.copy(deep=True)
                result = measure(name, lambda: adapter.predict_proba_batched(frame, chunk))
                pd.testing.assert_frame_equal(frame, before)
                validate_probability(result, len(frame))
                np.save(output / f"{name}.npy", result, allow_pickle=False)
                return result

            baseline = predict("prediction_cold", queries, batch_size)

            def compare(name: str, actual: NDArray, reference: NDArray = baseline) -> None:
                report["checks"][name] = comparison(actual, reference)
                write_json(path, report)

            for index in range(3):
                compare(f"repeat_{index}", predict(f"repeat_{index}", queries, batch_size))
            compare("unchunked", predict("unchunked", queries))
            for chunk in (1, 17, 32, 128):
                compare(f"chunk_{chunk}", predict(f"chunk_{chunk}", queries, chunk))
            permutation = np.random.default_rng(42).permutation(len(queries))
            permuted = predict("permutation", queries.iloc[permutation], batch_size)
            compare("permutation", permuted[np.argsort(permutation)])
            count = min(3, len(queries))
            compare("sentinels_alone", predict("sentinels_alone", queries.iloc[:count]), baseline[:count])
            if schema.categorical_features:
                unseen = queries.iloc[:count].copy()
                feature = schema.categorical_features[0]
                unseen[feature] = unseen[feature].astype(str)
                candidate = "__tabicl_unseen_smoke__"
                existing = set(selected.X[feature].astype(str))
                while candidate in existing:
                    candidate += "_"
                unseen[feature] = candidate
                predict("unseen_category", unseen)
                report["checks"]["unseen_category_finite_nonmutating"] = True
            invalid_frames = {
                "feature_order": queries.loc[:, list(reversed(schema.features))],
                "extra_feature": queries.assign(__extra_predictor=1),
                "missing_feature": queries.drop(columns=schema.features[0]),
                "missing_value": queries.assign(**{schema.features[0]: np.nan}),
                "nonfinite_value": queries.assign(**{schema.numeric_features[0]: np.inf}),
            }
            for name, invalid in invalid_frames.items():
                try:
                    adapter.predict_proba(invalid)
                except ValueError as exc:
                    report["checks"][f"reject_{name}"] = {"passed": True, "message": str(exc)}
                else:
                    raise AssertionError(f"Adapter accepted invalid {name} input.")
            pd.testing.assert_frame_equal(queries, query_before)
            report["checks"]["prediction_input_nonmutation"] = True
            if suite == "feasibility":
                timing_queries, timing_ids, timing_summary = select_queries(val, 2049)
                timing_ids.to_csv(output / "timing_queries.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
                report["timing_queries"] = timing_summary
                warmup = predict("timing_warmup", timing_queries, 128)
                durations = []
                for index in range(3):
                    name = f"timing_repeat_{index}"
                    compare(name, predict(name, timing_queries, 128), warmup)
                    durations.append(report["operations"][name]["duration_seconds"])
                report["timing"] = {
                    "repetitions_seconds": durations, "median_seconds": float(np.median(durations)),
                    "median_rows_per_second": float(2049 / np.median(durations)),
                }
            artifact = output / adapter.artifact_name
            measure("save", lambda: adapter.save(artifact))
            report["model"] = adapter.metadata()
            report["native_diagnostics"] = adapter.diagnostic_metadata()
        report["isolated_checkpoint_removed_before_reload"] = not isolated_path.exists()
        del adapter
        adapter = None
        gc.collect()
        if device.startswith("cuda:"):
            torch.cuda.empty_cache()
        reload_report = reload_in_subprocess(output, artifact, queries, device, batch_size)
        report["reload"] = reload_report
        report["checks"]["reload"] = comparison(
            np.load(output / "reloaded_probabilities.npy", allow_pickle=False), baseline,
        )
        numerical = [item for item in report["checks"].values() if isinstance(item, dict) and "eligible" in item]
        report["numerically_eligible"] = all(item["eligible"] for item in numerical)
        diagnostics = [report["native_diagnostics"], reload_report["native_diagnostics"]]
        recovery = any(item["recovery_detection"] != "no_recovery_detected" for item in diagnostics)
        swap = any(item["swap_fallback_detected"] for item in diagnostics)
        operations = [*report["operations"].values(), *reload_report["operations"].values()]
        report["resource_screen"] = {
            "recovery_detection": "recovery_detected" if recovery else "no_recovery_detected",
            "swap_fallback_detected": swap,
            "limitations": "No recovery detected is not proof of unchanged internal scheduling; logs are not a complete trace. Allocator headroom is not total VRAM headroom.",
        }
        report["resource_eligible"] = (
            suite == "feasibility" and not recovery and not swap
            and all(item.get("allocator_headroom_passed", False) for item in operations)
        )
        if not report["numerically_eligible"] or (suite == "feasibility" and not report["resource_eligible"]):
            raise RuntimeError("TabICL smoke acceptance criteria failed; inspect saved numerical/resource diagnostics. No automatic fallback.")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error={
            "type": type(exc).__name__, "message": str(exc),
            "notes": list(getattr(exc, "__notes__", [])), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        original = sys.exception()
        try:
            if adapter is not None:
                report["native_diagnostics"] = adapter.diagnostic_metadata()
            report.update(ended_at_utc=utc_now(), duration_seconds=time.perf_counter() - started)
            write_json(path, report)
        except Exception as exc:
            if original is None:
                raise
            LOGGER.exception("Could not persist smoke failure diagnostics.")
            original.add_note(f"Could not persist smoke diagnostics: {exc}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TabICLv2 explicit none/FP32 profile smoke check: TRAIN/VAL ONLY, no test scores. "
            "Requires a verified local checkpoint; never downloads. Native batch_size counts "
            "ensemble members; --batch-size counts external query rows. No recovery detected "
            "does not prove fixed internal scheduling."
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--suite", choices=("equivalence", "feasibility"), default="equivalence")
    parser.add_argument("--sample-size", type=int, choices=(512, 2000))
    parser.add_argument("--query-size", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reload-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.reload_request is not None:
        _reload_worker(args.reload_request)
        return
    if args.config is None or args.output_dir is None:
        parser.error("--config and --output-dir are required.")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1 before running smoke checks.")
    size = args.sample_size if args.sample_size is not None else (2000 if args.suite == "feasibility" else 512)
    queries = args.query_size if args.query_size is not None else (257 if size == 2000 else 65)
    output = run(
        load_config(args.config), output_dir=args.output_dir, sample_size=size,
        query_size=queries, batch_size=args.batch_size, suite=args.suite,
    )
    print(f"Smoke output: {output}. M07 remains pending real n2000 feasibility for both domains.")


if __name__ == "__main__":
    main()
