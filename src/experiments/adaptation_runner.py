"""List, validate, or execute one Candidate A adaptation run."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import data
from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments.identity import identity_fingerprints
from src.paths import PROJECT_ROOT, RESULTS_DIR

LOGGER = logging.getLogger(__name__)
TASK_TRAINED = {"catboost", "ft_transformer"}


def _subset(doc: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    matches = [s for s in doc["subsets"] if matrix.subset_key(s) == ref]
    if len(matches) != 1:
        raise ValueError("Manifest subset reference does not resolve uniquely.")
    return matches[0]


def _assignments(domain: str) -> pd.DataFrame:
    identity = "seqn" if domain == "nhanes" else "row_index"
    frame = pd.read_csv(data.SPLIT_ASSIGNMENT_PATHS[domain], usecols=["row_index", identity, "split"])
    indices = manifest._identities(frame["row_index"], f"{domain} row indices")
    manifest._identities(frame[identity], domain)
    if not np.array_equal(indices, np.arange(len(indices))) or set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError(f"{domain}: invalid canonical split assignments.")
    return frame


def _read_rows(domain: str, split: str, identities: list[int], schema: data.DatasetSchema) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Parse only the requested rows; provenance is kept outside predictor columns."""
    identity = "seqn" if domain == "nhanes" else "row_index"
    if not identities or identities != sorted(set(identities)):
        raise ValueError("Requested identities must be nonempty, unique, and sorted.")
    assignments = _assignments(domain)
    indexed = assignments.set_index(identity)
    if not set(identities) <= set(indexed.index):
        raise ValueError(f"{domain}/{split}: unknown participant identity.")
    selected = indexed.loc[identities]
    if not selected["split"].eq(split).all():
        raise ValueError(f"{domain}/{split}: identities outside the canonical partition.")
    positions = set(int(x) for x in selected["row_index"]) if domain == "nhanes" else set(identities)
    path = data.NHANES_COMMON_PATHS["strict_common_16"] if domain == "nhanes" else data.COMMON_PATHS[domain]
    expected = ["seqn", *schema.features, schema.target] if domain == "nhanes" else list(data.load_schema().model_columns)
    if pd.read_csv(path, nrows=0).columns.tolist() != expected:
        raise ValueError(f"{domain}: canonical feature schema/order mismatch.")

    def skip(line: int) -> bool:
        return line != 0 and line - 1 not in positions

    columns = [*schema.features, schema.target, *(["seqn"] if domain == "nhanes" else [])]
    frame = pd.read_csv(path, usecols=columns, skiprows=skip)
    positions_sorted = sorted(positions)
    if len(frame) != len(positions_sorted):
        raise ValueError(f"{domain}/{split}: missing canonical rows.")
    ordered_assignments = assignments.iloc[positions_sorted]
    if domain == "nhanes" and not np.array_equal(frame["seqn"].to_numpy(), ordered_assignments["seqn"].to_numpy()):
        raise ValueError("NHANES canonical row identities disagree with split assignments.")
    frame.insert(0, "_identity", ordered_assignments[identity].to_numpy())
    frame = frame.set_index("_identity")
    if not frame.index.is_unique:
        raise ValueError(f"{domain}/{split}: duplicate canonical row identities.")
    frame = frame.loc[identities].reset_index(drop=True)
    X = frame.loc[:, list(schema.features)].copy()
    y = frame[schema.target].copy()
    if X.isna().any().any():
        raise ValueError(f"{domain}/{split}: missing predictor values.")
    from src.experiments.metrics import binary_labels
    labels = binary_labels(y)
    provenance = pd.DataFrame({"domain": domain, "partition": split, "identity": identities})
    return X, pd.Series(labels, name=schema.target), provenance


def _load_subset(doc: dict[str, Any], ref: dict[str, Any], schema: data.DatasetSchema) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    subset = _subset(doc, ref)
    ids = subset["participant_ids"]
    hashes = identity_fingerprints(subset["domain"], subset["identity_column"], ids)
    if any(hashes[key] != subset[key] for key in hashes) or len(ids) != subset["subset_size"]:
        raise ValueError("Subset identity fingerprint or size mismatch.")
    X, y, provenance = _read_rows(subset["domain"], subset["canonical_partition"], ids, schema)
    counts = {str(label): int((y == label).sum()) for label in (0, 1)}
    if counts != subset["class_counts"]:
        raise ValueError("Subset class counts do not match manifest.")
    return X, y, provenance


def _whole_partition(domain: str, split: str, schema: data.DatasetSchema) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    assignments = _assignments(domain)
    column = "seqn" if domain == "nhanes" else "row_index"
    ids = sorted(int(value) for value in assignments.loc[assignments["split"].eq(split), column])
    return _read_rows(domain, split, ids, schema)


def _disjoint(*frames: pd.DataFrame) -> None:
    seen: set[tuple[str, int]] = set()
    for frame in frames:
        current = set(zip(frame["domain"], frame["identity"]))
        if len(current) != len(frame) or seen.intersection(current):
            raise ValueError("Fitting, selection and query identities must be unique and disjoint within each domain.")
        seen.update(current)


def assemble(doc: dict[str, Any], spec: dict[str, Any], mode: str, query_size: int, *,
             freeze_path: Path | None = None, runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if mode == "research":
        if runs is None:
            raise ValueError("Research assembly requires an approved final freeze and validated matrix.")
        check_freeze(freeze_path, runs, spec["run_id"])
    elif mode != "development":
        raise ValueError("Unsupported execution mode.")
    schema = data.load_feature_schema("strict_common_16")
    if {"domain", "row_index", "seqn", "id", schema.target}.intersection(schema.features):
        raise ValueError("Identity or target columns cannot be predictors.")
    fit_refs = [ref for ref in spec["subsets"] if ref["role"] != "validation"]
    parts = [_load_subset(doc, ref, schema) for ref in fit_refs]
    X = pd.concat([part[0] for part in parts], ignore_index=True)
    y = pd.concat([part[1] for part in parts], ignore_index=True)
    provenance = pd.concat([part[2] for part in parts], ignore_index=True)
    reserved = next((ref for ref in spec["subsets"] if ref["role"] == "validation"), None)
    empty_X = pd.DataFrame(columns=schema.features)
    empty_y = pd.Series(dtype=int, name=schema.target)
    empty_p = pd.DataFrame(columns=["domain", "partition", "identity"])
    selection = (empty_X, empty_y, empty_p)
    if spec["model"] in TASK_TRAINED:
        if reserved is not None:
            selection = _load_subset(doc, reserved, schema)
        else:
            selection = _whole_partition(spec["source"], "val", schema)
    if mode == "research":
        query = _whole_partition("nhanes", "test", schema)
    elif mode == "development":
        if type(query_size) is not int or query_size < 2:
            raise ValueError("Development query size must be an integer of at least two.")
        assignments = _assignments("nhanes")
        excluded = set(_subset(doc, reserved)["participant_ids"]) if reserved else set()
        available = sorted(int(v) for v in assignments.loc[assignments["split"].eq("val"), "seqn"] if v not in excluded)
        if len(available) < query_size:
            raise ValueError("Insufficient disjoint NHANES development query rows.")
        query = _read_rows("nhanes", "val", available[:query_size], schema)
    _disjoint(provenance, selection[2], query[2])
    b = spec["budget"]
    real_fit = int((provenance["domain"] == "nhanes").sum())
    real_selection = int((selection[2]["domain"] == "nhanes").sum())
    accounting = {
        "nhanes_available": b, "nhanes_fit_or_context_used": real_fit,
        "nhanes_selection_used": real_selection,
        "nhanes_reserved_unused": b - real_fit - real_selection,
        "synthetic_fit_used": int((provenance["domain"] != "nhanes").sum()),
        "synthetic_selection_used": int((selection[2]["domain"] != "nhanes").sum()),
        "query_labels_for_metrics": len(query[1]),
        "query_label_category": "development_only" if mode == "development" else "research_evaluation",
    }
    if accounting["nhanes_reserved_unused"] < 0:
        raise ValueError("NHANES label usage exceeds the declared budget.")
    return {"schema": schema, "fit": (X, y, provenance), "selection": selection, "query": query, "label_accounting": accounting}


def check_freeze(path: Path | None, runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    if path is None:
        raise ValueError("Research mode requires an explicit approved final protocol-freeze record.")
    freeze = json.loads(path.read_text())
    protocol_hash = manifest._sha256_file(PROJECT_ROOT / "docs/plans/x01_adaptation_protocol.md")
    evaluation_hash = manifest._sha256_file(PROJECT_ROOT / "docs/plans/e05_evaluation_protocol.md")
    budgets = freeze.get("approved_budgets")
    if not isinstance(budgets, list) or not budgets or any(type(b) is not int or b not in manifest.BUDGETS for b in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Final protocol-freeze must specify a nonempty unique approved budget grid.")
    expected_ids = {r["run_id"] for r in runs if r["budget"] == 0 or r["budget"] in budgets}
    approved_ids = freeze.get("approved_run_ids")
    if (
        freeze.get("schema_version") != "x02_freeze_v1" or freeze.get("status") != "approved"
        or freeze.get("protocol_version") != matrix.PROTOCOL or freeze.get("protocol_sha256") != protocol_hash
        or freeze.get("evaluation_protocol_sha256") != evaluation_hash
        or freeze.get("matrix_sha256") != matrix.matrix_sha256(runs)
        or not isinstance(freeze.get("approved_by"), str) or not freeze["approved_by"].strip()
        or not isinstance(freeze.get("approved_at"), str) or not freeze["approved_at"].strip()
        or not isinstance(freeze.get("feasibility_decision"), str) or not freeze["feasibility_decision"].strip()
        or not isinstance(freeze.get("feasibility_evidence"), str) or not freeze["feasibility_evidence"].strip()
        or not isinstance(freeze.get("approval_reference"), str) or not freeze["approval_reference"].strip()
        or not isinstance(approved_ids, list) or any(not isinstance(item, str) for item in approved_ids)
        or len(set(approved_ids)) != len(approved_ids) or set(approved_ids) != expected_ids
        or run_id not in approved_ids
    ):
        raise ValueError("Final protocol-freeze record is incomplete, stale, or does not approve this run.")
    return freeze


def _output_path(root: Path, mode: str, run_id: str) -> Path:
    allowed = (RESULTS_DIR / "adaptation" / "x01_v1" / mode).resolve()
    resolved = root.resolve()
    if resolved != allowed and not resolved.is_relative_to(allowed):
        raise ValueError(f"{mode} output root must be inside {allowed}.")
    return resolved / run_id


def _configuration(spec: dict[str, Any], device: str | None, checkpoint: str | None, cap: int | None) -> dict[str, Any]:
    profile = spec["profile"]
    params = dict(profile["model_params"])
    model = spec["model"]
    if device:
        if model == "catboost":
            raise ValueError("CatBoost uses the fixed CPU profile; device override is unsupported.")
        params["device"] = device
    if checkpoint:
        if model not in matrix.CHECKPOINTS:
            raise ValueError("Checkpoint overrides apply only to frozen foundation models.")
        params["model_path"] = checkpoint
    if cap is not None:
        if type(cap) is not int or cap <= 0 or model not in TASK_TRAINED:
            raise ValueError("Training cap must be a positive integer for a task-trained model.")
        name = "iterations" if model == "catboost" else "max_epochs"
        if cap > params[name]:
            raise ValueError("Development training cap cannot exceed the approved profile.")
        params[name] = cap
    return {"model_params": params, "prediction_batch_size": profile["prediction_batch_size"],
            "early_stopping_rounds": profile["early_stopping_rounds"], "development_training_cap": cap}


def _verify_checkpoint(model: str, params: dict[str, Any]) -> None:
    if model in matrix.CHECKPOINTS:
        path = params.get("model_path")
        if not path:
            raise ValueError("An explicit local checkpoint path is required for foundation models.")
        checkpoint = Path(path).expanduser().resolve()
        if manifest._sha256_file(checkpoint) != matrix.CHECKPOINTS[model]:
            raise ValueError(f"{model} checkpoint differs from the approved reference SHA-256.")


def _adapter(spec: dict[str, Any], schema: data.DatasetSchema, config: dict[str, Any]) -> Any:
    params = config["model_params"]
    model = spec["model"]
    _verify_checkpoint(model, params)
    if model == "catboost":
        from src.models.catboost import CatBoostAdapter
        return CatBoostAdapter(schema.features, schema.categorical_features, schema.target, seed=spec["root_seed"],
                               model_params=params, early_stopping_rounds=config["early_stopping_rounds"])
    if model == "ft_transformer":
        from src.models.ft_transformer import FTTransformerAdapter
        return FTTransformerAdapter(schema.features, schema.categorical_features, schema.target, seed=spec["root_seed"],
                                    model_params=params, prediction_batch_size=config["prediction_batch_size"])
    if model == "tabpfn":
        from src.models.tabpfn import TabPFNAdapter
        return TabPFNAdapter(schema.features, schema.categorical_features, schema.target, seed=spec["root_seed"], model_params=params)
    if model == "tabicl":
        from src.models.tabicl import TabICLAdapter
        return TabICLAdapter(schema.features, schema.categorical_features, schema.target, seed=spec["root_seed"], model_params=params)
    raise ValueError(f"Unsupported model: {model}")


def execute(manifest_dir: Path, run_id: str, mode: str, output_root: Path, *, query_size: int = 64,
            device: str | None = None, checkpoint: str | None = None, training_cap: int | None = None,
            freeze_path: Path | None = None) -> Path:
    if mode not in ("development", "research"):
        raise ValueError("Mode must explicitly be development or research.")
    if mode == "research" and training_cap is not None:
        raise ValueError("Reduced training caps are only permitted in development mode.")
    doc, runs = matrix.load(manifest_dir)
    spec = matrix.select(runs, run_id)
    freeze = check_freeze(freeze_path, runs, run_id) if mode == "research" else None
    config = _configuration(spec, device, checkpoint, training_cap)
    output = _output_path(output_root, mode, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    from src.experiments.results import write_json
    from src.experiments.runner import environment_metadata, utc_now
    started = time.perf_counter()
    status: dict[str, Any] = {
        "status": "running", "mode": mode, "research_comparison_eligible": mode == "research",
        "started_at_utc": utc_now(), "run_id": run_id, "specification": spec,
        "matrix_sha256": matrix.matrix_sha256(runs), "effective_configuration": config,
        "freeze": freeze, "query_size_requested": query_size if mode == "development" else None,
        "operation_seconds": {},
    }
    try:
        write_json(output / "metadata.json", status)
        write_json(output / "run_specification.json", spec)
        status["code_revision"] = manifest._revision()
        status["runner_sha256"] = manifest._sha256_file(Path(__file__))
        status["matrix_code_sha256"] = manifest._sha256_file(Path(matrix.__file__))
        adapter_module = {"catboost": "catboost", "ft_transformer": "ft_transformer",
                          "tabpfn": "tabpfn", "tabicl": "tabicl"}[spec["model"]]
        code_paths = [
            PROJECT_ROOT / "src/models" / f"{adapter_module}.py",
            PROJECT_ROOT / "src/models/base.py", PROJECT_ROOT / "src/experiments/config.py",
            PROJECT_ROOT / "src/experiments/metrics.py", PROJECT_ROOT / "src/experiments/results.py",
        ]
        if spec["model"] == "ft_transformer":
            code_paths.append(PROJECT_ROOT / "src/models/ft_transformer_config.py")
        status["execution_code_sha256"] = {
            str(path.relative_to(PROJECT_ROOT)): manifest._sha256_file(path) for path in code_paths
        }
        status["environment"] = environment_metadata(spec["model"])
        status["host"] = {"platform": platform.platform(), "machine": platform.machine()}
        before = manifest._file_fingerprints()
        manifest_metadata = json.loads((manifest_dir / "metadata.json").read_text())
        if before != manifest_metadata["source_file_sha256"]:
            raise ValueError("Canonical inputs changed since manifest validation.")
        tick = time.perf_counter()
        assembled = assemble(doc, spec, mode, query_size, freeze_path=freeze_path, runs=runs)
        if before != manifest._file_fingerprints():
            raise ValueError("Canonical inputs changed during assembly.")
        status["operation_seconds"]["assembly"] = time.perf_counter() - tick
        status["label_accounting"] = assembled["label_accounting"]
        status["feature_order"] = list(assembled["schema"].features)
        status["prepared_file_sha256"] = before
        for role in ("fit", "selection", "query"):
            assembled[role][2].to_csv(output / f"{role}_identities.csv", index=False)
        write_json(output / "metadata.json", status)
        random.seed(spec["root_seed"])
        np.random.seed(spec["root_seed"])
        adapter = _adapter(spec, assembled["schema"], config)
        if spec["model"] == "tabicl" and hasattr(adapter, "set_diagnostic_path"):
            adapter.set_diagnostic_path(output / "native_diagnostics.log")
        adapter.reset_peak_memory()
        adapter.synchronize()
        tick = time.perf_counter()
        adapter.fit(assembled["fit"][0], assembled["fit"][1], assembled["selection"][0], assembled["selection"][1])
        adapter.synchronize()
        status["operation_seconds"]["fit"] = time.perf_counter() - tick
        tick = time.perf_counter()
        adapter.save(output / adapter.artifact_name)
        if not (output / adapter.artifact_name).is_file():
            raise RuntimeError("Adapter did not persist its required fitted artifact.")
        _verify_checkpoint(spec["model"], config["model_params"])
        status["operation_seconds"]["save"] = time.perf_counter() - tick
        status["model_artifact"] = adapter.artifact_name
        status["model"] = adapter.metadata()
        status["validation_population"] = (
            "budgeted NHANES validation" if spec["budget"] and spec["model"] in TASK_TRAINED else
            "complete same-source validation" if spec["model"] in TASK_TRAINED else "unused"
        )
        status["adapter_metadata_note"] = (
            "The adapter's historical same-domain validation wording is generic; "
            "validation_population and selection_identities.csv describe this run."
        )
        adapter.synchronize()
        tick = time.perf_counter()
        probabilities = adapter.predict_proba_batched(assembled["query"][0], config["prediction_batch_size"])
        adapter.synchronize()
        status["operation_seconds"]["predict"] = time.perf_counter() - tick
        from src.experiments.metrics import evaluate_binary
        tick = time.perf_counter()
        metrics = evaluate_binary(assembled["query"][1], probabilities, 0.5, 10)
        predictions = assembled["query"][2].copy()
        predictions["y_true"] = assembled["query"][1].to_numpy()
        predictions["y_probability"] = probabilities
        predictions["y_pred"] = (np.asarray(probabilities) >= 0.5).astype(int)
        predictions.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        write_json(output / "metrics.json", {"mode": mode, "research_comparison_eligible": mode == "research", "metrics": metrics})
        status["operation_seconds"]["metrics_and_predictions"] = time.perf_counter() - tick
        status["resources"] = adapter.resource_metadata()
        status["output_sha256"] = {
            path.name: manifest._sha256_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name not in ("metadata.json", "metadata.json.tmp")
        }
        status.update(status="completed", ended_at_utc=utc_now(), total_duration_seconds=time.perf_counter() - started)
        write_json(output / "metadata.json", status)
    except BaseException as exc:
        status.update(status="failed", ended_at_utc=utc_now(), total_duration_seconds=time.perf_counter() - started,
                      error={"type": type(exc).__name__, "message": str(exc)})
        try:
            write_json(output / "metadata.json", status)
        except (OSError, ValueError, TypeError):
            LOGGER.exception("Could not write failure status for %s", output)
        raise
    return output


def _safe_filename(value: object) -> bool:
    return (
        isinstance(value, str) and value not in ("", ".", "..")
        and "/" not in value and "\\" not in value and "\0" not in value
    )


def _inventory_status(output: Path, spec: dict[str, Any], mode: str) -> tuple[str, str | None]:
    if not output.exists():
        return "missing", None
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return "incomplete", "metadata.json is missing."
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return "incomplete", f"Cannot read valid metadata.json ({type(exc).__name__})."
    if not isinstance(metadata, dict):
        return "incompatible", "metadata.json must be an object."
    state = metadata.get("status", "incomplete")
    if not isinstance(state, str):
        return "incompatible", "Metadata status must be a string."
    if metadata.get("run_id") != spec["run_id"] or metadata.get("mode") != mode or metadata.get("specification") != spec:
        return "incompatible", "Run identity, mode, or specification does not match."
    if "model_artifact" in metadata and not _safe_filename(metadata["model_artifact"]):
        return "incompatible", "model_artifact must be a safe filename."
    if "research_comparison_eligible" in metadata and type(metadata["research_comparison_eligible"]) is not bool:
        return "incompatible", "research_comparison_eligible must be a boolean."
    if "output_sha256" in metadata:
        hashes = metadata["output_sha256"]
        if not isinstance(hashes, dict) or any(
            not _safe_filename(name) or not isinstance(digest, str)
            for name, digest in hashes.items()
        ):
            return "incompatible", "output_sha256 must map safe filenames to string digests."
    if state not in ("running", "completed", "failed"):
        return "incomplete", "Run status is missing or unrecognized."
    if state != "completed":
        return state, None
    if "model_artifact" not in metadata:
        return "incomplete", "model_artifact is missing."
    required = {
        "run_specification.json", "fit_identities.csv", "selection_identities.csv",
        "query_identities.csv", "predictions.csv.gz", "metrics.json", metadata["model_artifact"],
    }
    if not all((output / name).is_file() for name in required):
        return "incomplete", "A required run artifact is missing."
    hashes = metadata.get("output_sha256")
    if (
        not isinstance(hashes, dict) or not required <= set(hashes)
        or metadata.get("research_comparison_eligible") is not (mode == "research")
    ):
        return "incompatible", "Artifact hashes or research eligibility are invalid."
    for name, digest in hashes.items():
        if not _safe_filename(name) or not isinstance(digest, str):
            return "incompatible", "Artifact hash entry has an invalid filename or digest."
        try:
            if not (output / name).is_file() or manifest._sha256_file(output / name) != digest:
                return "incompatible", "An artifact is missing or its fingerprint differs."
        except OSError as exc:
            return "incomplete", f"Cannot read an artifact ({type(exc).__name__})."
    return "completed", None


def inventory(runs: list[dict[str, Any]], output_root: Path, mode: str) -> list[dict[str, str]]:
    entries = []
    for spec in runs:
        output = _output_path(output_root, mode, spec["run_id"])
        state, reason = _inventory_status(output, spec, mode)
        entry = {"run_id": spec["run_id"], "status": state}
        if reason is not None:
            entry["reason"] = reason
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("matrix", "dry", "execute", "inventory"))
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("development", "research"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--training-cap", type=int)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.action == "execute":
        if not args.run_id or not args.mode or not args.output_root:
            parser.error("execute requires --run-id, --mode and --output-root")
        print(execute(args.manifest_dir, args.run_id, args.mode, args.output_root, query_size=args.query_size,
                      device=args.device, checkpoint=args.checkpoint, training_cap=args.training_cap, freeze_path=args.freeze))
        return
    doc, runs = matrix.load(args.manifest_dir)
    if args.action == "matrix":
        result: Any = {"schema_version": matrix.VERSION, "matrix_sha256": matrix.matrix_sha256(runs), "runs": runs}
    elif args.action == "dry":
        if not args.run_id:
            parser.error("dry requires --run-id")
        spec = matrix.select(runs, args.run_id)
        before = manifest._file_fingerprints()
        if before != json.loads((args.manifest_dir / "metadata.json").read_text())["source_file_sha256"]:
            raise ValueError("Canonical inputs changed since manifest validation.")
        assembled = assemble(doc, spec, "development", args.query_size)
        if before != manifest._file_fingerprints():
            raise ValueError("Canonical inputs changed during dry validation.")
        result = {"specification": spec, "label_accounting": assembled["label_accounting"],
                  "fit_rows": len(assembled["fit"][0]), "selection_rows": len(assembled["selection"][0]),
                  "development_query_rows": len(assembled["query"][0])}
    else:
        if not args.mode or not args.output_root:
            parser.error("inventory requires --mode and --output-root")
        result = inventory(runs, args.output_root, args.mode)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.export:
        with args.export.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
