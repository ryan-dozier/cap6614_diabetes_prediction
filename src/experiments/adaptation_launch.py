"""Plan, inventory, resume, archive, and execute approved Candidate A model jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from src.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
SCHEMA = "x03_launch_plan_v2"
BLOCKING_STATES = ("failed", "incomplete", "incompatible")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(manifest._json_bytes(value)).hexdigest()


def outside_runs(path: Path, run_root: Path) -> None:
    if path == run_root or path.is_relative_to(run_root) or run_root.is_relative_to(path):
        raise ValueError(f"Plans and job evidence must be separate from the run collection: {path}")


def _code_paths(project: Path) -> list[Path]:
    return sorted([
        *project.glob("src/experiments/*.py"),
        *project.glob("src/models/*.py"),
        project / "src/data.py", project / "src/paths.py", project / "src/domain_labels.py",
        project / "scripts/x03_newton_common.sh",
        project / "scripts/x03_newton_cpu.sbatch", project / "scripts/x03_newton_gpu.sbatch",
    ])


def _fingerprints(project: Path, manifest_dir: Path, freeze_path: Path,
                  checkpoints: dict[str, str]) -> dict[str, str]:
    paths = [
        *_code_paths(project), *manifest._input_paths(),
        manifest_dir / "assignments.json", manifest_dir / "metadata.json", freeze_path,
        project / "docs/plans/x01_adaptation_protocol.md",
        project / "docs/plans/e05_evaluation_protocol.md",
        project / "docs/results/x02_stage2e_feasibility.md",
        *(project / "configs" / name for name, _ in matrix.PROFILES.values()),
        *(Path(path) for path in checkpoints.values()),
    ]
    return {str(path): manifest._sha256_file(path) for path in sorted(set(paths))}


def _lazy_test_ids():
    """Ordered canonical NHANES test identities, loaded at most once; IDs only."""
    cache: list[list[int]] = []

    def load() -> list[int]:
        if not cache:
            assignments = runner._assignments("nhanes")
            cache.append(sorted(int(x) for x in assignments.loc[assignments["split"].eq("test"), "seqn"]))
        return cache[0]

    return load


def _read_provenance(path: Path) -> pd.DataFrame:
    # Mirrors src.reporting.adaptation._read_provenance; kept separate so planning
    # never imports that module's metric dependencies. Parity is tested.
    frame = pd.read_csv(path)
    if list(frame.columns) != ["domain", "partition", "identity"]:
        raise ValueError(f"{path.name}: invalid provenance columns")
    if len(frame):
        if frame["identity"].isna().any() or frame["identity"].dtype.kind not in "iu":
            raise ValueError(f"{path.name}: identities must be nonmissing integers")
        if frame.duplicated(["domain", "identity"]).any():
            raise ValueError(f"{path.name}: duplicate domain-qualified identity")
    return frame


def _expected_provenance(doc: dict[str, Any], spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Mirrors src.reporting.adaptation._expected_provenance; parity is tested.
    columns = ["domain", "partition", "identity"]
    fit: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for ref in spec["subsets"]:
        subset = runner._subset(doc, ref)
        rows = [{"domain": ref["domain"], "partition": ref["canonical_partition"], "identity": i}
                for i in subset["participant_ids"]]
        if ref["role"] == "validation":
            if spec["model"] in runner.TASK_TRAINED:
                selection.extend(rows)
        else:
            fit.extend(rows)
    if spec["condition"] == "synthetic_only" and spec["model"] in runner.TASK_TRAINED:
        assignments = runner._assignments(spec["source"])
        ids = sorted(int(x) for x in assignments.loc[assignments["split"].eq("val"), "row_index"])
        selection = [{"domain": spec["source"], "partition": "val", "identity": i} for i in ids]
    return pd.DataFrame(fit, columns=columns), pd.DataFrame(selection, columns=columns)


def _verify_completed(output: Path, spec: dict[str, Any], doc: dict[str, Any],
                      runs: list[dict[str, Any]], freeze: dict[str, Any],
                      source_hashes: dict[str, str], test_ids: list[int]) -> None:
    """Research-contract checks for one completed run, excluding prediction parsing.

    Mirrors src.reporting.adaptation._validate_completed except for saved prediction
    values: their bytes are already covered by the verified recorded hashes and their
    contents are revalidated by the final reporting gate, so resume decisions never
    read predictive outcomes.
    """
    meta = json.loads((output / "metadata.json").read_text())
    if meta.get("matrix_sha256") != matrix.matrix_sha256(runs):
        raise ValueError("matrix fingerprint differs from the validated matrix")
    if meta.get("freeze") != freeze:
        raise ValueError("recorded freeze differs from the approved freeze record")
    if meta.get("prepared_file_sha256") != source_hashes:
        raise ValueError("canonical source/split fingerprints differ")
    if json.loads((output / "run_specification.json").read_text()) != spec:
        raise ValueError("run specification artifact differs from approved matrix")
    config = meta.get("effective_configuration")
    if not isinstance(config, dict) or not isinstance(config.get("model_params"), dict):
        raise ValueError("effective configuration and model parameters must be objects")
    params = config["model_params"]
    device = params.get("device") if spec["model"] != "catboost" else None
    checkpoint = params.get("model_path") if spec["model"] in matrix.CHECKPOINTS else None
    if spec["model"] != "catboost" and (not isinstance(device, str) or not device.strip()):
        raise ValueError("recorded device must be a nonempty string")
    if spec["model"] in matrix.CHECKPOINTS and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("recorded checkpoint path must be a nonempty string")
    if config != runner._configuration(spec, device, checkpoint, None):
        raise ValueError("effective scientific configuration differs from approved profile")
    if spec["model"] in matrix.CHECKPOINTS:
        recorded = meta.get("model")
        pin = recorded.get("checkpoint") if isinstance(recorded, dict) else None
        if not isinstance(pin, dict) or pin.get("sha256") != matrix.CHECKPOINTS[spec["model"]]:
            raise ValueError("recorded checkpoint fingerprint differs from approved checkpoint")
    schema = runner.data.load_feature_schema("strict_common_16")
    if {"domain", "row_index", "seqn", "id", schema.target}.intersection(schema.features):
        raise ValueError("identity or target columns cannot be model features")
    if meta.get("feature_order") != list(schema.features):
        raise ValueError("feature order differs from approved schema")
    fit, selection = _expected_provenance(doc, spec)
    for role, expected in (("fit", fit), ("selection", selection)):
        actual = _read_provenance(output / f"{role}_identities.csv")
        if actual.to_dict("records") != expected.to_dict("records"):
            raise ValueError(f"{role} identities differ from manifest assignments")
    query = _read_provenance(output / "query_identities.csv")
    expected_query = pd.DataFrame({"domain": "nhanes", "partition": "test", "identity": test_ids})
    if query.to_dict("records") != expected_query.to_dict("records"):
        raise ValueError("query provenance differs from ordered canonical test identities")
    runner._disjoint(fit, selection, query)
    real_fit = int((fit["domain"] == "nhanes").sum())
    real_selection = int((selection["domain"] == "nhanes").sum())
    accounting = {
        "nhanes_available": spec["budget"], "nhanes_fit_or_context_used": real_fit,
        "nhanes_selection_used": real_selection,
        "nhanes_reserved_unused": spec["budget"] - real_fit - real_selection,
        "synthetic_fit_used": len(fit) - real_fit,
        "synthetic_selection_used": len(selection) - real_selection,
        "query_labels_for_metrics": len(test_ids), "query_label_category": "research_evaluation",
    }
    if meta.get("label_accounting") != accounting or accounting["nhanes_reserved_unused"] < 0:
        raise ValueError("label accounting differs from assigned identities")


def _classify(output: Path, spec: dict[str, Any], doc: dict[str, Any], runs: list[dict[str, Any]],
              freeze: dict[str, Any], source_hashes: dict[str, str], test_ids) -> tuple[str, str | None]:
    """Classify one expected research output location without reading predictions."""
    state, reason = runner._inventory_status(output, spec, "research")
    if state == "missing":
        return "missing", None
    if state == "running":
        return "incomplete", ("Recorded status is running; confirm the owning job has ended, "
                              "then archive this output explicitly before retrying.")
    if state == "failed":
        try:
            error = json.loads((output / "metadata.json").read_text()).get("error")
        except (OSError, UnicodeError, json.JSONDecodeError):
            error = None
        message = error.get("message") if isinstance(error, dict) else None
        return "failed", (f"Recorded failure: {message}" if message
                          else "Recorded failure without a readable error message.")
    if state != "completed":
        return state, reason
    try:
        _verify_completed(output, spec, doc, runs, freeze, source_hashes, test_ids())
    except (ValueError, KeyError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return "incompatible", f"{type(exc).__name__}: {exc}"
    return "verified_complete", None


def make_plan(*, project: Path, python: Path, manifest_dir: Path, freeze_path: Path,
              run_root: Path, model: str = "all", resume: bool = False,
              tabpfn_checkpoint: Path | None = None,
              tabicl_checkpoint: Path | None = None) -> dict[str, Any]:
    project = project.expanduser().resolve()
    if project != PROJECT_ROOT.resolve():
        raise ValueError("Run this module from the requested project's Python environment and source tree.")
    # Keep the environment executable's path: resolving a venv symlink changes its environment.
    python = python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python must be an executable file: {python}")
    if model not in (*manifest.MODELS, "all"):
        raise ValueError(f"Unsupported model: {model}")
    manifest_dir = manifest_dir.expanduser().resolve()
    freeze_path = freeze_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    runner._output_path(run_root, "research", "layout_check")
    if run_root.exists() and not run_root.is_dir():
        raise ValueError(f"Run root is not a directory: {run_root}")
    doc, runs = matrix.load(manifest_dir)
    freeze = runner.check_freeze(freeze_path, runs, runs[0]["run_id"])
    expected_profiles = {name: digest for name, digest in matrix.PROFILES.values()}
    if (
        sorted(freeze["approved_budgets"]) != sorted(manifest.BUDGETS)
        or len(runs) != 220 or len({run["run_id"] for run in runs}) != 220
        or freeze.get("manifest_sha256") != runs[0]["manifest_sha256"]
        or freeze.get("manifest_metadata_sha256") != manifest._sha256_file(manifest_dir / "metadata.json")
        or freeze.get("reference_profile_sha256") != expected_profiles
        or freeze.get("checkpoint_sha256") != matrix.CHECKPOINTS
        or freeze.get("feasibility_evidence") != "docs/results/x02_stage2e_feasibility.md"
        or freeze.get("feasibility_evidence_sha256") != manifest._sha256_file(
            project / "docs/results/x02_stage2e_feasibility.md"
        )
    ):
        raise ValueError("Freeze is incompatible with the approved full-grid manifest, profiles, or evidence.")
    source_hashes = json.loads((manifest_dir / "metadata.json").read_text())["source_file_sha256"]
    test_ids = _lazy_test_ids()
    counts = {}
    for name in manifest.MODELS:
        counts[name] = dict(Counter(run["condition"] for run in runs if run["model"] == name))
        if counts[name] != {"real_only": 15, "mixed": 30, "synthetic_only": 10}:
            raise ValueError(f"Unexpected grid counts for {name}: {counts[name]}")
    selected = [run for run in runs if model == "all" or run["model"] == model]
    if run_root.exists():
        unexpected = sorted(path.name for path in run_root.iterdir()
                            if path.name not in {run["run_id"] for run in runs})
        if unexpected:
            raise ValueError(f"Unexpected entries in the canonical run collection: {unexpected}")
    supplied = {"tabpfn": tabpfn_checkpoint, "tabicl": tabicl_checkpoint}
    checkpoints = {}
    for name in matrix.CHECKPOINTS:
        if any(run["model"] == name for run in selected):
            if supplied[name] is None:
                raise ValueError(f"An explicit {name} checkpoint location is required.")
            checkpoint = supplied[name].expanduser().resolve()
            runner._verify_checkpoint(name, {"model_path": str(checkpoint)})
            checkpoints[name] = str(checkpoint)
    entries = []
    for spec in selected:
        checkpoint = checkpoints.get(spec["model"])
        config = runner._configuration(spec, None, checkpoint, None)
        output = runner._output_path(run_root, "research", spec["run_id"])
        if output.resolve() != run_root / spec["run_id"]:
            raise ValueError(f"Run output escapes its canonical location: {output}")
        command = [
            str(python), "-u", "-m", "src.experiments.adaptation_runner", "execute",
            "--manifest-dir", str(manifest_dir), "--run-id", spec["run_id"],
            "--mode", "research", "--output-root", str(run_root), "--freeze", str(freeze_path),
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", checkpoint])
        entry = {
            "run_id": spec["run_id"], "specification": spec,
            "effective_configuration": config, "command": command, "output_path": str(output),
        }
        if resume:
            state, blocking = _classify(output, spec, doc, runs, freeze, source_hashes, test_ids)
            entry["output_status"] = "pending" if state == "missing" else state
            if state == "verified_complete":
                # Pin completed-run evidence so later changes invalidate this plan.
                entry["completed_metadata_sha256"] = manifest._sha256_file(output / "metadata.json")
            if state in BLOCKING_STATES:
                entry["blocking_reason"] = blocking
        else:
            entry["output_status"] = "existing" if os.path.lexists(output) else "pending"
        entries.append(entry)
    deployment = {
        "project": str(project), "python": str(python), "manifest_dir": str(manifest_dir),
        "freeze_path": str(freeze_path), "run_root": str(run_root),
        "tabpfn_checkpoint": checkpoints.get("tabpfn"), "tabicl_checkpoint": checkpoints.get("tabicl"),
    }
    plan = {
        "schema_version": SCHEMA, "model": model, "mode": "research", "resume": resume,
        "deployment": deployment, "matrix_sha256": matrix.matrix_sha256(runs),
        "freeze_sha256": manifest._sha256_file(freeze_path),
        "file_sha256": _fingerprints(project, manifest_dir, freeze_path, checkpoints),
        "grid_counts": counts, "runs": entries,
    }
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": plan["model"], "mode": plan["mode"], "resume": plan.get("resume", False),
        "run_count": len(plan["runs"]),
        "pending": sum(run["output_status"] == "pending" for run in plan["runs"]),
        "verified_complete": sum(run["output_status"] == "verified_complete" for run in plan["runs"]),
        "existing_run_ids": [run["run_id"] for run in plan["runs"] if run["output_status"] == "existing"],
        "blocked": {run["run_id"]: run.get("blocking_reason")
                    for run in plan["runs"] if run["output_status"] in BLOCKING_STATES},
        "grid_counts": plan["grid_counts"], "run_root": plan["deployment"]["run_root"],
        "plan_sha256": plan["plan_sha256"], "freeze_sha256": plan["freeze_sha256"],
    }


def validate_plan(plan: dict[str, Any], model: str) -> None:
    if plan.get("schema_version") != SCHEMA or plan.get("model") != model or model not in manifest.MODELS:
        raise ValueError("Execution requires a plan for exactly the submitted model.")
    resume = bool(plan.get("resume"))
    if not resume:
        existing = [run["run_id"] for run in plan["runs"] if os.path.lexists(run["output_path"])]
        if existing:
            raise FileExistsError(f"Existing outputs; generate an explicit resume plan or archive first: {', '.join(existing)}")
    deployment = {key: Path(value) if value is not None else None
                  for key, value in plan["deployment"].items()}
    current = make_plan(model=model, resume=resume, **deployment)
    if current != plan:
        raise ValueError("Deployment, scientific inputs, completed-run evidence, code, or commands "
                         "changed since planning; regenerate and review a fresh plan.")
    blocked = {run["run_id"]: run.get("blocking_reason")
               for run in plan["runs"] if run["output_status"] in BLOCKING_STATES}
    if blocked:
        raise ValueError("Blocked outputs require explicit archival or investigation before resume: "
                         + json.dumps(blocked, sort_keys=True))


def execute_plan(plan_path: Path, model: str, job_dir: Path) -> None:
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, model)
    project = Path(plan["deployment"]["project"])
    run_root = Path(plan["deployment"]["run_root"])
    job_dir = job_dir.expanduser().resolve()
    outside_runs(plan_path, run_root)
    outside_runs(job_dir, run_root)
    if Path(sys.executable).absolute() != Path(plan["deployment"]["python"]):
        raise ValueError("Execution Python differs from the validated deployment plan.")
    if not os.access("/usr/bin/time", os.X_OK):
        raise ValueError("/usr/bin/time is required for per-run resource logging.")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Execution requires a Slurm job ID and the approved four-CPU allocation.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "4":
            raise ValueError(f"Incorrect numerical thread limit: {name}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONNOUSERSITE"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Missing offline/environment setting: {name}")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise ValueError("PYTHONPATH/PYTHONHOME overrides are unsupported.")
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir()
    (job_dir / "plan.json").write_bytes(manifest._json_bytes(plan))
    pending = [run for run in plan["runs"] if run["output_status"] == "pending"]
    skipped = [run["run_id"] for run in plan["runs"] if run["output_status"] == "verified_complete"]
    status = {"status": "failed", "job_id": os.environ["SLURM_JOB_ID"],
              "model": model, "plan_sha256": plan["plan_sha256"],
              "freeze_sha256": plan["freeze_sha256"], "resume": bool(plan.get("resume")),
              "no_op": False, "skipped_verified_run_ids": skipped,
              "completed_run_ids": [], "invalid_run_ids": [],
              "remaining_run_ids": [run["run_id"] for run in pending], "current_run_id": None}
    if not pending:
        # Explicit no-op: every selected run is already verified complete.
        status.update(status="completed", no_op=True)
        with (job_dir / "job.log").open("x") as log:
            log.write(f"No-op: all {len(skipped)} selected runs are verified complete; "
                      "no model command was invoked.\n")
        (job_dir / "status.json").write_bytes(manifest._json_bytes(status))
        return
    deployment = plan["deployment"]
    doc, runs = matrix.load(Path(deployment["manifest_dir"]))
    freeze = runner.check_freeze(Path(deployment["freeze_path"]), runs, runs[0]["run_id"])
    source_hashes = json.loads((Path(deployment["manifest_dir"]) / "metadata.json").read_text())["source_file_sha256"]
    test_ids = _lazy_test_ids()
    with (job_dir / "job.log").open("x") as log:
        def record(command: list[str]) -> None:
            log.write(f"Command: {json.dumps(command)}\n")
            log.flush()
            subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)

        try:
            environment = {name: value for name, value in os.environ.items()
                           if name.startswith("SLURM_") or name in (
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                               "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONNOUSERSITE",
                           )}
            log.write(json.dumps({"host": platform.node(), "environment": environment}, sort_keys=True) + "\n")
            for command in (
                ["git", "rev-parse", "HEAD"], ["git", "branch", "--show-current"],
                ["git", "status", "--short", "--untracked-files=all"],
                ["git", "diff", "--stat"], ["git", "diff", "--cached", "--stat"],
                ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]],
            ):
                record(command)
            packages = ["numpy", "pandas", "scikit-learn", {
                "catboost": "catboost", "ft_transformer": "rtdl-revisiting-models",
                "tabpfn": "tabpfn", "tabicl": "tabicl",
            }[model]]
            record([plan["deployment"]["python"], "-c",
                    "import importlib.metadata, platform; print(platform.python_version()); "
                    f"print({{name: importlib.metadata.version(name) for name in {packages!r}}})"])
            if model != "catboost":
                record(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"])
                record([plan["deployment"]["python"], "-c",
                        "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda); "
                        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                        "print(torch.cuda.get_device_name(0))"])
            for run in pending:
                if runner._output_path(run_root, "research", run["run_id"]) != Path(run["output_path"]):
                    raise ValueError("Resolved run-root layout changed since planning.")
                current_hashes = _fingerprints(
                    project, Path(plan["deployment"]["manifest_dir"]),
                    Path(plan["deployment"]["freeze_path"]),
                    {name: plan["deployment"][f"{name}_checkpoint"]
                     for name in matrix.CHECKPOINTS if plan["deployment"][f"{name}_checkpoint"]},
                )
                if current_hashes != plan["file_sha256"]:
                    raise ValueError("Validated plan inputs/code changed before execution.")
                if os.path.lexists(run["output_path"]):
                    raise FileExistsError(f"Output appeared after preflight: {run['run_id']}")
                log.write(f"Running: {run['run_id']}\n")
                log.flush()
                status["current_run_id"] = run["run_id"]
                (job_dir / f"{run['run_id']}.command.json").write_bytes(manifest._json_bytes(run["command"]))
                with (job_dir / f"{run['run_id']}.log").open("x") as run_log:
                    subprocess.run(
                        ["/usr/bin/time", "-v", "-o", str(job_dir / f"{run['run_id']}.time"),
                         *run["command"]],
                        cwd=project, stdout=run_log, stderr=subprocess.STDOUT, check=True,
                    )
                # A zero exit status is not success: verify the new artifacts.
                state, reason = _classify(Path(run["output_path"]), run["specification"],
                                          doc, runs, freeze, source_hashes, test_ids)
                if state != "verified_complete":
                    status["invalid_run_ids"].append({"run_id": run["run_id"],
                                                      "state": state, "reason": reason})
                    status["remaining_run_ids"].remove(run["run_id"])
                    raise ValueError(f"{run['run_id']} finished with exit status zero but failed "
                                     f"artifact verification ({state}): {reason}")
                status["completed_run_ids"].append(run["run_id"])
                status["remaining_run_ids"].remove(run["run_id"])
                status["current_run_id"] = None
            status["status"] = "completed"
        finally:
            (job_dir / "status.json").write_bytes(manifest._json_bytes(status))


def archive_runs(*, manifest_dir: Path, freeze_path: Path, run_root: Path, archive_dir: Path,
                 run_ids: list[str], reason: str, allow_incompatible: bool = False) -> list[dict[str, Any]]:
    """Move explicitly selected failed/incomplete outputs aside before a planned retry.

    Nothing is deleted or overwritten; verified-complete runs are refused, and
    incompatible/corrupted outputs stay blocked unless allow_incompatible records
    a deliberate, documented decision after manual investigation.
    """
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("Archival requires an explicit list of unique run IDs.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Archival requires a nonempty documented reason.")
    manifest_dir = manifest_dir.expanduser().resolve()
    freeze_path = freeze_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    runner._output_path(run_root, "research", "layout_check")
    archive_dir = archive_dir.expanduser().resolve()
    outside_runs(archive_dir, run_root)
    doc, runs = matrix.load(manifest_dir)
    freeze = runner.check_freeze(freeze_path, runs, runs[0]["run_id"])
    source_hashes = json.loads((manifest_dir / "metadata.json").read_text())["source_file_sha256"]
    test_ids = _lazy_test_ids()
    specs = {run["run_id"]: run for run in runs}
    unknown = sorted(set(run_ids) - set(specs))
    if unknown:
        raise ValueError(f"Run IDs outside the approved matrix: {', '.join(unknown)}")

    def check(run_id: str) -> tuple[Path, str, str | None]:
        output = runner._output_path(run_root, "research", run_id)
        if output != run_root / run_id:
            raise ValueError(f"Run output escapes its canonical location: {output}")
        if output.is_symlink():
            raise ValueError(f"Refusing to archive a symlink: {output}")
        state, state_reason = _classify(output, specs[run_id], doc, runs, freeze, source_hashes, test_ids)
        if state == "verified_complete":
            raise ValueError(f"{run_id} is verified complete and must never be archived.")
        if state == "missing":
            raise ValueError(f"{run_id} has no output to archive.")
        if state == "incompatible" and not allow_incompatible:
            raise ValueError(f"{run_id} is incompatible/corrupted ({state_reason}); it stays blocked "
                             "for manual investigation unless --allow-incompatible records a "
                             "deliberate decision.")
        destination = archive_dir / run_id
        record_path = archive_dir / f"{run_id}.archive.json"
        if os.path.lexists(destination) or os.path.lexists(record_path):
            raise FileExistsError(f"Archive destination already exists for {run_id}: {destination}")
        return output, state, state_reason

    for run_id in run_ids:
        check(run_id)
    archive_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for run_id in run_ids:
        # Recheck the selected state immediately before moving anything.
        output, state, state_reason = check(run_id)
        destination = archive_dir / run_id
        provenance: dict[str, Any] = {"available": False}
        try:
            meta = json.loads((output / "metadata.json").read_text())
            if isinstance(meta, dict):
                provenance = {"available": True,
                              **{key: meta.get(key) for key in
                                 ("run_id", "mode", "status", "error", "code_revision",
                                  "started_at_utc", "ended_at_utc")}}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            provenance = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        try:
            os.rename(output, destination)
        except OSError as exc:
            raise OSError(f"Could not move {output} to {destination}; the original is retained "
                          f"({exc}). Use an archive destination on the same filesystem.") from exc
        record = {"run_id": run_id, "original_path": str(output), "archive_path": str(destination),
                  "reason": reason, "state": state, "state_reason": state_reason,
                  "allow_incompatible": allow_incompatible,
                  "archived_at_utc": datetime.now(timezone.utc).isoformat(),
                  "provenance": provenance}
        with (archive_dir / f"{run_id}.archive.json").open("xb") as handle:
            handle.write(manifest._json_bytes(record))
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "inventory"):
        command = sub.add_parser(name)
        command.add_argument("--model", choices=(*manifest.MODELS, "all"), default="all")
        command.add_argument("--project", type=Path, default=PROJECT_ROOT)
        command.add_argument("--python", type=Path, default=Path(sys.executable))
        command.add_argument("--manifest-dir", type=Path, required=True)
        command.add_argument("--freeze", dest="freeze_path", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--tabpfn-checkpoint", type=Path)
        command.add_argument("--tabicl-checkpoint", type=Path)
        if name == "plan":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--export", type=Path)
    execution = sub.add_parser("execute")
    execution.add_argument("--model", choices=manifest.MODELS, required=True)
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--job-dir", type=Path, required=True)
    archive = sub.add_parser("archive")
    archive.add_argument("--manifest-dir", type=Path, required=True)
    archive.add_argument("--freeze", dest="freeze_path", type=Path, required=True)
    archive.add_argument("--run-root", type=Path, required=True)
    archive.add_argument("--archive-dir", type=Path, required=True)
    archive.add_argument("--reason", required=True)
    archive.add_argument("--run-id", dest="run_ids", action="append", required=True)
    archive.add_argument("--allow-incompatible", action="store_true")
    args = vars(parser.parse_args())
    action = args.pop("action")
    logging.basicConfig(level=logging.INFO)
    try:
        if action == "plan":
            export = args.pop("export")
            if export is not None:
                export = export.expanduser().resolve()
            plan = make_plan(**args)
            if export:
                outside_runs(export, Path(plan["deployment"]["run_root"]))
                with export.open("xb") as handle:
                    handle.write(manifest._json_bytes(plan))
                print(json.dumps(summary(plan), indent=2, sort_keys=True))
            else:
                print(manifest._json_bytes(plan).decode(), end="")
        elif action == "inventory":
            plan = make_plan(**args, resume=True)
            report = {
                "model": plan["model"], "run_root": plan["deployment"]["run_root"],
                "counts": dict(Counter(run["output_status"] for run in plan["runs"])),
                "runs": [{"run_id": run["run_id"], "status": run["output_status"],
                          "reason": run.get("blocking_reason")} for run in plan["runs"]],
            }
            print(json.dumps(report, indent=2, sort_keys=True))
        elif action == "archive":
            records = archive_runs(**args)
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            execute_plan(args["plan"], args["model"], args["job_dir"])
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Command failed with exit status %s: %s", exc.returncode, exc.cmd)
        sys.exit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
    except (ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
