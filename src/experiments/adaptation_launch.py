"""Plan and execute approved Candidate A model jobs without automatic recovery."""

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
from pathlib import Path
from typing import Any

from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from src.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
SCHEMA = "x03a_launch_plan_v1"


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


def make_plan(*, project: Path, python: Path, manifest_dir: Path, freeze_path: Path,
              run_root: Path, model: str = "all",
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
    _, runs = matrix.load(manifest_dir)
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
        entries.append({
            "run_id": spec["run_id"], "specification": spec,
            "effective_configuration": config, "command": command, "output_path": str(output),
            "output_status": "existing" if os.path.lexists(output) else "pending",
        })
    deployment = {
        "project": str(project), "python": str(python), "manifest_dir": str(manifest_dir),
        "freeze_path": str(freeze_path), "run_root": str(run_root),
        "tabpfn_checkpoint": checkpoints.get("tabpfn"), "tabicl_checkpoint": checkpoints.get("tabicl"),
    }
    plan = {
        "schema_version": SCHEMA, "model": model, "mode": "research",
        "deployment": deployment, "matrix_sha256": matrix.matrix_sha256(runs),
        "freeze_sha256": manifest._sha256_file(freeze_path),
        "file_sha256": _fingerprints(project, manifest_dir, freeze_path, checkpoints),
        "grid_counts": counts, "runs": entries,
    }
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": plan["model"], "mode": plan["mode"], "run_count": len(plan["runs"]),
        "pending": sum(run["output_status"] == "pending" for run in plan["runs"]),
        "existing_run_ids": [run["run_id"] for run in plan["runs"] if run["output_status"] == "existing"],
        "grid_counts": plan["grid_counts"], "run_root": plan["deployment"]["run_root"],
        "plan_sha256": plan["plan_sha256"], "freeze_sha256": plan["freeze_sha256"],
    }


def validate_plan(plan: dict[str, Any], model: str) -> None:
    if plan.get("schema_version") != SCHEMA or plan.get("model") != model or model not in manifest.MODELS:
        raise ValueError("Execution requires a plan for exactly the submitted model.")
    existing = [run["run_id"] for run in plan["runs"] if os.path.lexists(run["output_path"])]
    if existing:
        raise FileExistsError(f"Existing outputs; X03A refuses all execution: {', '.join(existing)}")
    deployment = {key: Path(value) if value is not None else None
                  for key, value in plan["deployment"].items()}
    current = make_plan(model=model, **deployment)
    if current != plan:
        raise ValueError("Deployment, scientific inputs, code, or commands changed since planning; regenerate and review.")


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
    status = {"status": "failed", "job_id": os.environ["SLURM_JOB_ID"],
              "model": model, "plan_sha256": plan["plan_sha256"],
              "freeze_sha256": plan["freeze_sha256"], "completed_run_ids": []}
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
            for run in plan["runs"]:
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
                (job_dir / f"{run['run_id']}.command.json").write_bytes(manifest._json_bytes(run["command"]))
                with (job_dir / f"{run['run_id']}.log").open("x") as run_log:
                    subprocess.run(
                        ["/usr/bin/time", "-v", "-o", str(job_dir / f"{run['run_id']}.time"),
                         *run["command"]],
                        cwd=project, stdout=run_log, stderr=subprocess.STDOUT, check=True,
                    )
                status["completed_run_ids"].append(run["run_id"])
            status["status"] = "completed"
        finally:
            (job_dir / "status.json").write_bytes(manifest._json_bytes(status))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--model", choices=(*manifest.MODELS, "all"), default="all")
    planning.add_argument("--project", type=Path, default=PROJECT_ROOT)
    planning.add_argument("--python", type=Path, default=Path(sys.executable))
    planning.add_argument("--manifest-dir", type=Path, required=True)
    planning.add_argument("--freeze", dest="freeze_path", type=Path, required=True)
    planning.add_argument("--run-root", type=Path, required=True)
    planning.add_argument("--tabpfn-checkpoint", type=Path)
    planning.add_argument("--tabicl-checkpoint", type=Path)
    planning.add_argument("--export", type=Path)
    execution = sub.add_parser("execute")
    execution.add_argument("--model", choices=manifest.MODELS, required=True)
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--job-dir", type=Path, required=True)
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
