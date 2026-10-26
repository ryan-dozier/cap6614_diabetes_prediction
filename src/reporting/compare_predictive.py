from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

from src.paths import PROJECT_ROOT, RESULTS_DIR
from src.reporting.tables import build_report, generate_tables
from src.reporting.validation import (
    ReportValidationError,
    _check_manifest,
    _input_run_dir,
    validate_comparison,
)


def _resolve_from_project(value: Path) -> Path:
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _output_directory(value: Path, manifest: dict[str, Any]) -> Path:
    output = _resolve_from_project(value)
    results_root = RESULTS_DIR.resolve()
    reports_path = RESULTS_DIR / "reports"
    reports_root = reports_path.resolve()
    if reports_path.is_symlink() or reports_root != results_root / "reports":
        raise ReportValidationError(f"results/reports must be a real directory under results/: {reports_root}")
    if output == reports_root or not output.is_relative_to(reports_root):
        raise ReportValidationError(f"output directory must be a new child of {reports_root}")
    if output.exists() or output.is_symlink():
        raise ReportValidationError(f"refusing to overwrite existing output directory: {output}")

    runs_root = (RESULTS_DIR / "predictive").resolve()
    for spec in manifest["runs"]:
        input_dir = _input_run_dir(spec, runs_root)
        if output == input_dir or output.is_relative_to(input_dir) or input_dir.is_relative_to(output):
            raise ReportValidationError(
                f"output directory overlaps input run {spec['run_id']!r}: {input_dir}"
            )
    return output


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _reporting_environment() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_state = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_state = {
            "available": True,
            "revision": revision,
            "dirty": bool(dirty_state),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        git_state = {
            "available": False,
            "revision": None,
            "dirty": None,
            "error": str(exc),
        }
    return {
        "python": platform.python_version(),
        "packages": packages,
        "git": git_state,
    }


def _failure_diagnostic(
    output: Path,
    generated_at: str,
    stage: str,
    error: Exception,
    input_validation_status: str,
    figures_requested: bool,
) -> None:
    partial_outputs = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "validation.json"
    )
    diagnostic = {
        "status": "failed",
        "input_validation_status": input_validation_status,
        "report_generation_status": "failed",
        "failed_stage": stage,
        "figures_requested": figures_requested,
        "partial_outputs": partial_outputs,
        "generated_at_utc": generated_at,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    try:
        _write_json(output / "validation.json", diagnostic)
    except Exception as diagnostic_error:
        print(
            f"Could not write failure diagnostics after {type(error).__name__}: "
            f"{diagnostic_error}",
            file=sys.stderr,
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReportValidationError(f"manifest {path}: cannot read valid JSON ({exc})") from exc
    _check_manifest(manifest)
    return manifest


def run_report(
    manifest_path: Path,
    output_arg: Path,
    *,
    with_figures: bool = False,
) -> Path:
    resolved_manifest = _resolve_from_project(manifest_path)
    manifest = _load_manifest(resolved_manifest)
    output = _output_directory(output_arg, manifest)
    output.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    input_validation_status = "in_progress"
    stage = "input_validation"
    figure_files: list[str] = []
    matplotlib_version: str | None = None
    try:
        runs, validation = validate_comparison(manifest, resolved_manifest)
        input_validation_status = "passed"
        validation.update({
            "status": "in_progress",
            "input_validation_status": input_validation_status,
            "report_generation_status": "in_progress",
            "generated_at_utc": generated_at,
        })
        stage = "output_writing"
        _write_json(output / "validation.json", validation)

        stage = "table_generation"
        tables = generate_tables(runs, manifest)
        if "calibration_bins.csv" not in tables:
            raise ValueError("report table generation omitted calibration_bins.csv")

        if with_figures:
            stage = "plotting"
            from src.reporting.figures import FIGURE_FILES, generate_figures

            generated_figure_files, matplotlib_version = generate_figures(
                runs, manifest, output / "figures",
            )
            expected_figures = set(FIGURE_FILES)
            if set(generated_figure_files) != expected_figures:
                raise RuntimeError(
                    "plotting did not produce the complete requested figure set: "
                    f"expected {sorted(expected_figures)}, got {sorted(generated_figure_files)}"
                )
            figure_files = sorted(generated_figure_files)

        stage = "report_rendering"
        report = build_report(
            runs,
            manifest,
            validation,
            generated_at,
            figure_files=[name for name in figure_files if name.endswith(".png")],
        )
        report_manifest = {
            "report_schema_version": manifest["report_schema_version"],
            "generated_at_utc": generated_at,
            "source_manifest": str(resolved_manifest),
            "selection": manifest["comparison"],
            "resolved_runs": [
                {
                    "role": run.spec["role"],
                    "run_id": run.spec["run_id"],
                    "model": run.spec["model"],
                    "train_domain": run.spec["train_domain"],
                    "artifact_dir": str(run.run_dir),
                    "artifact_sha256": run.artifact_hashes,
                }
                for run in runs
            ],
            "source_manifest_sha256": validation["input_manifest_sha256"],
            "reporting_environment": _reporting_environment(),
            "figures_requested": with_figures,
            "figure_files": [f"figures/{name}" for name in figure_files],
            "calibration_table_file": "calibration_bins.csv",
            "matplotlib_version": matplotlib_version,
        }

        stage = "output_writing"
        _write_json(output / "manifest.json", report_manifest)
        for name, contents in tables.items():
            (output / name).write_text(contents, encoding="utf-8", newline="")
        (output / "report.md").write_text(report, encoding="utf-8", newline="")
        required_outputs = {
            "manifest.json",
            "validation.json",
            "metrics.csv",
            "metric_differences.csv",
            "runtime.csv",
            "calibration_bins.csv",
            "report.md",
        }
        required_outputs.update(f"figures/{name}" for name in figure_files)
        missing_outputs = sorted(
            name for name in required_outputs if not (output / name).is_file()
        )
        if missing_outputs:
            raise OSError(f"report output files are missing: {missing_outputs}")

        validation.update({
            "status": "completed",
            "input_validation_status": input_validation_status,
            "report_generation_status": "completed",
        })
        _write_json(output / "validation.json", validation)
    except Exception as exc:
        if stage == "input_validation" and input_validation_status == "in_progress":
            input_validation_status = "failed"
        _failure_diagnostic(
            output, generated_at, stage, exc, input_validation_status, with_figures,
        )
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate explicitly selected completed predictive runs and write comparison tables."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Versioned report manifest.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory under results/reports/ (relative paths resolve from the project root).",
    )
    parser.add_argument(
        "--with-figures", action="store_true",
        help="Generate PNG/PDF performance, calibration, and prediction-runtime figures.",
    )
    args = parser.parse_args(argv)
    created_output: Path | None = None
    requested_output = _resolve_from_project(args.output_dir)
    output_preexisted = requested_output.exists() or requested_output.is_symlink()
    try:
        output = run_report(args.manifest, args.output_dir, with_figures=args.with_figures)
    except Exception as exc:
        if not output_preexisted and (requested_output / "validation.json").is_file():
            created_output = requested_output
        print(f"Predictive comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if created_output is not None:
            print(f"Failure details: {created_output / 'validation.json'}", file=sys.stderr)
        return 1
    print(f"Predictive comparison report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
