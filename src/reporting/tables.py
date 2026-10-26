from __future__ import annotations

import csv
import io
import math
from typing import Any

from src.domain_labels import display_domain_name
from src.reporting.validation import METRICS, RunArtifacts

CALIBRATION_COLUMNS = (
    "run_id",
    "model",
    "train_domain_id",
    "train_domain_label",
    "evaluation_domain_id",
    "evaluation_domain_label",
    "bin",
    "lower_bound",
    "upper_bound",
    "count",
    "mean_predicted_probability",
    "observed_positive_rate",
    "absolute_gap",
)


def ordered_runs(runs: list[RunArtifacts], manifest: dict[str, Any]) -> list[RunArtifacts]:
    by_role = {run.spec["role"]: run for run in runs}
    return [
        by_role[f"{domain}_{model}"]
        for domain in manifest["comparison"]["training_domains"]
        for model in manifest["comparison"]["models"]
    ]


def build_metric_rows(runs: list[RunArtifacts], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = manifest["comparison"]
    rows: list[dict[str, Any]] = []
    for run in ordered_runs(runs, manifest):
        train_domain = run.spec["train_domain"]
        for domain in comparison["evaluation_domains"]:
            metrics = run.metrics[domain]
            rows.append({
                "run_id": run.spec["run_id"],
                "model": run.spec["model"],
                "train_domain_id": train_domain,
                "train_domain_label": display_domain_name(train_domain),
                "evaluation_domain_id": domain,
                "evaluation_domain_label": display_domain_name(domain),
                "feature_set": comparison["feature_set"],
                "training_budget": len(run.training_identities),
                "model_seed": comparison["model_seed"],
                "sampling_seed": comparison["sampling_seed"],
                "threshold": run.config["threshold"],
                "prediction_batch_size": run.config["prediction_batch_size"],
                "test_sample_size": metrics["n_rows"],
                "positive_count": metrics["positive_count"],
                "positive_prevalence": metrics["positive_prevalence"],
                **{metric: metrics[metric] for metric in METRICS},
            })
    return rows


def _difference(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return value_b - value_a


def build_difference_rows(runs: list[RunArtifacts], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_role = {run.spec["role"]: run for run in runs}
    rows = []
    for domain in manifest["comparison"]["training_domains"]:
        catboost = by_role[f"{domain}_catboost"]
        tabpfn = by_role[f"{domain}_tabpfn"]
        for evaluation_domain in manifest["comparison"]["evaluation_domains"]:
            cat_metrics = catboost.metrics[evaluation_domain]
            tab_metrics = tabpfn.metrics[evaluation_domain]
            rows.append({
                "training_domain_id": domain,
                "training_domain_label": display_domain_name(domain),
                "evaluation_domain_id": evaluation_domain,
                "evaluation_domain_label": display_domain_name(evaluation_domain),
                "catboost_run_id": catboost.spec["run_id"],
                "tabpfn_run_id": tabpfn.spec["run_id"],
                **{
                    f"tabpfn_minus_catboost_{metric}": _difference(
                        cat_metrics[metric], tab_metrics[metric],
                    )
                    for metric in METRICS
                },
            })
    return rows


def build_runtime_rows(runs: list[RunArtifacts], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    domains = manifest["comparison"]["evaluation_domains"]
    result = []
    for run in ordered_runs(runs, manifest):
        metadata = run.metadata
        compute = metadata["compute_settings"]
        resources = metadata["resources"]
        memory = resources.get("peak_gpu_memory")
        if memory is not None and not isinstance(memory, dict):
            memory = None
        device_name = compute.get("device_name")
        if run.spec["model"] == "catboost":
            device = compute.get("task_type")
            fit_label = "fit"
        else:
            device = compute.get("device")
            fit_label = "context_preparation"
        result.append({
            "run_id": run.spec["run_id"],
            "model": run.spec["model"],
            "training_domain_id": run.spec["train_domain"],
            "training_domain_label": display_domain_name(run.spec["train_domain"]),
            "fit_or_context_preparation_seconds": metadata.get("fit_duration_seconds"),
            "fit_time_kind": fit_label,
            "total_duration_seconds": metadata.get("total_duration_seconds"),
            **{
                f"prediction_{domain}_seconds": metadata["prediction_duration_seconds"].get(domain)
                for domain in domains
            },
            "device": device,
            "hardware": device_name or (metadata.get("environment") or {}).get("machine"),
            "prediction_batch_size": run.config["prediction_batch_size"],
            "cpu_peak_memory_bytes": resources.get("peak_cpu_memory_bytes"),
            "pytorch_peak_allocated_bytes": memory.get("allocated_bytes") if memory else None,
            "pytorch_peak_reserved_bytes": memory.get("reserved_bytes") if memory else None,
            "total_gpu_usage_bytes": resources.get("total_gpu_usage_bytes"),
            "memory_scope": (
                memory.get("scope") if memory else
                "Not recorded; PyTorch allocator peaks are not total GPU usage."
            ),
        })
    return result


def build_calibration_rows(
    runs: list[RunArtifacts], manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for run in ordered_runs(runs, manifest):
        train_domain = run.spec["train_domain"]
        for domain in manifest["comparison"]["evaluation_domains"]:
            for bin_result in run.metrics[domain]["calibration_bins"]:
                rows.append({
                    "run_id": run.spec["run_id"],
                    "model": run.spec["model"],
                    "train_domain_id": train_domain,
                    "train_domain_label": display_domain_name(train_domain),
                    "evaluation_domain_id": domain,
                    "evaluation_domain_label": display_domain_name(domain),
                    "bin": bin_result["bin"],
                    "lower_bound": bin_result["lower_bound"],
                    "upper_bound": bin_result["upper_bound"],
                    "count": bin_result["count"],
                    "mean_predicted_probability": bin_result["mean_predicted_probability"],
                    "observed_positive_rate": bin_result["observed_positive_rate"],
                    "absolute_gap": bin_result["absolute_gap"],
                })
    return rows


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot write an empty report table.")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Report tables cannot contain non-finite values.")
        return f"{value:.6f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[_display(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rendered
    )
    return "\n".join(lines)


def build_report(
    runs: list[RunArtifacts],
    manifest: dict[str, Any],
    validation: dict[str, Any],
    generated_at: str,
    *,
    figure_files: list[str] | None = None,
) -> str:
    by_role = {run.spec["role"]: run for run in runs}
    comparison = manifest["comparison"]
    sections = [
        "# Predictive experiment comparison",
        "",
        f"Generated: {generated_at}",
        "",
        "## Scope and validation",
        "",
        f"- Input validation: **{validation.get('input_validation_status', validation['status'])}** "
        f"across {len(runs)} explicitly selected completed runs.",
        f"- Saved-artifact agreement checks configuration, selected training identities, evaluation identities/labels, "
        f"recorded fingerprints, and prediction-derived metrics (absolute/relative tolerance "
        f"{validation['metric_absolute_and_relative_tolerance']:g}).",
        f"- {validation['claims']}",
        "",
    ]
    sections.extend([
        "## Predictive performance and paired differences",
    ])
    for train_domain in comparison["training_domains"]:
        sections.extend(["", f"### Training on {display_domain_name(train_domain)} (`{train_domain}`)", ""])
        metric_rows: list[list[Any]] = []
        for model in comparison["models"]:
            run = by_role[f"{train_domain}_{model}"]
            for domain in comparison["evaluation_domains"]:
                result = run.metrics[domain]
                metric_rows.append([
                    model,
                    display_domain_name(domain),
                    result["n_rows"],
                    result["positive_count"],
                    result["positive_prevalence"],
                    *(result[key] for key in METRICS),
                ])
        sections.append(_markdown_table(
            ["Model", "Evaluation domain", "N", "Positive", "Prevalence", "ROC-AUC", "AP", "F1", "Brier"],
            metric_rows,
        ))
        differences = build_difference_rows(runs, manifest)
        difference_rows = [
            [
                row["evaluation_domain_label"],
                *(row[f"tabpfn_minus_catboost_{metric}"] for metric in METRICS),
            ]
            for row in differences if row["training_domain_id"] == train_domain
        ]
        sections.extend([
            "",
            "Paired differences (TabPFN minus CatBoost; positive favors TabPFN for AUC/AP/F1, "
            "negative favors TabPFN for Brier):",
            "",
            _markdown_table(["Evaluation domain", "Δ ROC-AUC", "Δ AP", "Δ F1", "Δ Brier"], difference_rows),
        ])
    if figure_files:
        figure_set = set(figure_files)
        sections.extend(["", "Metric figures use a common, non-truncated scale across training domains."])
        for metric, caption in (
            ("roc_auc", "ROC-AUC (higher is better)."),
            ("average_precision", "Average precision (higher is better)."),
            ("brier_score", "Brier score (lower is better)."),
        ):
            name = f"performance_{metric}.png"
            if name in figure_set:
                sections.extend([
                    "",
                    f"![{caption}](figures/{name})",
                    "",
                    f"*{caption}* [PNG](figures/{name}) · [PDF](figures/performance_{metric}.pdf)",
                ])

    sections.extend(["", "## Calibration", ""])
    if figure_files:
        figure_set = set(figure_files)
        for name, caption in (
            ("calibration_reliability.png", "Reliability curves; the diagonal denotes perfect calibration."),
            ("calibration_bin_counts.png", "Per-bin sample support for each plotted reliability curve."),
        ):
            if name in figure_set:
                stem = name.removesuffix(".png")
                sections.extend([
                    f"![{caption}](figures/{name})",
                    "",
                    f"*{caption}* [PNG](figures/{name}) · [PDF](figures/{stem}.pdf)",
                    "",
                ])
        sections.append(
            "The [calibration-bin data](calibration_bins.csv) preserves empty bins with blank undefined values. "
            "Empty bins are omitted from plotted points; the companion count plot shows their zero support."
        )
    else:
        sections.append("[Calibration-bin data](calibration_bins.csv) is available; figures were not requested.")

    sections.extend(["", "## Computational cost", ""])
    runtime_rows = []
    for row in build_runtime_rows(runs, manifest):
        runtime_rows.append([
            row["model"],
            row["training_domain_label"],
            row["fit_time_kind"],
            row["fit_or_context_preparation_seconds"],
            row["total_duration_seconds"],
            *(row[f"prediction_{domain}_seconds"] for domain in comparison["evaluation_domains"]),
            row["device"],
            row["hardware"],
            row["prediction_batch_size"],
            row["pytorch_peak_allocated_bytes"],
            row["pytorch_peak_reserved_bytes"],
            row["total_gpu_usage_bytes"],
        ])
    sections.extend([
        "Run-level fitting/context preparation and total duration appear once per run; prediction durations are "
        "separate columns per evaluation domain and must not be summed as run-level repetitions.",
        "TabPFN fitting is reported as context preparation; externally pretrained compute is not included. "
        "PyTorch allocated/reserved memory is an allocator peak and is not total GPU usage. Missing resource "
        "measurements are left unreported.",
        "",
        _markdown_table(
            [
                "Model", "Training domain", "Fit/context kind", "Fit/context seconds", "Total seconds",
                *(f"Predict {display_domain_name(domain)} seconds" for domain in comparison["evaluation_domains"]),
                "Device", "Hardware", "Batch size", "PyTorch allocated bytes",
                "PyTorch reserved bytes", "Total GPU bytes",
            ],
            runtime_rows,
        ),
        "",
        "Prediction durations are descriptive of these recorded runs, not intrinsic CPU/GPU efficiency. "
        "Differences between TabPFN runs cannot be attributed solely to their batch sizes or training domain.",
    ])
    if figure_files and "prediction_runtime.png" in figure_files:
        sections.extend([
            "",
            "![Per-domain prediction duration by model and training domain](figures/prediction_runtime.png)",
            "",
            "*Per-domain prediction duration; positive values use a log scale. Missing/zero values are annotated "
            "rather than replaced.* [PNG](figures/prediction_runtime.png) · "
            "[PDF](figures/prediction_runtime.pdf)",
        ])
    sections.extend([
        "",
        "## Limitations and provenance",
        "",
        "- These are single-seed observations, not statistical significance claims. Equal training-row budgets "
        "do not match external pretraining resources, validation access, hardware, or compute.",
        "- No confidence intervals or statistical significance claims are reported.",
        "- Higher AUC on a different evaluation population is not evidence by itself of improved transfer.",
        "- Average precision and Brier scores depend on evaluation population and prevalence. NHANES results "
        "describe the existing held-out sample and evaluation protocol, not national-population estimates.",
        "- No model selection, threshold tuning, or calibration fitting occurs here.",
        "- Original Synthetic-trained TabPFN used batch size 256; Playground S5E12-trained TabPFN used batch "
        "size 1024. Model and hardware differences remain confounded.",
        *[f"- {item}" for item in manifest["expected_methodological_differences"]],
        "Test prediction rows were explicitly aligned by authoritative identity before pairing. NHANES uses "
        "`seqn`; source and Playground S5E12 use `row_index`. Domain IDs are retained in CSV machine fields, "
        "with display labels in separate columns. Cross-run population checks compared each evaluation "
        f"domain across {len({check['compared_run_id'] for check in validation['cross_run_population_checks']}) + 1} "
        "selected runs against a fixed reference run.",
        "",
    ])
    if figure_files:
        sections.append("Generated figure files are linked above; calibration values are available in `calibration_bins.csv`.")
    return "\n".join(sections)


def generate_tables(runs: list[RunArtifacts], manifest: dict[str, Any]) -> dict[str, str]:
    metrics = build_metric_rows(runs, manifest)
    differences = build_difference_rows(runs, manifest)
    runtime = build_runtime_rows(runs, manifest)
    calibration = build_calibration_rows(runs, manifest)
    return {
        "metrics.csv": _csv_text(metrics),
        "metric_differences.csv": _csv_text(differences),
        "runtime.csv": _csv_text(runtime),
        "calibration_bins.csv": _csv_text(calibration),
    }
