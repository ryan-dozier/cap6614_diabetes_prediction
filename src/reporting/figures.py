from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.domain_labels import display_domain_name
from src.reporting.tables import build_calibration_rows, ordered_runs
from src.reporting.validation import METRICS, RunArtifacts

MODEL_STYLES = {
    "catboost": {"label": "CatBoost", "color": "#2878B5", "marker": "o"},
    "tabpfn": {"label": "TabPFN", "color": "#E87500", "marker": "s"},
    "tabicl": {"label": "TabICLv2", "color": "#25855A", "marker": "^"},
}
MODEL_NAMES = {key: value["label"] for key, value in MODEL_STYLES.items()}
MODEL_COLORS = {key: value["color"] for key, value in MODEL_STYLES.items()}
FIGURE_FILES = (
    "performance_roc_auc.png",
    "performance_roc_auc.pdf",
    "performance_average_precision.png",
    "performance_average_precision.pdf",
    "performance_brier_score.png",
    "performance_brier_score.pdf",
    "calibration_reliability.png",
    "calibration_reliability.pdf",
    "calibration_bin_counts.png",
    "calibration_bin_counts.pdf",
    "prediction_runtime.png",
    "prediction_runtime.pdf",
)
METRIC_TITLES = {
    "roc_auc": ("ROC-AUC", "Higher is better"),
    "average_precision": ("Average precision", "Higher is better"),
    "brier_score": ("Brier score", "Lower is better; overall probability accuracy"),
}


def performance_points(
    runs: list[RunArtifacts], manifest: dict[str, Any], metric: str,
) -> list[dict[str, Any]]:
    if metric not in METRICS:
        raise ValueError(f"Unsupported predictive comparison metric: {metric}")
    return [
        {
            "run_id": run.spec["run_id"],
            "model": run.spec["model"],
            "model_label": MODEL_STYLES[run.spec["model"]]["label"],
            "model_color": MODEL_STYLES[run.spec["model"]]["color"],
            "model_marker": MODEL_STYLES[run.spec["model"]]["marker"],
            "training_domain_id": run.spec["train_domain"],
            "evaluation_domain_id": domain,
            "evaluation_domain_label": display_domain_name(domain),
            "value": run.metrics[domain][metric],
        }
        for run in ordered_runs(runs, manifest)
        for domain in manifest["comparison"]["evaluation_domains"]
    ]


def calibration_points(
    runs: list[RunArtifacts], manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return build_calibration_rows(runs, manifest)


def runtime_points(
    runs: list[RunArtifacts], manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    points = []
    for run in ordered_runs(runs, manifest):
        for domain in manifest["comparison"]["evaluation_domains"]:
            points.append({
                "run_id": run.spec["run_id"],
                "model": run.spec["model"],
                "model_label": MODEL_NAMES[run.spec["model"]],
                "training_domain_id": run.spec["train_domain"],
                "evaluation_domain_id": domain,
                "evaluation_domain_label": display_domain_name(domain),
                "duration_seconds": run.metadata["prediction_duration_seconds"].get(domain),
                "device": run.metadata["compute_settings"].get("device")
                or run.metadata["compute_settings"].get("task_type"),
                "hardware": (
                    run.metadata["compute_settings"].get("device_name")
                    or (run.metadata.get("environment") or {}).get("machine")
                ),
                "prediction_batch_size": run.config["prediction_batch_size"],
                "native_ensemble_batch_size": run.metadata["compute_settings"].get("ensemble_batch_size"),
                "native_ensemble_size": (run.metadata.get("model") or {}).get("effective_ensemble_count"),
                "column_output_storage": (
                    (run.config.get("model_params") or {}).get("offload_mode")
                    if run.spec["model"] == "tabicl" else None
                ),
                "kv_cache": (run.config.get("model_params") or {}).get("kv_cache"),
            })
    return points


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save_figure(figure: Any, plt: Any, output: Path, stem: str) -> list[str]:
    png_name = f"{stem}.png"
    pdf_name = f"{stem}.pdf"
    try:
        figure.savefig(output / png_name, dpi=240, bbox_inches="tight")
        figure.savefig(output / pdf_name, bbox_inches="tight")
    finally:
        plt.close(figure)
    return [png_name, pdf_name]


def _performance_figure(
    runs: list[RunArtifacts],
    manifest: dict[str, Any],
    metric: str,
    output: Path,
    plt: Any,
) -> list[str]:
    title, direction = METRIC_TITLES[metric]
    points = performance_points(runs, manifest, metric)
    train_domains = manifest["comparison"]["training_domains"]
    eval_domains = manifest["comparison"]["evaluation_domains"]
    values = [point["value"] for point in points if point["value"] is not None]
    if metric in {"roc_auc", "average_precision"}:
        y_limits = (0, 1)
    else:
        maximum = max(values, default=0.0)
        y_limits = (0, min(1.0, max(0.5, math.ceil(maximum * 10) / 10 + 0.05)))

    figure, axes = plt.subplots(
        1, len(train_domains), figsize=(14, 5.5), sharey=True, squeeze=False,
    )
    axes = list(axes[0])
    offsets = {
        model: (index - (len(manifest["comparison"]["models"]) - 1) / 2) * 0.22
        for index, model in enumerate(manifest["comparison"]["models"])
    }
    for axis, train_domain in zip(axes, train_domains):
        for model_index, model in enumerate(manifest["comparison"]["models"]):
            x_values = []
            y_values = []
            for eval_index, domain in enumerate(eval_domains):
                point = next(
                    item for item in points
                    if item["training_domain_id"] == train_domain
                    and item["model"] == model
                    and item["evaluation_domain_id"] == domain
                )
                x = eval_index + offsets[model]
                if point["value"] is None:
                    axis.text(
                        x,
                        y_limits[0] + (y_limits[1] - y_limits[0]) * 0.035,
                        "unavailable",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=MODEL_STYLES[model]["color"],
                    )
                else:
                    x_values.append(x)
                    y_values.append(point["value"])
            axis.scatter(
                x_values,
                y_values,
                color=MODEL_STYLES[model]["color"],
                marker=MODEL_STYLES[model]["marker"],
                label=MODEL_STYLES[model]["label"],
                s=62,
                zorder=3,
            )
        axis.set_xticks(range(len(eval_domains)))
        axis.set_xticklabels(
            [display_domain_name(domain) for domain in eval_domains],
            rotation=16,
            ha="right",
        )
        axis.set_ylim(*y_limits)
        axis.set_title(f"Training: {display_domain_name(train_domain)}")
        axis.set_ylabel(title)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_xlabel("Evaluation domain")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(labels), frameon=False)
    figure.suptitle(f"{title} — {direction}", y=0.91, fontsize=14)
    figure.tight_layout(rect=(0, 0.06, 1, 0.86))
    return _save_figure(figure, plt, output, f"performance_{metric}")


def _calibration_figures(
    runs: list[RunArtifacts],
    manifest: dict[str, Any],
    output: Path,
    plt: Any,
) -> list[str]:
    rows = calibration_points(runs, manifest)
    train_domains = manifest["comparison"]["training_domains"]
    eval_domains = manifest["comparison"]["evaluation_domains"]
    models = manifest["comparison"]["models"]

    figure, axes = plt.subplots(
        len(train_domains),
        len(eval_domains),
        figsize=(15, 9),
        sharex=True,
        sharey=True,
    )
    sparse_bin_threshold = 20
    sparse_bin_count = sum(0 < row["count"] < sparse_bin_threshold for row in rows)
    for row_index, train_domain in enumerate(train_domains):
        for column_index, domain in enumerate(eval_domains):
            axis = axes[row_index, column_index]
            values = next(
                run.metrics[domain]
                for run in runs
                if run.spec["train_domain"] == train_domain
                and run.spec["model"] == models[0]
            )
            axis.plot(
                [0, 1], [0, 1], linestyle="--", color="#555555",
                linewidth=1, label="Ideal",
            )
            for model in models:
                selected = [
                    row for row in rows
                    if row["train_domain_id"] == train_domain
                    and row["evaluation_domain_id"] == domain
                    and row["model"] == model
                    and row["count"] > 0
                    and row["mean_predicted_probability"] is not None
                    and row["observed_positive_rate"] is not None
                ]
                selected.sort(key=lambda row: row["bin"])
                axis.plot(
                    [row["mean_predicted_probability"] for row in selected],
                    [row["observed_positive_rate"] for row in selected],
                    color=MODEL_STYLES[model]["color"],
                    marker=MODEL_STYLES[model]["marker"],
                    markersize=3.5,
                    linewidth=1.4,
                    label=MODEL_STYLES[model]["label"],
                )
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(color="#e1e1e1", linewidth=0.6)
            axis.set_title(
                f"{display_domain_name(train_domain)} → {display_domain_name(domain)}\n"
                f"N={values['n_rows']:,}; positive prevalence={values['positive_prevalence']:.3f}",
                fontsize=8,
            )
            if row_index == len(train_domains) - 1:
                axis.set_xlabel("Mean predicted probability")
            if column_index == 0:
                axis.set_ylabel("Observed positive fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(models) + 1,
        frameon=False,
    )
    figure.suptitle("Reliability by training and evaluation domain", y=0.99, fontsize=14)
    figure.text(
        0.5,
        0.015,
        f"Empty bins are omitted. {sparse_bin_count} nonempty bin(s) have fewer than "
        f"{sparse_bin_threshold} rows; their observed rates may be unstable.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    files = _save_figure(figure, plt, output, "calibration_reliability")

    figure, axes = plt.subplots(
        len(train_domains),
        len(eval_domains),
        figsize=(15, 9),
        sharex=True,
    )
    bins = sorted({row["bin"] for row in rows})
    bar_width = 0.23
    model_offsets = {
        model: (index - (len(models) - 1) / 2) * bar_width
        for index, model in enumerate(models)
    }
    for row_index, train_domain in enumerate(train_domains):
        for column_index, domain in enumerate(eval_domains):
            axis = axes[row_index, column_index]
            for model in models:
                selected = [
                    row for row in rows
                    if row["train_domain_id"] == train_domain
                    and row["evaluation_domain_id"] == domain
                    and row["model"] == model
                ]
                selected_by_bin = {row["bin"]: row["count"] for row in selected}
                axis.bar(
                    [bin_index + model_offsets[model] for bin_index in bins],
                    [selected_by_bin.get(bin_index, 0) for bin_index in bins],
                    width=bar_width * 0.94,
                    color=MODEL_STYLES[model]["color"],
                    label=MODEL_STYLES[model]["label"],
                )
            axis.set_title(
                f"{display_domain_name(train_domain)} → {display_domain_name(domain)}",
                fontsize=9,
            )
            axis.set_xticks(bins, [str(value + 1) for value in bins])
            axis.set_xlabel("Calibration bin")
            if column_index == 0:
                axis.set_ylabel("Rows in bin")
            axis.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(models),
        frameon=False,
    )
    figure.suptitle("Calibration-bin support (empty bins shown at zero)", y=0.99, fontsize=14)
    figure.text(
        0.5, 0.015,
        "Low-count calibration points can be unstable; panel y-axis scales differ to show sparse-bin support.",
        ha="center", va="bottom", fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    files.extend(_save_figure(figure, plt, output, "calibration_bin_counts"))
    return files


def _runtime_figure(
    runs: list[RunArtifacts],
    manifest: dict[str, Any],
    output: Path,
    plt: Any,
) -> list[str]:
    points = runtime_points(runs, manifest)
    train_domains = manifest["comparison"]["training_domains"]
    eval_domains = manifest["comparison"]["evaluation_domains"]
    models = manifest["comparison"]["models"]
    figure, axes = plt.subplots(1, len(train_domains), figsize=(14, 6.5), sharey=True)
    axes = [axes] if len(train_domains) == 1 else list(axes)
    has_positive_duration = any(
        isinstance(point["duration_seconds"], (int, float))
        and point["duration_seconds"] > 0
        for point in points
    )
    offsets = {
        model: (index - (len(models) - 1) / 2) * 0.22
        for index, model in enumerate(models)
    }
    for axis, train_domain in zip(axes, train_domains):
        for model in models:
            x_values = []
            y_values = []
            for domain_index, domain in enumerate(eval_domains):
                point = next(
                    item for item in points
                    if item["training_domain_id"] == train_domain
                    and item["model"] == model
                    and item["evaluation_domain_id"] == domain
                )
                x = domain_index + offsets[model]
                duration = point["duration_seconds"]
                if isinstance(duration, (int, float)) and duration > 0:
                    x_values.append(x)
                    y_values.append(duration)
                else:
                    text = "0 s" if duration == 0 else "not recorded"
                    axis.text(
                        x,
                        0.03,
                        text,
                        transform=axis.get_xaxis_transform(),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        color=MODEL_STYLES[model]["color"],
                        fontsize=7,
                    )
            axis.scatter(
                x_values,
                y_values,
                color=MODEL_STYLES[model]["color"],
                marker=MODEL_STYLES[model]["marker"],
                label=MODEL_STYLES[model]["label"],
                s=62,
                zorder=3,
            )
        if has_positive_duration:
            axis.set_yscale("log")
        axis.set_xticks(range(len(eval_domains)))
        axis.set_xticklabels(
            [display_domain_name(domain) for domain in eval_domains],
            rotation=16,
            ha="right",
        )
        axis.set_title(f"Training: {display_domain_name(train_domain)}")
        axis.set_ylabel("Prediction duration (seconds; log scale)" if has_positive_duration else "Prediction duration")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_xlabel("Evaluation domain")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(labels), frameon=False)
    hardware_lines = []
    for run in ordered_runs(runs, manifest):
        compute = run.metadata["compute_settings"]
        device = compute.get("device") or compute.get("task_type") or "unknown device"
        hardware = compute.get("device_name") or (run.metadata.get("environment") or {}).get("machine")
        extra = ""
        if run.spec["model"] == "tabicl":
            extra = (
                f", native ensemble batch {compute.get('ensemble_batch_size')}, "
                f"column output {run.config['model_params'].get('offload_mode')}, "
                f"KV cache {run.config['model_params'].get('kv_cache')}"
            )
        hardware_lines.append(
            f"{MODEL_STYLES[run.spec['model']]['label']}/{display_domain_name(run.spec['train_domain'])}: "
            f"{device} ({hardware or 'hardware not recorded'}), external query batch "
            f"{run.config['prediction_batch_size']}{extra}"
        )
    figure.suptitle("Prediction duration by evaluation population", y=0.99, fontsize=14)
    figure.text(
        0.5,
        0.02,
        "\n".join(hardware_lines)
        + "\nCPU/GPU timing is not an intrinsic efficiency comparison; differences cannot be attributed "
        "solely to query batch size or training domain.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.2, 1, 0.94))
    return _save_figure(figure, plt, output, "prediction_runtime")


def generate_figures(
    runs: list[RunArtifacts],
    manifest: dict[str, Any],
    output_directory: Path,
) -> tuple[list[str], str]:
    matplotlib, plt = _pyplot()
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    try:
        for metric in ("roc_auc", "average_precision", "brier_score"):
            generated.extend(_performance_figure(runs, manifest, metric, output_directory, plt))
        generated.extend(_calibration_figures(runs, manifest, output_directory, plt))
        generated.extend(_runtime_figure(runs, manifest, output_directory, plt))
        return generated, matplotlib.__version__
    finally:
        plt.close("all")
