from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.config import ExperimentConfig


def create_run_directory(config: ExperimentConfig) -> Path:
    output = config.resolved_output_dir
    output.mkdir(parents=True, exist_ok=False)
    return output


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON.")


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, default=_json_default, allow_nan=False, indent=2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot serialize an empty result table.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluations(output: Path, evaluations: dict[str, dict[str, Any]]) -> None:
    write_json(output / "metrics.json", evaluations)
    metric_rows = []
    calibration_rows = []
    for domain, evaluation in evaluations.items():
        row = {"domain": domain, "split": "test"}
        row.update({
            key: value for key, value in evaluation.items()
            if key not in {"confusion_matrix", "calibration_bins", "warnings"}
        })
        row.update(evaluation["confusion_matrix"])
        row["warnings"] = "; ".join(evaluation["warnings"])
        metric_rows.append(row)
        calibration_rows.extend(
            {"domain": domain, "split": "test", **bin_result}
            for bin_result in evaluation["calibration_bins"]
        )
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "calibration_bins.csv", calibration_rows)


def write_predictions(output: Path, domain: str, predictions: pd.DataFrame) -> None:
    predictions.to_csv(
        output / f"predictions_{domain}_test.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
