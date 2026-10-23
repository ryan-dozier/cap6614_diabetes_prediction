from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.experiments.config import positive_integer


def validate_threshold(threshold: float) -> None:
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and strictly between 0 and 1.")


def real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers.")
    return array.astype(np.float64)


def binary_labels(values: ArrayLike) -> NDArray[np.int64]:
    labels = real_vector(values, "labels")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1.")
    return labels.astype(np.int64)


def probabilities(values: ArrayLike) -> NDArray[np.float64]:
    result = real_vector(values, "probabilities")
    if ((result < 0) | (result > 1)).any():
        raise ValueError("probabilities must be in [0, 1].")
    return result


def evaluate_binary(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Average precision, not trapezoidal PR-curve area, is the PR summary."""
    labels = binary_labels(y_true)
    probability = probabilities(y_probability)
    if len(labels) != len(probability):
        raise ValueError("labels and probabilities must have matching lengths.")
    validate_threshold(threshold)
    positive_integer(calibration_bins, "calibration_bins")
    predicted = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    bins: list[dict[str, Any]] = []
    bin_indices = np.minimum((probability * calibration_bins).astype(np.int64), calibration_bins - 1)
    ece = 0.0
    for index in range(calibration_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        mean_probability = float(probability[mask].mean()) if count else None
        positive_rate = float(labels[mask].mean()) if count else None
        gap = abs(mean_probability - positive_rate) if mean_probability is not None and positive_rate is not None else None
        if gap is not None:
            ece += count / len(labels) * gap
        bins.append({
            "bin": index,
            "lower_bound": index / calibration_bins,
            "upper_bound": (index + 1) / calibration_bins,
            "count": count,
            "mean_predicted_probability": mean_probability,
            "observed_positive_rate": positive_rate,
            "absolute_gap": gap,
        })
    return {
        "n_rows": len(labels),
        "positive_count": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probability)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, probability)) if both_classes else None,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probability)),
        "ece": ece,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
        "warnings": [] if both_classes else [
            "Evaluation labels contain one class; roc_auc and average_precision are undefined and reported as null."
        ],
        "calibration_bins": bins,
    }
