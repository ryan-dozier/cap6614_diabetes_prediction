from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.experiments.config import positive_integer
from src.experiments.metrics import binary_labels


@dataclass
class Partition:
    X: pd.DataFrame
    y: pd.Series
    provenance: pd.DataFrame
    identities: frozenset[tuple[str, int]]
    summary: dict[str, Any]


def identity_fingerprints(domain: str, identity_column: str, values: Iterable[int]) -> dict[str, str]:
    identity_lines = [
        f"{domain}\t{identity_column}\t{int(value)}\n"
        for value in values
    ]
    return {
        "identity_sha256": hashlib.sha256("".join(sorted(identity_lines)).encode()).hexdigest(),
        "ordered_identity_sha256": hashlib.sha256("".join(identity_lines).encode()).hexdigest(),
    }


def partition_summary(
    domain: str, split: str, provenance: pd.DataFrame, y: pd.Series,
) -> dict[str, Any]:
    identity_column = "seqn" if domain == "nhanes" else "row_index"
    labels = binary_labels(y)
    return {
        "domain": domain, "split": split, "rows": len(labels),
        "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "identity_column": identity_column,
        **identity_fingerprints(domain, identity_column, provenance[identity_column]),
    }


def select_training(
    partition: Partition, sample_size: int | None, sampling_seed: int | None,
) -> Partition:
    if partition.summary["split"] != "train":
        raise ValueError("Training sampling may only select rows from the train partition.")
    if sample_size is None:
        return partition
    positive_integer(sample_size, "train_sample_size")
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32:
        raise ValueError("Sampled training requires an explicit integer sampling_seed in [0, 2**32).")
    available = len(partition.X)
    if sample_size < 2 or sample_size > available:
        raise ValueError(f"train_sample_size must be between 2 and available training rows ({available}); got {sample_size}.")
    if len(partition.y) != available or len(partition.provenance) != available:
        raise ValueError("Training X/y/provenance lengths must match.")
    identity_column = partition.summary["identity_column"]
    identities = partition.provenance[identity_column]
    if identities.isna().any() or identities.duplicated().any():
        raise ValueError("Training identities must be nonmissing and unique.")
    order = np.argsort(identities.to_numpy(), kind="stable")
    labels = binary_labels(partition.y.iloc[order])
    class_rows = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(rows) == 0 for rows in class_rows):
        raise ValueError("Training sampling requires both binary classes.")
    # Nearest proportional positive count (half-up), constrained to feasible
    # allocations retaining both classes. Within-class draws are without replacement.
    positives = int(np.floor(sample_size * len(class_rows[1]) / available + 0.5))
    positives = min(max(positives, 1, sample_size - len(class_rows[0])), sample_size - 1, len(class_rows[1]))
    rng = np.random.default_rng(sampling_seed)
    selected = np.sort(np.concatenate([
        rng.choice(class_rows[0], sample_size - positives, replace=False),
        rng.choice(class_rows[1], positives, replace=False),
    ]))
    positions = order[selected]
    X = partition.X.iloc[positions].reset_index(drop=True).copy()
    y = partition.y.iloc[positions].reset_index(drop=True).copy()
    provenance = partition.provenance.iloc[positions].reset_index(drop=True).copy()
    domain = partition.summary["domain"]
    return Partition(
        X=X, y=y, provenance=provenance,
        identities=frozenset((domain, int(value)) for value in provenance[identity_column]),
        summary=partition_summary(domain, "train", provenance, y),
    )


def sampling_metadata(
    full: Partition, selected: Partition, sample_size: int | None, seed: int | None,
) -> dict[str, Any]:
    return {
        "available_training_rows": len(full.X),
        "selected_training_rows": len(selected.X),
        "requested_budget": sample_size,
        "sampling_seed": seed,
        "algorithm": (
            "Identity-sorted input; proportional binary allocation with half-up rounding "
            "and both-class feasibility constraints; NumPy default_rng within-class "
            "sampling without replacement; selected rows in ascending identity order."
            if sample_size is not None else "All training rows, original gateway order."
        ),
        "selected_class_counts": selected.summary["class_counts"],
        "identity_sha256": selected.summary["identity_sha256"],
        "ordered_identity_sha256": selected.summary["ordered_identity_sha256"],
    }
