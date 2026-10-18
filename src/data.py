from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# =============================================================================
# Types
# =============================================================================

Domain = Literal["source", "kaggle"]
Split = Literal["train", "val", "test"]

VALID_DOMAINS = ("source", "kaggle")
VALID_SPLITS = ("train", "val", "test")


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data/processed"
SPLITS_DIR = ROOT / "data/splits"

METADATA_PATH = PROCESSED_DIR / "metadata.json"

COMMON_PATHS = {
    "source": PROCESSED_DIR / "source_common.csv",
    "kaggle": PROCESSED_DIR / "kaggle_common.csv",
}

SPLIT_ASSIGNMENT_PATHS = {
    "source": SPLITS_DIR / "source_split_assignments.csv",
    "kaggle": SPLITS_DIR / "kaggle_split_assignments.csv",
}


# =============================================================================
# Dataset schema
# =============================================================================


@dataclass(frozen=True)
class DatasetSchema:
    target: str
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    source_only_excluded: tuple[str, ...]

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.features + (self.target,)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_domain(domain: str) -> Domain:
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Expected one of: {VALID_DOMAINS}"
        )

    return domain  # type: ignore[return-value]


def _validate_split(split: str) -> Split:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of: {VALID_SPLITS}"
        )

    return split  # type: ignore[return-value]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required prepared-data file does not exist:\n"
            f"  {path}\n\n"
            "Run:\n"
            "  python3 src/prepare_data.py\n"
            "before running experiments."
        )


# =============================================================================
# Metadata
# =============================================================================


@lru_cache(maxsize=1)
def _load_metadata_cached() -> dict:
    _require_file(METADATA_PATH)

    with METADATA_PATH.open("r") as file:
        metadata = json.load(file)

    required_keys = {
        "schema_version",
        "target",
        "features",
        "datasets",
    }

    missing = required_keys - set(metadata)

    if missing:
        raise ValueError(
            "Prepared-data metadata is missing required keys: "
            + ", ".join(sorted(missing))
        )

    return metadata


def load_metadata() -> dict:
    """
    Return preparation metadata.

    A deep copy is returned so experiment code cannot accidentally modify the
    cached metadata object.
    """
    return copy.deepcopy(_load_metadata_cached())


@lru_cache(maxsize=1)
def load_schema() -> DatasetSchema:
    metadata = _load_metadata_cached()

    features = metadata["features"]

    return DatasetSchema(
        target=metadata["target"],
        features=tuple(features["all_shared"]),
        categorical_features=tuple(features["categorical"]),
        numeric_features=tuple(features["numeric"]),
        source_only_excluded=tuple(
            features["source_only_excluded"]
        ),
    )


# =============================================================================
# Canonical datasets
# =============================================================================


@lru_cache(maxsize=2)
def _load_common_cached(domain: Domain) -> pd.DataFrame:
    path = COMMON_PATHS[domain]
    _require_file(path)

    df = pd.read_csv(path)

    schema = load_schema()

    expected_columns = list(schema.model_columns)
    actual_columns = list(df.columns)

    if actual_columns != expected_columns:
        raise ValueError(
            f"{domain} canonical dataset schema mismatch.\n"
            f"Expected:\n  {expected_columns}\n"
            f"Observed:\n  {actual_columns}"
        )

    metadata = _load_metadata_cached()

    expected_rows = metadata["datasets"][domain]["rows"]

    if len(df) != expected_rows:
        raise ValueError(
            f"{domain} canonical dataset row-count mismatch. "
            f"Expected {expected_rows:,}, observed {len(df):,}."
        )

    return df


def load_common(domain: Domain) -> pd.DataFrame:
    """
    Load the complete canonical 24-feature dataset for a domain.

    This should normally be used for analysis rather than model training.
    Experiment code should usually call load_split().
    """
    domain = _validate_domain(domain)

    return _load_common_cached(domain).copy()


# =============================================================================
# Split assignments
# =============================================================================


@lru_cache(maxsize=2)
def _load_assignments_cached(
    domain: Domain,
) -> pd.DataFrame:
    path = SPLIT_ASSIGNMENT_PATHS[domain]
    _require_file(path)

    assignments = pd.read_csv(path)

    required_columns = {
        "row_index",
        "split",
    }

    if domain == "kaggle":
        required_columns.add("id")

    missing = required_columns - set(assignments.columns)

    if missing:
        raise ValueError(
            f"{domain} split assignments are missing columns: "
            + ", ".join(sorted(missing))
        )

    common = _load_common_cached(domain)

    if len(assignments) != len(common):
        raise ValueError(
            f"{domain} split-assignment row count does not match "
            f"the canonical dataset."
        )

    # prepare_data.py writes one assignment row per original dataset row,
    # ordered by original row_index.
    row_indices = assignments["row_index"].to_numpy()

    expected_indices = np.arange(len(common))

    if not np.array_equal(
        row_indices,
        expected_indices,
    ):
        raise ValueError(
            f"{domain} split assignments do not contain the expected "
            "ordered row indices."
        )

    observed_splits = set(assignments["split"].unique())

    if observed_splits != set(VALID_SPLITS):
        raise ValueError(
            f"{domain} split assignments contain unexpected split "
            f"labels: {sorted(observed_splits)}"
        )

    # Validate against the counts recorded during preparation.
    metadata = _load_metadata_cached()

    metadata_splits = metadata["datasets"][domain]["splits"]

    for split in VALID_SPLITS:
        observed_count = int(
            (assignments["split"] == split).sum()
        )

        expected_count = int(
            metadata_splits[split]["rows"]
        )

        if observed_count != expected_count:
            raise ValueError(
                f"{domain}/{split} split-size mismatch. "
                f"Expected {expected_count:,}, "
                f"observed {observed_count:,}."
            )

    return assignments


def load_split_assignments(
    domain: Domain,
) -> pd.DataFrame:
    """
    Return the authoritative row-to-split assignment table.
    """
    domain = _validate_domain(domain)

    return _load_assignments_cached(domain).copy()


# =============================================================================
# Fixed train / validation / test splits
# =============================================================================


def load_split(
    domain: Domain,
    split: Split,
    *,
    include_row_index: bool = False,
    include_id: bool = False,
) -> pd.DataFrame:
    """
    Load one fixed dataset split.

    The split is reconstructed from the canonical dataset and the authoritative
    split-assignment file.

    Parameters
    ----------
    domain:
        "source" or "kaggle".

    split:
        "train", "val", or "test".

    include_row_index:
        Add the original canonical-dataset row index as a column.

    include_id:
        Add the original Kaggle id column. Only valid for domain="kaggle".

        id is intentionally excluded by default so it cannot accidentally enter
        normal model training. It should only be enabled for explicit artifact
        experiments.

    Returns
    -------
    pandas.DataFrame
        The requested split, with rows in canonical dataset order.
    """

    domain = _validate_domain(domain)
    split = _validate_split(split)

    if include_id and domain != "kaggle":
        raise ValueError(
            "include_id=True is only valid for the Kaggle domain."
        )

    common = _load_common_cached(domain)
    assignments = _load_assignments_cached(domain)

    mask = assignments["split"].eq(split)

    row_indices = (
        assignments.loc[mask, "row_index"]
        .to_numpy(dtype=np.int64)
    )

    df = (
        common.iloc[row_indices]
        .reset_index(drop=True)
        .copy()
    )

    if include_row_index:
        df.insert(
            0,
            "row_index",
            row_indices,
        )

    if include_id:
        ids = (
            assignments.loc[mask, "id"]
            .to_numpy()
        )

        insert_location = (
            1 if include_row_index else 0
        )

        df.insert(
            insert_location,
            "id",
            ids,
        )

    return df


# =============================================================================
# Feature / target convenience loader
# =============================================================================


def load_xy(
    domain: Domain,
    split: Split,
    *,
    include_id: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a fixed split and return predictors (X) and binary target (y).

    No encoding, scaling, imputation, feature selection, or other learned
    preprocessing is performed here.
    """

    df = load_split(
        domain,
        split,
        include_id=include_id,
    )

    schema = load_schema()

    feature_columns = list(schema.features)

    if include_id:
        feature_columns = ["id"] + feature_columns

    X = df[feature_columns].copy()
    y = df[schema.target].copy()

    return X, y


# =============================================================================
# Split summaries
# =============================================================================


def split_summary(
    domain: Domain,
    split: Split,
) -> dict:
    """
    Return a small summary of a prepared split.
    """

    df = load_split(domain, split)

    schema = load_schema()

    counts = (
        df[schema.target]
        .value_counts()
        .sort_index()
    )

    proportions = (
        df[schema.target]
        .value_counts(normalize=True)
        .sort_index()
    )

    return {
        "domain": domain,
        "split": split,
        "rows": len(df),
        "features": len(schema.features),
        "class_counts": {
            int(label): int(count)
            for label, count in counts.items()
        },
        "class_proportions": {
            int(label): float(proportion)
            for label, proportion
            in proportions.items()
        },
    }


# =============================================================================
# Manual smoke test
# =============================================================================


def main() -> None:
    schema = load_schema()

    print("=" * 80)
    print("DATA SCHEMA")
    print("=" * 80)

    print(f"Target: {schema.target}")
    print(f"Features: {len(schema.features)}")
    print(
        f"Categorical: {len(schema.categorical_features)}"
    )
    print(
        f"Numeric: {len(schema.numeric_features)}"
    )

    print("\n" + "=" * 80)
    print("FIXED SPLITS")
    print("=" * 80)

    for domain in VALID_DOMAINS:
        print(f"\n{domain.upper()}")

        for split in VALID_SPLITS:
            summary = split_summary(
                domain,  # type: ignore[arg-type]
                split,   # type: ignore[arg-type]
            )

            positive = (
                summary["class_proportions"]
                .get(1, 0.0)
            )

            print(
                f"  {split:<5} "
                f"rows={summary['rows']:>8,}  "
                f"positive={positive:.6f}"
            )

    print("\n" + "=" * 80)
    print("X / Y TEST")
    print("=" * 80)

    X, y = load_xy("source", "train")

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Target dtype: {y.dtype}")

    X_kaggle, y_kaggle = load_xy(
        "kaggle",
        "train",
    )

    print(f"\nKaggle X shape: {X_kaggle.shape}")
    print(f"Kaggle y shape: {y_kaggle.shape}")

    # Explicit id test.
    X_with_id, _ = load_xy(
        "kaggle",
        "test",
        include_id=True,
    )

    print(
        "\nKaggle test with explicit id: "
        f"{X_with_id.shape}"
    )

    print(
        f"First columns: "
        f"{list(X_with_id.columns[:4])}"
    )


if __name__ == "__main__":
    main()