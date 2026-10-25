from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from src.domain_labels import display_domain_name

import numpy as np
import pandas as pd

from src.paths import NHANES_PREPARATION_DIR, PROCESSED_DIR, SPLITS_DIR


# =============================================================================
# Types
# =============================================================================

Domain = Literal["source", "kaggle", "nhanes"]
Split = Literal["train", "val", "test"]
FeatureSet = Literal[
    "full_24",
    "strict_common_16",
    "common_20",
    "common_22",
]

VALID_DOMAINS = (
    "source",
    "kaggle",
    "nhanes",
)

VALID_SPLITS = (
    "train",
    "val",
    "test",
)

VALID_FEATURE_SETS = (
    "full_24",
    "strict_common_16",
    "common_20",
    "common_22",
)

DEFAULT_FEATURE_SET = {
    "source": "full_24",
    "kaggle": "full_24",
    "nhanes": "strict_common_16",
}

NHANES_FEATURE_SETS = {
    "strict_common_16",
    "common_20",
    "common_22",
}


# =============================================================================
# Paths
# =============================================================================

METADATA_PATH = (
    PROCESSED_DIR / "metadata.json"
)

NHANES_METADATA_PATH = (
    NHANES_PREPARATION_DIR / "metadata.json"
)

COMMON_PATHS = {
    "source":
        PROCESSED_DIR / "source_common.csv",

    "kaggle":
        PROCESSED_DIR / "kaggle_common.csv",
}

NHANES_COMMON_PATHS = {
    "strict_common_16":
        PROCESSED_DIR / "nhanes_common_16.csv",

    "common_20":
        PROCESSED_DIR / "nhanes_common_20.csv",

    "common_22":
        PROCESSED_DIR / "nhanes_common_22.csv",
}

SPLIT_ASSIGNMENT_PATHS = {
    "source":
        SPLITS_DIR / "source_split_assignments.csv",

    "kaggle":
        SPLITS_DIR / "kaggle_split_assignments.csv",

    "nhanes":
        SPLITS_DIR / "nhanes_split_assignments.csv",
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
        return (
            self.features
            + (self.target,)
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_domain(
    domain: str,
) -> Domain:
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Expected one of: {VALID_DOMAINS}"
        )

    return domain  # type: ignore[return-value]


def _validate_split(
    split: str,
) -> Split:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of: {VALID_SPLITS}"
        )

    return split  # type: ignore[return-value]


def _validate_feature_set(
    feature_set: str,
) -> FeatureSet:
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Expected one of: {VALID_FEATURE_SETS}"
        )

    return feature_set  # type: ignore[return-value]


def _resolve_feature_set(
    domain: Domain,
    feature_set: str | None,
) -> FeatureSet:
    if feature_set is None:
        feature_set = DEFAULT_FEATURE_SET[
            domain
        ]

    resolved = _validate_feature_set(
        feature_set
    )

    if (
        domain == "nhanes"
        and resolved not in NHANES_FEATURE_SETS
    ):
        raise ValueError(
            "NHANES does not support feature_set='full_24'. "
            "Use one of: "
            "'strict_common_16', 'common_20', 'common_22'."
        )

    return resolved


def _require_file(
    path: Path,
    *,
    preparation_hint: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Required prepared-data file does not exist:\n"
            f"  {path}\n\n"
            "Run:\n"
            f"  {preparation_hint}\n"
            "before running experiments."
        )


# =============================================================================
# Metadata
# =============================================================================


@lru_cache(maxsize=1)
def _load_metadata_cached() -> dict:
    _require_file(
        METADATA_PATH,
        preparation_hint=(
            "python3 -m scripts.prepare_data"
        ),
    )

    with METADATA_PATH.open("r") as file:
        metadata = json.load(file)

    required_keys = {
        "schema_version",
        "target",
        "features",
        "datasets",
    }

    missing = (
        required_keys
        - set(metadata)
    )

    if missing:
        raise ValueError(
            "Prepared-data metadata is missing required keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    return metadata


def load_metadata() -> dict:
    """
    Return Source/Kaggle preparation metadata.

    A deep copy is returned so experiment code cannot accidentally modify the
    cached metadata object.
    """

    return copy.deepcopy(
        _load_metadata_cached()
    )


@lru_cache(maxsize=1)
def _load_nhanes_metadata_cached() -> dict:
    _require_file(
        NHANES_METADATA_PATH,
        preparation_hint=(
            "python3 -m scripts.prepare_nhanes"
        ),
    )

    with NHANES_METADATA_PATH.open(
        "r"
    ) as file:
        metadata = json.load(file)

    required_keys = {
        "dataset",
        "target",
        "primary_feature_set",
        "feature_sets",
        "primary_splits",
    }

    missing = (
        required_keys
        - set(metadata)
    )

    if missing:
        raise ValueError(
            "NHANES metadata is missing required keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    return metadata


def load_nhanes_metadata() -> dict:
    """
    Return NHANES preparation metadata.
    """

    return copy.deepcopy(
        _load_nhanes_metadata_cached()
    )


@lru_cache(maxsize=1)
def load_schema() -> DatasetSchema:
    """
    Return the original Source/Kaggle 24-feature schema.

    This preserves the existing API used by the synthetic-domain analysis
    scripts. For feature-set-specific schemas, call load_feature_schema().
    """

    metadata = (
        _load_metadata_cached()
    )

    features = metadata[
        "features"
    ]

    return DatasetSchema(
        target=metadata["target"],
        features=tuple(
            features["all_shared"]
        ),
        categorical_features=tuple(
            features["categorical"]
        ),
        numeric_features=tuple(
            features["numeric"]
        ),
        source_only_excluded=tuple(
            features[
                "source_only_excluded"
            ]
        ),
    )


# =============================================================================
# Feature sets
# =============================================================================


FEATURE_SET_EXCLUSIONS = {
    "full_24":
        set(),

    "common_22": {
        "diet_score",
        "waist_to_hip_ratio",
    },

    "common_20": {
        "diet_score",
        "waist_to_hip_ratio",
        "ldl_cholesterol",
        "triglycerides",
    },

    "strict_common_16": {
        "diet_score",
        "waist_to_hip_ratio",
        "ldl_cholesterol",
        "triglycerides",
        "education_level",
        "employment_status",
        "income_level",
        "screen_time_hours_per_day",
    },
}


EXPECTED_FEATURE_COUNTS = {
    "full_24": 24,
    "common_22": 22,
    "common_20": 20,
    "strict_common_16": 16,
}


@lru_cache(maxsize=4)
def load_feature_schema(
    feature_set: FeatureSet,
) -> DatasetSchema:
    """
    Return the schema for one named feature set.

    full_24 uses the canonical Source/Kaggle metadata order.

    The three NHANES-compatible feature sets use the ordered feature_names
    written by prepare_nhanes.py. This makes that metadata the authoritative
    definition of the cross-domain column order and prevents Source/Kaggle and
    NHANES from independently reconstructing the same feature set in different
    orders.
    """

    feature_set = (
        _validate_feature_set(
            feature_set
        )
    )

    base = load_schema()

    if feature_set == "full_24":
        features = base.features
    else:
        nhanes_metadata = (
            _load_nhanes_metadata_cached()
        )

        feature_info = (
            nhanes_metadata[
                "feature_sets"
            ][feature_set]
        )

        features = tuple(
            feature_info[
                "feature_names"
            ]
        )

        unknown_features = [
            feature
            for feature in features
            if feature not in base.features
        ]

        if unknown_features:
            raise ValueError(
                f"{feature_set}: NHANES metadata contains "
                "features that are not in the canonical "
                "Source/Kaggle schema: "
                f"{unknown_features}"
            )

    expected_count = (
        EXPECTED_FEATURE_COUNTS[
            feature_set
        ]
    )

    if len(features) != expected_count:
        raise ValueError(
            f"{feature_set}: expected "
            f"{expected_count} features, "
            f"found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError(
            f"{feature_set}: duplicate feature names "
            "were found in the authoritative feature order."
        )

    categorical = tuple(
        feature
        for feature in features
        if feature
        in base.categorical_features
    )

    numeric = tuple(
        feature
        for feature in features
        if feature
        in base.numeric_features
    )

    if (
        len(categorical)
        + len(numeric)
        != len(features)
    ):
        typed_features = (
            set(categorical)
            | set(numeric)
        )

        untyped = [
            feature
            for feature in features
            if feature not in typed_features
        ]

        raise ValueError(
            f"{feature_set}: features are missing "
            f"type information: {untyped}"
        )

    return DatasetSchema(
        target=base.target,
        features=features,
        categorical_features=
            categorical,
        numeric_features=
            numeric,
        source_only_excluded=
            base.source_only_excluded,
    )


def feature_names(
    feature_set: FeatureSet,
) -> tuple[str, ...]:
    """
    Return features in authoritative model-column order.
    """

    return (
        load_feature_schema(
            feature_set
        )
        .features
    )


# =============================================================================
# Canonical Source / Kaggle datasets
# =============================================================================


@lru_cache(maxsize=2)
def _load_source_kaggle_full_cached(
    domain: Literal[
        "source",
        "kaggle",
    ],
) -> pd.DataFrame:
    path = COMMON_PATHS[
        domain
    ]

    _require_file(
        path,
        preparation_hint=(
            "python3 -m scripts.prepare_data"
        ),
    )

    df = pd.read_csv(
        path
    )

    schema = load_schema()

    expected_columns = list(
        schema.model_columns
    )

    actual_columns = list(
        df.columns
    )

    if (
        actual_columns
        != expected_columns
    ):
        raise ValueError(
            f"{domain} canonical dataset "
            "schema mismatch.\n"
            f"Expected:\n  "
            f"{expected_columns}\n"
            f"Observed:\n  "
            f"{actual_columns}"
        )

    metadata = (
        _load_metadata_cached()
    )

    expected_rows = metadata[
        "datasets"
    ][domain]["rows"]

    if len(df) != expected_rows:
        raise ValueError(
            f"{domain} canonical dataset "
            "row-count mismatch. "
            f"Expected {expected_rows:,}, "
            f"observed {len(df):,}."
        )

    return df


# =============================================================================
# NHANES prepared datasets
# =============================================================================


@lru_cache(maxsize=3)
def _load_nhanes_feature_set_cached(
    feature_set: FeatureSet,
) -> pd.DataFrame:
    if (
        feature_set
        not in NHANES_FEATURE_SETS
    ):
        raise ValueError(
            f"NHANES does not support "
            f"feature set '{feature_set}'."
        )

    path = NHANES_COMMON_PATHS[
        feature_set
    ]

    _require_file(
        path,
        preparation_hint=(
            "python3 -m scripts.prepare_nhanes"
        ),
    )

    df = pd.read_csv(
        path
    )

    schema = (
        load_feature_schema(
            feature_set
        )
    )

    expected_columns = [
        "seqn",
        *schema.features,
        schema.target,
    ]

    actual_columns = list(
        df.columns
    )

    if (
        actual_columns
        != expected_columns
    ):
        raise ValueError(
            f"nhanes/{feature_set} "
            "schema mismatch.\n"
            f"Expected:\n  "
            f"{expected_columns}\n"
            f"Observed:\n  "
            f"{actual_columns}"
        )

    if df["seqn"].isna().any():
        raise ValueError(
            f"nhanes/{feature_set}: "
            "missing SEQN values."
        )

    if df["seqn"].duplicated().any():
        raise ValueError(
            f"nhanes/{feature_set}: "
            "duplicate SEQN values."
        )

    if (
        df[
            list(
                schema.model_columns
            )
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            f"nhanes/{feature_set}: "
            "complete-case dataset "
            "contains missing values."
        )

    metadata = (
        _load_nhanes_metadata_cached()
    )

    info = metadata[
        "feature_sets"
    ][feature_set]

    expected_features = tuple(
        info["feature_names"]
    )

    if (
        expected_features
        != schema.features
    ):
        raise ValueError(
            f"nhanes/{feature_set}: "
            "metadata feature order does "
            "not match data.py feature order."
        )

    expected_rows = int(
        info["rows"]
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"nhanes/{feature_set}: "
            f"expected {expected_rows:,} "
            f"rows, observed {len(df):,}."
        )

    return df


def _load_domain_dataset_internal(
    domain: Domain,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """
    Internal loader.

    Source/Kaggle return feature columns + target.
    NHANES additionally retains SEQN internally.
    """

    if domain in {
        "source",
        "kaggle",
    }:
        full = (
            _load_source_kaggle_full_cached(
                domain
            )
        )

        schema = (
            load_feature_schema(
                feature_set
            )
        )

        return full[
            list(
                schema.model_columns
            )
        ]

    return (
        _load_nhanes_feature_set_cached(
            feature_set
        )
    )


def load_common(
    domain: Domain,
    *,
    feature_set: FeatureSet | None = None,
    include_seqn: bool = False,
) -> pd.DataFrame:
    """
    Load a complete prepared dataset for one domain.

    Defaults
    --------
    source/kaggle:
        full_24

    nhanes:
        strict_common_16

    For Source/Kaggle this preserves the historical behavior of load_common().
    NHANES SEQN is excluded by default so it cannot accidentally enter model
    training; pass include_seqn=True when provenance is needed.
    """

    domain = (
        _validate_domain(
            domain
        )
    )

    resolved = (
        _resolve_feature_set(
            domain,
            feature_set,
        )
    )

    if (
        include_seqn
        and domain != "nhanes"
    ):
        raise ValueError(
            "include_seqn=True is only "
            "valid for the NHANES domain."
        )

    df = (
        _load_domain_dataset_internal(
            domain,
            resolved,
        )
        .copy()
    )

    schema = (
        load_feature_schema(
            resolved
        )
    )

    if domain == "nhanes":
        columns = [
            *(
                ["seqn"]
                if include_seqn
                else []
            ),
            *schema.features,
            schema.target,
        ]

        return df[
            columns
        ].copy()

    return df


# =============================================================================
# Split assignments
# =============================================================================


@lru_cache(maxsize=2)
def _load_source_kaggle_assignments_cached(
    domain: Literal[
        "source",
        "kaggle",
    ],
) -> pd.DataFrame:
    path = (
        SPLIT_ASSIGNMENT_PATHS[
            domain
        ]
    )

    _require_file(
        path,
        preparation_hint=(
            "python3 -m scripts.prepare_data"
        ),
    )

    assignments = pd.read_csv(
        path
    )

    required_columns = {
        "row_index",
        "split",
    }

    if domain == "kaggle":
        required_columns.add(
            "id"
        )

    missing = (
        required_columns
        - set(
            assignments.columns
        )
    )

    if missing:
        raise ValueError(
            f"{domain} split assignments "
            "are missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    common = (
        _load_source_kaggle_full_cached(
            domain
        )
    )

    if len(assignments) != len(common):
        raise ValueError(
            f"{domain} split-assignment "
            "row count does not match "
            "the canonical dataset."
        )

    row_indices = (
        assignments[
            "row_index"
        ]
        .to_numpy()
    )

    expected_indices = np.arange(
        len(common)
    )

    if not np.array_equal(
        row_indices,
        expected_indices,
    ):
        raise ValueError(
            f"{domain} split assignments "
            "do not contain the expected "
            "ordered row indices."
        )

    observed_splits = set(
        assignments[
            "split"
        ].unique()
    )

    if (
        observed_splits
        != set(VALID_SPLITS)
    ):
        raise ValueError(
            f"{domain} split assignments "
            "contain unexpected split "
            f"labels: "
            f"{sorted(observed_splits)}"
        )

    metadata = (
        _load_metadata_cached()
    )

    metadata_splits = metadata[
        "datasets"
    ][domain]["splits"]

    for split in VALID_SPLITS:
        observed_count = int(
            (
                assignments["split"]
                == split
            ).sum()
        )

        expected_count = int(
            metadata_splits[
                split
            ]["rows"]
        )

        if (
            observed_count
            != expected_count
        ):
            raise ValueError(
                f"{domain}/{split} "
                "split-size mismatch. "
                f"Expected "
                f"{expected_count:,}, "
                f"observed "
                f"{observed_count:,}."
            )

    return assignments


@lru_cache(maxsize=1)
def _load_nhanes_assignments_cached() -> pd.DataFrame:
    path = (
        SPLIT_ASSIGNMENT_PATHS[
            "nhanes"
        ]
    )

    _require_file(
        path,
        preparation_hint=(
            "python3 -m scripts.prepare_nhanes"
        ),
    )

    assignments = pd.read_csv(
        path
    )

    required_columns = {
        "row_index",
        "seqn",
        "split",
    }

    missing = (
        required_columns
        - set(
            assignments.columns
        )
    )

    if missing:
        raise ValueError(
            "nhanes split assignments "
            "are missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    primary = (
        _load_nhanes_feature_set_cached(
            "strict_common_16"
        )
    )

    if (
        len(assignments)
        != len(primary)
    ):
        raise ValueError(
            "nhanes split-assignment "
            "row count does not match "
            "STRICT_COMMON_16."
        )

    expected_row_indices = np.arange(
        len(primary)
    )

    observed_row_indices = (
        assignments[
            "row_index"
        ].to_numpy()
    )

    if not np.array_equal(
        observed_row_indices,
        expected_row_indices,
    ):
        raise ValueError(
            "nhanes split assignments "
            "do not contain expected "
            "ordered row indices."
        )

    expected_seqn = (
        primary["seqn"]
        .to_numpy()
    )

    observed_seqn = (
        assignments["seqn"]
        .to_numpy()
    )

    if not np.array_equal(
        observed_seqn,
        expected_seqn,
    ):
        raise ValueError(
            "nhanes split assignments "
            "do not align with "
            "STRICT_COMMON_16 SEQN order."
        )

    observed_splits = set(
        assignments[
            "split"
        ].unique()
    )

    if (
        observed_splits
        != set(VALID_SPLITS)
    ):
        raise ValueError(
            "nhanes split assignments "
            "contain unexpected labels: "
            f"{sorted(observed_splits)}"
        )

    metadata = (
        _load_nhanes_metadata_cached()
    )

    metadata_splits = metadata[
        "primary_splits"
    ]

    for split in VALID_SPLITS:
        observed_count = int(
            (
                assignments["split"]
                == split
            ).sum()
        )

        expected_count = int(
            metadata_splits[
                split
            ]["rows"]
        )

        if (
            observed_count
            != expected_count
        ):
            raise ValueError(
                f"nhanes/{split} "
                "split-size mismatch. "
                f"Expected "
                f"{expected_count:,}, "
                f"observed "
                f"{observed_count:,}."
            )

    return assignments


def load_split_assignments(
    domain: Domain,
) -> pd.DataFrame:
    """
    Return the authoritative row/participant-to-split assignment table.

    Source/Kaggle assignments are row-index based.
    NHANES assignments are participant (SEQN) based.
    """

    domain = (
        _validate_domain(
            domain
        )
    )

    if domain == "nhanes":
        return (
            _load_nhanes_assignments_cached()
            .copy()
        )

    return (
        _load_source_kaggle_assignments_cached(
            domain
        )
        .copy()
    )


# =============================================================================
# Fixed train / validation / test splits
# =============================================================================


def load_split(
    domain: Domain,
    split: Split,
    *,
    include_row_index: bool = False,
    include_id: bool = False,
    include_seqn: bool = False,
    feature_set: FeatureSet | None = None,
) -> pd.DataFrame:
    """
    Load one fixed dataset split.

    Parameters
    ----------
    domain:
        "source", "kaggle", or "nhanes".

    split:
        "train", "val", or "test".

    include_row_index:
        Include the authoritative preparation row index.

    include_id:
        Include the original Kaggle id column. Only valid for Kaggle.

    include_seqn:
        Include NHANES participant identifier SEQN. Only valid for NHANES.

    feature_set:
        One of:
            full_24
            strict_common_16
            common_20
            common_22

        Defaults to full_24 for Source/Kaggle and strict_common_16 for NHANES.

    Notes
    -----
    NHANES sensitivity sets reuse the STRICT_COMMON_16 SEQN split assignment.
    This guarantees that a participant cannot move between train/validation/test
    merely because a different feature set is selected.
    """

    domain = (
        _validate_domain(
            domain
        )
    )

    split = (
        _validate_split(
            split
        )
    )

    resolved = (
        _resolve_feature_set(
            domain,
            feature_set,
        )
    )

    if (
        include_id
        and domain != "kaggle"
    ):
        raise ValueError(
            "include_id=True is only "
            "valid for the Kaggle domain."
        )

    if (
        include_seqn
        and domain != "nhanes"
    ):
        raise ValueError(
            "include_seqn=True is only "
            "valid for the NHANES domain."
        )

    schema = (
        load_feature_schema(
            resolved
        )
    )

    # -------------------------------------------------------------------------
    # Source / Kaggle: preserve historical row-index reconstruction.
    # -------------------------------------------------------------------------

    if domain in {
        "source",
        "kaggle",
    }:
        common = (
            _load_source_kaggle_full_cached(
                domain
            )
        )

        assignments = (
            _load_source_kaggle_assignments_cached(
                domain
            )
        )

        mask = (
            assignments[
                "split"
            ].eq(split)
        )

        row_indices = (
            assignments.loc[
                mask,
                "row_index",
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        df = (
            common.iloc[
                row_indices
            ][
                list(
                    schema.model_columns
                )
            ]
            .reset_index(
                drop=True
            )
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
                assignments.loc[
                    mask,
                    "id",
                ]
                .to_numpy()
            )

            insert_location = (
                1
                if include_row_index
                else 0
            )

            df.insert(
                insert_location,
                "id",
                ids,
            )

        return df

    # -------------------------------------------------------------------------
    # NHANES: split by authoritative SEQN membership.
    # -------------------------------------------------------------------------

    nhanes = (
        _load_nhanes_feature_set_cached(
            resolved
        )
    )

    assignments = (
        _load_nhanes_assignments_cached()
    )

    assignment_map = (
        assignments[
            [
                "row_index",
                "seqn",
                "split",
            ]
        ]
    )

    joined = nhanes.merge(
        assignment_map,
        on="seqn",
        how="left",
        validate="one_to_one",
    )

    if joined["split"].isna().any():
        missing_count = int(
            joined["split"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"nhanes/{resolved}: "
            f"{missing_count:,} rows do not "
            "have authoritative split assignments."
        )

    df = (
        joined.loc[
            joined["split"].eq(
                split
            )
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    output_columns = [
        *(
            ["row_index"]
            if include_row_index
            else []
        ),
        *(
            ["seqn"]
            if include_seqn
            else []
        ),
        *schema.features,
        schema.target,
    ]

    return df[
        output_columns
    ].copy()


# =============================================================================
# Feature / target convenience loader
# =============================================================================


def load_xy(
    domain: Domain,
    split: Split,
    *,
    include_id: bool = False,
    feature_set: FeatureSet | None = None,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Load a fixed split and return predictors (X) and binary target (y).

    No encoding, scaling, imputation, feature selection, or other learned
    preprocessing is performed here.
    """

    domain = (
        _validate_domain(
            domain
        )
    )

    resolved = (
        _resolve_feature_set(
            domain,
            feature_set,
        )
    )

    df = load_split(
        domain,
        split,
        include_id=include_id,
        feature_set=resolved,
    )

    schema = (
        load_feature_schema(
            resolved
        )
    )

    feature_columns = list(
        schema.features
    )

    if include_id:
        feature_columns = [
            "id",
            *feature_columns,
        ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        schema.target
    ].copy()

    return X, y


# =============================================================================
# Split summaries
# =============================================================================


def split_summary(
    domain: Domain,
    split: Split,
    *,
    feature_set: FeatureSet | None = None,
) -> dict:
    """
    Return a small summary of a prepared split.
    """

    domain = (
        _validate_domain(
            domain
        )
    )

    resolved = (
        _resolve_feature_set(
            domain,
            feature_set,
        )
    )

    df = load_split(
        domain,
        split,
        feature_set=resolved,
    )

    schema = (
        load_feature_schema(
            resolved
        )
    )

    counts = (
        df[
            schema.target
        ]
        .value_counts()
        .sort_index()
    )

    proportions = (
        df[
            schema.target
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    return {
        "domain":
            domain,

        "split":
            split,

        "feature_set":
            resolved,

        "rows":
            len(df),

        "features":
            len(
                schema.features
            ),

        "class_counts": {
            int(label):
                int(count)
            for label, count
            in counts.items()
        },

        "class_proportions": {
            int(label):
                float(proportion)
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
    print("BASE DATA SCHEMA")
    print("=" * 80)

    print(
        f"Target: "
        f"{schema.target}"
    )

    print(
        f"Full shared features: "
        f"{len(schema.features)}"
    )

    print(
        f"Categorical: "
        f"{len(schema.categorical_features)}"
    )

    print(
        f"Numeric: "
        f"{len(schema.numeric_features)}"
    )

    print(
        "\nFeature sets:"
    )

    for feature_set in (
        "full_24",
        "strict_common_16",
        "common_20",
        "common_22",
    ):
        feature_schema = (
            load_feature_schema(
                feature_set
            )
        )

        print(
            f"  {feature_set:<18} "
            f"features="
            f"{len(feature_schema.features):>2}"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FIXED SPLITS"
    )

    print(
        "=" * 80
    )

    smoke_configs = (
        (
            "source",
            "full_24",
        ),
        (
            "kaggle",
            "full_24",
        ),
        (
            "nhanes",
            "strict_common_16",
        ),
    )

    for (
        domain,
        feature_set,
    ) in smoke_configs:
        print(
            f"\n{display_domain_name(domain, include_id=True)} "
            f"[{feature_set}]"
        )

        for split in VALID_SPLITS:
            summary = (
                split_summary(
                    domain,  # type: ignore[arg-type]
                    split,   # type: ignore[arg-type]
                    feature_set=feature_set,  # type: ignore[arg-type]
                )
            )

            positive = (
                summary[
                    "class_proportions"
                ]
                .get(
                    1,
                    0.0,
                )
            )

            print(
                f"  {split:<5} "
                f"rows="
                f"{summary['rows']:>8,}  "
                f"positive="
                f"{positive:.6f}"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "THREE-DOMAIN FEATURE-ORDER TEST"
    )

    print(
        "=" * 80
    )

    strict_columns = None

    for domain in VALID_DOMAINS:
        X, y = load_xy(
            domain,  # type: ignore[arg-type]
            "train",
            feature_set=
                "strict_common_16",
        )

        print(
            f"{display_domain_name(domain, include_id=True):<32} "
            f"X={X.shape} "
            f"y={y.shape} "
            f"target_dtype={y.dtype}"
        )

        columns = tuple(
            X.columns
        )

        if strict_columns is None:
            strict_columns = columns
        elif columns != strict_columns:
            raise RuntimeError(
                "STRICT_COMMON_16 feature "
                "order differs across domains."
            )

    print(
        "\nSTRICT_COMMON_16 "
        "feature order: OK"
    )

    # Explicit Kaggle id test.
    X_with_id, _ = load_xy(
        "kaggle",
        "test",
        include_id=True,
    )

    print(
        "\nPlayground S5E12 test (kaggle) with explicit id: "
        f"{X_with_id.shape}"
    )

    print(
        f"First columns: "
        f"{list(X_with_id.columns[:4])}"
    )

    # Explicit NHANES SEQN test.
    nhanes_with_seqn = (
        load_split(
            "nhanes",
            "test",
            include_seqn=True,
            feature_set=
                "strict_common_16",
        )
    )

    print(
        "\nNHANES 2015–2016 test (nhanes) with explicit SEQN: "
        f"{nhanes_with_seqn.shape}"
    )

    print(
        f"First columns: "
        f"{list(nhanes_with_seqn.columns[:4])}"
    )


if __name__ == "__main__":
    main()
