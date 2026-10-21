from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import (
    KAGGLE_RAW_PATH as KAGGLE_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT as ROOT,
    SOURCE_RAW_PATH as SOURCE_PATH,
    SPLITS_DIR,
)

# =============================================================================
# Project paths
# =============================================================================

SOURCE_COMMON_PATH = PROCESSED_DIR / "source_common.csv"
KAGGLE_COMMON_PATH = PROCESSED_DIR / "kaggle_common.csv"
KAGGLE_IDS_PATH = PROCESSED_DIR / "kaggle_ids.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

SOURCE_SPLIT_ASSIGNMENTS_PATH = SPLITS_DIR / "source_split_assignments.csv"
KAGGLE_SPLIT_ASSIGNMENTS_PATH = SPLITS_DIR / "kaggle_split_assignments.csv"


# =============================================================================
# Experimental configuration
# =============================================================================

SEED = 42

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

TARGET = "diagnosed_diabetes"
KAGGLE_ID = "id"


# =============================================================================
# Frozen shared-feature schema
#
# These are the 24 predictors shared between the source dataset and S5E12.
# Keeping this explicit prevents the experiment from silently changing if one
# of the raw input files is replaced or modified.
# =============================================================================

PREPARATION_VERSION = 1

CATEGORICAL_FEATURES = [
    "education_level",
    "employment_status",
    "ethnicity",
    "gender",
    "income_level",
    "smoking_status",
]

NUMERIC_FEATURES = [
    "age",
    "alcohol_consumption_per_week",
    "bmi",
    "cardiovascular_history",
    "cholesterol_total",
    "diastolic_bp",
    "diet_score",
    "family_history_diabetes",
    "hdl_cholesterol",
    "heart_rate",
    "hypertension_history",
    "ldl_cholesterol",
    "physical_activity_minutes_per_week",
    "screen_time_hours_per_day",
    "sleep_hours_per_day",
    "systolic_bp",
    "triglycerides",
    "waist_to_hip_ratio",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# Variables present in the source dataset but intentionally absent from the
# common cross-domain feature space.
SOURCE_ONLY_FEATURES = [
    "diabetes_risk_score",
    "diabetes_stage",
    "glucose_fasting",
    "glucose_postprandial",
    "hba1c",
    "insulin_level",
]


# Freeze the categorical vocabulary observed during the initial inspection.
# We do not encode these values here. Model-specific preprocessing will happen
# later using training data only.
EXPECTED_CATEGORIES = {
    "education_level": {
        "Graduate",
        "Highschool",
        "No formal",
        "Postgraduate",
    },
    "employment_status": {
        "Employed",
        "Retired",
        "Student",
        "Unemployed",
    },
    "ethnicity": {
        "Asian",
        "Black",
        "Hispanic",
        "Other",
        "White",
    },
    "gender": {
        "Female",
        "Male",
        "Other",
    },
    "income_level": {
        "High",
        "Low",
        "Lower-Middle",
        "Middle",
        "Upper-Middle",
    },
    "smoking_status": {
        "Current",
        "Former",
        "Never",
    },
}


# =============================================================================
# Utility functions
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def sha256_file(path: Path) -> str:
    """
    Compute a SHA-256 hash of an input file.

    Recording this in metadata allows us to verify later that experiments used
    the exact same raw dataset files.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def ensure_output_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Loading and schema validation
# =============================================================================


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print_header("LOADING RAW DATA")

    print(f"Source: {SOURCE_PATH}")
    source = pd.read_csv(SOURCE_PATH)

    print(f"Kaggle: {KAGGLE_PATH}")
    kaggle = pd.read_csv(KAGGLE_PATH)

    print(f"\nSource shape: {source.shape}")
    print(f"Kaggle shape: {kaggle.shape}")

    return source, kaggle


def validate_exact_columns(
    df: pd.DataFrame,
    expected_columns: set[str],
    dataset_name: str,
) -> None:
    actual_columns = set(df.columns)

    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)

    if missing or unexpected:
        messages = [f"{dataset_name} schema mismatch."]

        if missing:
            messages.append(
                f"Missing columns: {', '.join(missing)}"
            )

        if unexpected:
            messages.append(
                f"Unexpected columns: {', '.join(unexpected)}"
            )

        raise ValueError("\n".join(messages))


def validate_raw_schema(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
) -> None:
    print_header("VALIDATING RAW SCHEMA")

    expected_source_columns = set(
        FEATURES
        + SOURCE_ONLY_FEATURES
        + [TARGET]
    )

    expected_kaggle_columns = set(
        FEATURES
        + [KAGGLE_ID, TARGET]
    )

    validate_exact_columns(
        source,
        expected_source_columns,
        "Source dataset",
    )

    validate_exact_columns(
        kaggle,
        expected_kaggle_columns,
        "Kaggle dataset",
    )

    print("Source schema: OK")
    print("Kaggle schema: OK")


# =============================================================================
# Canonical shared-feature representation
# =============================================================================


def create_common_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract only the canonical 24 shared predictors and target.

    id and source-only variables are intentionally excluded.
    """
    return df[FEATURES + [TARGET]].copy()


def normalize_common_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Normalize representation without learning anything from the data.

    This function does NOT:
      - standardize values
      - one-hot encode categories
      - impute missing data
      - rebalance classes

    Those operations, where required, belong inside experiment-specific
    preprocessing fitted on training data only.
    """
    df = df.copy()

    for feature in NUMERIC_FEATURES:
        try:
            df[feature] = pd.to_numeric(
                df[feature],
                errors="raise",
            )
        except Exception as exc:
            raise ValueError(
                f"{dataset_name}: numeric conversion failed "
                f"for feature '{feature}'."
            ) from exc

    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].astype("string")

    try:
        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="raise",
        )
    except Exception as exc:
        raise ValueError(
            f"{dataset_name}: target could not be converted "
            "to numeric values."
        ) from exc

    if df[TARGET].isna().any():
        raise ValueError(
            f"{dataset_name}: target contains missing values."
        )

    target_values = set(df[TARGET].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: unexpected target values: "
            f"{sorted(target_values)}"
        )

    df[TARGET] = df[TARGET].astype("int64")

    return df


# =============================================================================
# Data validation
# =============================================================================


def validate_missing_values(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    missing = df[FEATURES + [TARGET]].isna().sum()

    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"{dataset_name} contains missing values:\n"
            f"{missing.to_string()}"
        )


def validate_numeric_values(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    numeric_values = df[NUMERIC_FEATURES].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"{dataset_name} contains NaN or infinite "
            "numeric values."
        )


def validate_category_vocabularies(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    for feature in CATEGORICAL_FEATURES:
        observed = set(
            df[feature]
            .dropna()
            .astype(str)
            .unique()
        )

        expected = EXPECTED_CATEGORIES[feature]

        missing_categories = sorted(expected - observed)
        unexpected_categories = sorted(observed - expected)

        if missing_categories or unexpected_categories:
            message = [
                f"{dataset_name}: categorical vocabulary mismatch "
                f"for '{feature}'."
            ]

            if missing_categories:
                message.append(
                    "Missing expected categories: "
                    + ", ".join(missing_categories)
                )

            if unexpected_categories:
                message.append(
                    "Unexpected categories: "
                    + ", ".join(unexpected_categories)
                )

            raise ValueError("\n".join(message))


def validate_common_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    validate_missing_values(df, dataset_name)
    validate_numeric_values(df, dataset_name)
    validate_category_vocabularies(df, dataset_name)

    target_values = set(df[TARGET].unique())

    if target_values != {0, 1}:
        raise ValueError(
            f"{dataset_name}: expected binary target {{0, 1}}, "
            f"observed {sorted(target_values)}."
        )


# =============================================================================
# Duplicate diagnostics
# =============================================================================


def duplicate_statistics(
    df: pd.DataFrame,
) -> dict[str, int]:
    """
    Report duplicates without removing them.

    Duplicates may themselves be an artifact of the synthetic-data generation
    process, so the preparation stage should not silently discard them.
    """

    exact_duplicates = int(
        df.duplicated(
            subset=FEATURES + [TARGET],
            keep="first",
        ).sum()
    )

    feature_duplicates = int(
        df.duplicated(
            subset=FEATURES,
            keep="first",
        ).sum()
    )

    duplicate_feature_mask = df.duplicated(
        subset=FEATURES,
        keep=False,
    )

    rows_in_duplicate_feature_groups = int(
        duplicate_feature_mask.sum()
    )

    conflicting_label_groups = 0

    # Only perform the relatively expensive groupby if duplicate feature
    # vectors actually exist.
    if rows_in_duplicate_feature_groups > 0:
        duplicates = df.loc[
            duplicate_feature_mask,
            FEATURES + [TARGET],
        ]

        label_counts = (
            duplicates
            .groupby(
                FEATURES,
                dropna=False,
                sort=False,
            )[TARGET]
            .nunique()
        )

        conflicting_label_groups = int(
            (label_counts > 1).sum()
        )

    return {
        "exact_duplicate_rows_beyond_first":
            exact_duplicates,
        "feature_duplicate_rows_beyond_first":
            feature_duplicates,
        "rows_in_duplicate_feature_groups":
            rows_in_duplicate_feature_groups,
        "conflicting_label_feature_groups":
            conflicting_label_groups,
    }


def print_duplicate_statistics(
    dataset_name: str,
    stats: dict[str, int],
) -> None:
    print(f"\n{dataset_name}:")

    print(
        "  Exact duplicate rows beyond first: "
        f"{stats['exact_duplicate_rows_beyond_first']:,}"
    )

    print(
        "  Duplicate feature rows beyond first: "
        f"{stats['feature_duplicate_rows_beyond_first']:,}"
    )

    print(
        "  Rows participating in duplicate feature groups: "
        f"{stats['rows_in_duplicate_feature_groups']:,}"
    )

    print(
        "  Duplicate feature groups with conflicting labels: "
        f"{stats['conflicting_label_feature_groups']:,}"
    )


# =============================================================================
# Train / validation / test splitting
# =============================================================================


def create_stratified_split_indices(
    df: pd.DataFrame,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Create deterministic 70/15/15 stratified row-index splits.

    Split sizes are calculated explicitly so datasets such as 700,000 rows
    produce exactly 490,000 / 105,000 / 105,000 samples.
    """

    if not np.isclose(
        TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION,
        1.0,
    ):
        raise ValueError(
            "Train/validation/test fractions must sum to 1."
        )

    total_rows = len(df)

    train_size = int(round(total_rows * TRAIN_FRACTION))
    val_size = int(round(total_rows * VAL_FRACTION))

    # Assign any rounding remainder to the test set.
    test_size = total_rows - train_size - val_size

    indices = np.arange(total_rows)

    train_indices, remaining_indices = train_test_split(
        indices,
        train_size=train_size,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df[TARGET],
    )

    remaining_targets = df.iloc[remaining_indices][TARGET]

    val_indices, test_indices = train_test_split(
        remaining_indices,
        train_size=val_size,
        test_size=test_size,
        random_state=seed + 1,
        stratify=remaining_targets,
    )

    return {
        "train": np.asarray(train_indices),
        "val": np.asarray(val_indices),
        "test": np.asarray(test_indices),
    }


def validate_split_indices(
    split_indices: dict[str, np.ndarray],
    total_rows: int,
    dataset_name: str,
) -> None:
    train = set(split_indices["train"].tolist())
    val = set(split_indices["val"].tolist())
    test = set(split_indices["test"].tolist())

    if train & val:
        raise RuntimeError(
            f"{dataset_name}: train and validation splits overlap."
        )

    if train & test:
        raise RuntimeError(
            f"{dataset_name}: train and test splits overlap."
        )

    if val & test:
        raise RuntimeError(
            f"{dataset_name}: validation and test splits overlap."
        )

    union = train | val | test

    expected = set(range(total_rows))

    if union != expected:
        missing = len(expected - union)
        unexpected = len(union - expected)

        raise RuntimeError(
            f"{dataset_name}: split coverage error. "
            f"Missing={missing}, unexpected={unexpected}."
        )


# =============================================================================
# Split summaries
# =============================================================================


def class_distribution(
    target: pd.Series,
) -> dict:
    counts = target.value_counts().sort_index()
    proportions = target.value_counts(
        normalize=True
    ).sort_index()

    return {
        "rows": int(len(target)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in counts.items()
        },
        "class_proportions": {
            str(int(label)): float(proportion)
            for label, proportion in proportions.items()
        },
    }


def summarize_splits(
    df: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
) -> dict:
    summary = {}

    for split_name, indices in split_indices.items():
        target = df.iloc[indices][TARGET]

        summary[split_name] = class_distribution(target)

    return summary


def print_split_summary(
    dataset_name: str,
    summary: dict,
) -> None:
    print(f"\n{dataset_name}")

    for split_name in ["train", "val", "test"]:
        info = summary[split_name]

        positive_fraction = (
            info["class_proportions"].get("1", 0.0)
        )

        print(
            f"  {split_name:<5} "
            f"rows={info['rows']:>8,}  "
            f"positive={positive_fraction:.6f}"
        )


# =============================================================================
# Saving prepared datasets and splits
# =============================================================================


def save_common_datasets(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    raw_kaggle: pd.DataFrame,
) -> None:
    print_header("SAVING CANONICAL DATASETS")

    source.to_csv(
        SOURCE_COMMON_PATH,
        index=False,
    )

    kaggle.to_csv(
        KAGGLE_COMMON_PATH,
        index=False,
    )

    kaggle_ids = pd.DataFrame(
        {
            "row_index": np.arange(len(raw_kaggle)),
            KAGGLE_ID: raw_kaggle[KAGGLE_ID].to_numpy(),
        }
    )

    kaggle_ids.to_csv(
        KAGGLE_IDS_PATH,
        index=False,
    )

    print(SOURCE_COMMON_PATH.relative_to(ROOT))
    print(KAGGLE_COMMON_PATH.relative_to(ROOT))
    print(KAGGLE_IDS_PATH.relative_to(ROOT))


def save_split_assignments(
    split_indices: dict[str, np.ndarray],
    path: Path,
    kaggle_ids: pd.Series | None = None,
) -> None:
    split_labels = np.empty(
        sum(len(indices) for indices in split_indices.values()),
        dtype=object,
    )

    for split_name, indices in split_indices.items():
        split_labels[indices] = split_name

    assignment = pd.DataFrame(
        {
            "row_index": np.arange(len(split_labels)),
            "split": split_labels,
        }
    )

    if kaggle_ids is not None:
        assignment.insert(
            1,
            KAGGLE_ID,
            kaggle_ids.to_numpy(),
        )

    assignment.to_csv(
        path,
        index=False,
    )


def save_split_csvs(
    dataset_name: str,
    df: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
) -> None:
    """
    Save convenience CSV files for each fixed split.

    The canonical common dataset plus split-assignment file is sufficient to
    reconstruct these, but saving explicit split CSVs makes model development
    simpler and guarantees every model can consume exactly the same rows.
    """

    for split_name, indices in split_indices.items():
        output_path = (
            SPLITS_DIR
            / f"{dataset_name}_{split_name}.csv"
        )

        split_df = (
            df.iloc[indices]
            .reset_index(drop=True)
        )

        split_df.to_csv(
            output_path,
            index=False,
        )


# =============================================================================
# Metadata
# =============================================================================


def build_metadata(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    source_duplicates: dict,
    kaggle_duplicates: dict,
    source_split_summary: dict,
    kaggle_split_summary: dict,
) -> dict:
    return {
        "schema_version": PREPARATION_VERSION,
        "random_seed": SEED,
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "target": TARGET,
        "features": {
            "all_shared": FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "source_only_excluded":
                SOURCE_ONLY_FEATURES,
        },
        "categorical_vocabularies": {
            feature: sorted(values)
            for feature, values
            in EXPECTED_CATEGORIES.items()
        },
        "raw_files": {
            "source": {
                "path": str(
                    SOURCE_PATH.relative_to(ROOT)
                ),
                "sha256": sha256_file(SOURCE_PATH),
            },
            "kaggle": {
                "path": str(
                    KAGGLE_PATH.relative_to(ROOT)
                ),
                "sha256": sha256_file(KAGGLE_PATH),
            },
        },
        "datasets": {
            "source": {
                "rows": int(len(source)),
                "columns": int(source.shape[1]),
                "target_distribution":
                    class_distribution(source[TARGET]),
                "duplicates":
                    source_duplicates,
                "splits":
                    source_split_summary,
            },
            "kaggle": {
                "rows": int(len(kaggle)),
                "columns": int(kaggle.shape[1]),
                "target_distribution":
                    class_distribution(kaggle[TARGET]),
                "duplicates":
                    kaggle_duplicates,
                "splits":
                    kaggle_split_summary,
            },
        },
        "outputs": {
            "source_common": str(
                SOURCE_COMMON_PATH.relative_to(ROOT)
            ),
            "kaggle_common": str(
                KAGGLE_COMMON_PATH.relative_to(ROOT)
            ),
            "kaggle_ids": str(
                KAGGLE_IDS_PATH.relative_to(ROOT)
            ),
            "source_split_assignments": str(
                SOURCE_SPLIT_ASSIGNMENTS_PATH.relative_to(ROOT)
            ),
            "kaggle_split_assignments": str(
                KAGGLE_SPLIT_ASSIGNMENTS_PATH.relative_to(ROOT)
            ),
        },
    }


def save_metadata(metadata: dict) -> None:
    with METADATA_PATH.open("w") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    print_header("METADATA")
    print(METADATA_PATH.relative_to(ROOT))


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_output_directories()

    # -------------------------------------------------------------------------
    # Load and validate raw inputs.
    # -------------------------------------------------------------------------

    source_raw, kaggle_raw = load_datasets()

    validate_raw_schema(
        source_raw,
        kaggle_raw,
    )

    # -------------------------------------------------------------------------
    # Create the common cross-domain representation.
    # -------------------------------------------------------------------------

    source = create_common_dataset(source_raw)
    kaggle = create_common_dataset(kaggle_raw)

    source = normalize_common_dataset(
        source,
        "Source dataset",
    )

    kaggle = normalize_common_dataset(
        kaggle,
        "Kaggle dataset",
    )

    print_header("VALIDATING CANONICAL DATA")

    validate_common_dataset(
        source,
        "Source dataset",
    )

    validate_common_dataset(
        kaggle,
        "Kaggle dataset",
    )

    print("Source canonical dataset: OK")
    print("Kaggle canonical dataset: OK")

    # -------------------------------------------------------------------------
    # Duplicate diagnostics.
    # -------------------------------------------------------------------------

    print_header("DUPLICATE DIAGNOSTICS")

    source_duplicates = duplicate_statistics(source)
    kaggle_duplicates = duplicate_statistics(kaggle)

    print_duplicate_statistics(
        "Source",
        source_duplicates,
    )

    print_duplicate_statistics(
        "Kaggle",
        kaggle_duplicates,
    )

    # -------------------------------------------------------------------------
    # Create deterministic stratified splits.
    # -------------------------------------------------------------------------

    print_header("CREATING FIXED STRATIFIED SPLITS")

    source_split_indices = create_stratified_split_indices(
        source,
        SEED,
    )

    kaggle_split_indices = create_stratified_split_indices(
        kaggle,
        SEED,
    )

    validate_split_indices(
        source_split_indices,
        len(source),
        "Source dataset",
    )

    validate_split_indices(
        kaggle_split_indices,
        len(kaggle),
        "Kaggle dataset",
    )

    source_split_summary = summarize_splits(
        source,
        source_split_indices,
    )

    kaggle_split_summary = summarize_splits(
        kaggle,
        kaggle_split_indices,
    )

    print_split_summary(
        "Source",
        source_split_summary,
    )

    print_split_summary(
        "Kaggle",
        kaggle_split_summary,
    )

    # -------------------------------------------------------------------------
    # Save canonical datasets.
    # -------------------------------------------------------------------------

    save_common_datasets(
        source,
        kaggle,
        kaggle_raw,
    )

    # -------------------------------------------------------------------------
    # Save the exact split assignments.
    # -------------------------------------------------------------------------

    print_header("SAVING SPLIT ASSIGNMENTS")

    save_split_assignments(
        source_split_indices,
        SOURCE_SPLIT_ASSIGNMENTS_PATH,
    )

    save_split_assignments(
        kaggle_split_indices,
        KAGGLE_SPLIT_ASSIGNMENTS_PATH,
        kaggle_ids=kaggle_raw[KAGGLE_ID],
    )

    print(
        SOURCE_SPLIT_ASSIGNMENTS_PATH.relative_to(ROOT)
    )

    print(
        KAGGLE_SPLIT_ASSIGNMENTS_PATH.relative_to(ROOT)
    )

    # -------------------------------------------------------------------------
    # Also save convenient per-split CSVs.
    # -------------------------------------------------------------------------

    print_header("SAVING SPLIT DATASETS")

    save_split_csvs(
        "source",
        source,
        source_split_indices,
    )

    save_split_csvs(
        "kaggle",
        kaggle,
        kaggle_split_indices,
    )

    for dataset_name in ["source", "kaggle"]:
        for split_name in ["train", "val", "test"]:
            print(
                (
                    SPLITS_DIR
                    / f"{dataset_name}_{split_name}.csv"
                ).relative_to(ROOT)
            )

    # -------------------------------------------------------------------------
    # Reproducibility metadata.
    # -------------------------------------------------------------------------

    metadata = build_metadata(
        source,
        kaggle,
        source_duplicates,
        kaggle_duplicates,
        source_split_summary,
        kaggle_split_summary,
    )

    save_metadata(metadata)

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    print_header("PREPARATION COMPLETE")

    print(f"Shared predictors: {len(FEATURES)}")
    print(
        f"  Categorical: {len(CATEGORICAL_FEATURES)}"
    )
    print(
        f"  Numeric:     {len(NUMERIC_FEATURES)}"
    )

    print(
        "\nSource-only variables excluded from "
        "cross-domain experiments:"
    )

    for feature in SOURCE_ONLY_FEATURES:
        print(f"  {feature}")

    print(
        "\nNo scaling, encoding, imputation, resampling, "
        "or feature selection was performed."
    )

    print(
        "Model-specific preprocessing must be fitted "
        "using training data only."
    )


if __name__ == "__main__":
    main()