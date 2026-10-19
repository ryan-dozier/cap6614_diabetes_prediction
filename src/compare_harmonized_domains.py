from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = ROOT / "data/processed/source_common.csv"
KAGGLE_PATH = ROOT / "data/processed/kaggle_common.csv"
NHANES_PATH = ROOT / "data/processed/nhanes_harmonized.csv"

OUTPUT_DIR = ROOT / "results/harmonized_domain_comparison"

NUMERIC_SUMMARY_PATH = OUTPUT_DIR / "numeric_summary.csv"
CATEGORICAL_FREQUENCIES_PATH = OUTPUT_DIR / "categorical_frequencies.csv"
CATEGORICAL_VALUES_PATH = OUTPUT_DIR / "categorical_values.csv"
TARGET_SUMMARY_PATH = OUTPUT_DIR / "target_summary.csv"
FEATURE_COMPATIBILITY_PATH = OUTPUT_DIR / "feature_compatibility.csv"
COMPLETE_CASE_SUMMARY_PATH = OUTPUT_DIR / "complete_case_summary.csv"


# =============================================================================
# Shared three-domain feature space
# =============================================================================

TARGET = "diagnosed_diabetes"

CATEGORICAL_FEATURES = (
    "education_level",
    "employment_status",
    "ethnicity",
    "gender",
    "income_level",
    "smoking_status",
)

NUMERIC_FEATURES = (
    "age",
    "alcohol_consumption_per_week",
    "bmi",
    "cardiovascular_history",
    "cholesterol_total",
    "diastolic_bp",
    "family_history_diabetes",
    "hdl_cholesterol",
    "heart_rate",
    "hypertension_history",
    "physical_activity_minutes_per_week",
    "screen_time_hours_per_day",
    "sleep_hours_per_day",
    "systolic_bp",
)

COMMON_20 = (
    "age",
    "alcohol_consumption_per_week",
    "bmi",
    "cardiovascular_history",
    "cholesterol_total",
    "diastolic_bp",
    "education_level",
    "employment_status",
    "ethnicity",
    "family_history_diabetes",
    "gender",
    "hdl_cholesterol",
    "heart_rate",
    "hypertension_history",
    "income_level",
    "physical_activity_minutes_per_week",
    "screen_time_hours_per_day",
    "sleep_hours_per_day",
    "smoking_status",
    "systolic_bp",
)

DOMAINS = (
    "source",
    "kaggle",
    "nhanes",
)


# =============================================================================
# Utilities
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required prepared dataset:\n"
            f"  {path.relative_to(ROOT)}"
        )


def load_datasets() -> dict[str, pd.DataFrame]:
    print_header("LOADING PREPARED DATASETS")

    paths = {
        "source": SOURCE_PATH,
        "kaggle": KAGGLE_PATH,
        "nhanes": NHANES_PATH,
    }

    datasets = {}

    for domain, path in paths.items():
        require_file(path)

        df = pd.read_csv(path)

        datasets[domain] = df

        print(
            f"{domain:<8} "
            f"rows={len(df):>8,}  "
            f"columns={len(df.columns):>3}"
        )

    return datasets


# =============================================================================
# Schema validation
# =============================================================================


def validate_schema(
    datasets: dict[str, pd.DataFrame],
) -> None:
    print_header("SCHEMA VALIDATION")

    required = set(COMMON_20) | {TARGET}

    for domain in DOMAINS:
        df = datasets[domain]

        missing = sorted(
            required - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{domain} is missing required columns:\n"
                + "\n".join(
                    f"  {column}"
                    for column in missing
                )
            )

        duplicate_columns = (
            df.columns[
                df.columns.duplicated()
            ]
            .tolist()
        )

        if duplicate_columns:
            raise ValueError(
                f"{domain} contains duplicate column names: "
                f"{duplicate_columns}"
            )

        print(
            f"{domain:<8} "
            f"COMMON_20 + target: OK"
        )


# =============================================================================
# Target comparison
# =============================================================================


def build_target_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("TARGET SUMMARY")

    rows = []

    for domain in DOMAINS:
        series = pd.to_numeric(
            datasets[domain][TARGET],
            errors="coerce",
        )

        valid = series.dropna()

        invalid = valid.loc[
            ~valid.isin([0, 1])
        ]

        if not invalid.empty:
            raise ValueError(
                f"{domain}: target contains values outside {{0, 1}}: "
                f"{sorted(invalid.unique().tolist())}"
            )

        negatives = int(
            valid.eq(0).sum()
        )

        positives = int(
            valid.eq(1).sum()
        )

        row = {
            "domain": domain,
            "rows": len(series),
            "target_nonmissing": len(valid),
            "target_missing": int(
                series.isna().sum()
            ),
            "negative_rows": negatives,
            "positive_rows": positives,
            "positive_fraction": (
                positives / len(valid)
                if len(valid)
                else np.nan
            ),
        }

        rows.append(row)

        print(
            f"{domain:<8} "
            f"n={len(valid):>8,}  "
            f"positive={positives:>8,}  "
            f"rate={row['positive_fraction']:.4f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# Complete-case comparison
# =============================================================================


def build_complete_case_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("COMMON_20 COMPLETE-CASE SUMMARY")

    required = [
        *COMMON_20,
        TARGET,
    ]

    rows = []

    for domain in DOMAINS:
        df = datasets[domain]

        complete = df.dropna(
            subset=required
        )

        positive_fraction = (
            float(
                pd.to_numeric(
                    complete[TARGET],
                    errors="raise",
                ).mean()
            )
            if len(complete)
            else np.nan
        )

        row = {
            "domain": domain,
            "rows_total": len(df),
            "rows_complete_common_20":
                len(complete),
            "rows_removed":
                len(df) - len(complete),
            "retained_fraction": (
                len(complete) / len(df)
                if len(df)
                else np.nan
            ),
            "positive_fraction_complete":
                positive_fraction,
        }

        rows.append(row)

        print(
            f"{domain:<8} "
            f"complete={len(complete):>8,} / "
            f"{len(df):>8,}  "
            f"retained={row['retained_fraction']:.2%}  "
            f"positive={positive_fraction:.4f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# Numeric comparison
# =============================================================================


def describe_numeric_series(
    series: pd.Series,
) -> dict:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        return {
            "count": 0,
            "missing": len(series),
            "missing_fraction": 1.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    quantiles = valid.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    return {
        "count": int(valid.size),
        "missing": int(
            numeric.isna().sum()
        ),
        "missing_fraction": float(
            numeric.isna().mean()
        ),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(valid.max()),
    }


def build_numeric_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("NUMERIC FEATURE SUMMARY")

    rows = []

    for feature in NUMERIC_FEATURES:
        print(f"\n{feature}")

        for domain in DOMAINS:
            stats = describe_numeric_series(
                datasets[domain][feature]
            )

            rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    **stats,
                }
            )

            print(
                f"  {domain:<8} "
                f"n={stats['count']:>8,}  "
                f"mean={stats['mean']:>10.4f}  "
                f"std={stats['std']:>10.4f}  "
                f"p05={stats['p05']:>10.4f}  "
                f"p50={stats['median']:>10.4f}  "
                f"p95={stats['p95']:>10.4f}"
            )

    return pd.DataFrame(rows)


# =============================================================================
# Categorical comparison
# =============================================================================


def build_categorical_values(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("CATEGORICAL VALUE COMPATIBILITY")

    rows = []

    for feature in CATEGORICAL_FEATURES:
        print(f"\n{feature}")

        domain_values = {}

        for domain in DOMAINS:
            values = sorted(
                datasets[domain][feature]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            domain_values[domain] = set(
                values
            )

            print(
                f"  {domain:<8} "
                f"{values}"
            )

        all_values = set().union(
            *domain_values.values()
        )

        common_values = set.intersection(
            *domain_values.values()
        )

        for value in sorted(all_values):
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "source_present":
                        value in domain_values["source"],
                    "kaggle_present":
                        value in domain_values["kaggle"],
                    "nhanes_present":
                        value in domain_values["nhanes"],
                    "present_all_domains":
                        value in common_values,
                }
            )

    return pd.DataFrame(rows)


def build_categorical_frequencies(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("CATEGORICAL FREQUENCIES")

    rows = []

    for feature in CATEGORICAL_FEATURES:
        print(f"\n{feature}")

        for domain in DOMAINS:
            series = (
                datasets[domain][feature]
                .astype("string")
            )

            nonmissing = series.dropna()

            counts = (
                nonmissing
                .value_counts()
                .sort_index()
            )

            print(
                f"  {domain:<8} "
                f"n={len(nonmissing):,}"
            )

            for value, count in counts.items():
                fraction = (
                    count / len(nonmissing)
                    if len(nonmissing)
                    else np.nan
                )

                rows.append(
                    {
                        "domain": domain,
                        "feature": feature,
                        "value": str(value),
                        "count": int(count),
                        "fraction": float(fraction),
                        "missing": int(
                            series.isna().sum()
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Feature-level compatibility report
# =============================================================================


def pooled_standardized_difference(
    first: pd.Series,
    second: pd.Series,
) -> float:
    first = pd.to_numeric(
        first,
        errors="coerce",
    ).dropna()

    second = pd.to_numeric(
        second,
        errors="coerce",
    ).dropna()

    if first.empty or second.empty:
        return np.nan

    std_first = first.std()
    std_second = second.std()

    pooled_std = np.sqrt(
        (
            std_first ** 2
            + std_second ** 2
        )
        / 2.0
    )

    if not np.isfinite(pooled_std):
        return np.nan

    if pooled_std == 0:
        if first.mean() == second.mean():
            return 0.0

        return np.inf

    return float(
        (
            second.mean()
            - first.mean()
        )
        / pooled_std
    )


def categorical_total_variation(
    first: pd.Series,
    second: pd.Series,
) -> float:
    first = first.dropna().astype(str)
    second = second.dropna().astype(str)

    if first.empty or second.empty:
        return np.nan

    categories = sorted(
        set(first.unique())
        | set(second.unique())
    )

    first_freq = (
        first.value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    second_freq = (
        second.value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    return float(
        0.5
        * np.abs(
            first_freq.values
            - second_freq.values
        ).sum()
    )


def build_feature_compatibility(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("FEATURE COMPATIBILITY SUMMARY")

    rows = []

    for feature in COMMON_20:
        is_categorical = (
            feature in CATEGORICAL_FEATURES
        )

        if is_categorical:
            source_kaggle = categorical_total_variation(
                datasets["source"][feature],
                datasets["kaggle"][feature],
            )

            source_nhanes = categorical_total_variation(
                datasets["source"][feature],
                datasets["nhanes"][feature],
            )

            kaggle_nhanes = categorical_total_variation(
                datasets["kaggle"][feature],
                datasets["nhanes"][feature],
            )

            metric = "total_variation"

        else:
            source_kaggle = pooled_standardized_difference(
                datasets["source"][feature],
                datasets["kaggle"][feature],
            )

            source_nhanes = pooled_standardized_difference(
                datasets["source"][feature],
                datasets["nhanes"][feature],
            )

            kaggle_nhanes = pooled_standardized_difference(
                datasets["kaggle"][feature],
                datasets["nhanes"][feature],
            )

            metric = "standardized_mean_difference"

        rows.append(
            {
                "feature": feature,
                "feature_type": (
                    "categorical"
                    if is_categorical
                    else "numeric"
                ),
                "metric": metric,
                "source_vs_kaggle":
                    source_kaggle,
                "source_vs_nhanes":
                    source_nhanes,
                "kaggle_vs_nhanes":
                    kaggle_nhanes,
                "abs_source_vs_kaggle": (
                    abs(source_kaggle)
                    if pd.notna(source_kaggle)
                    else np.nan
                ),
                "abs_source_vs_nhanes": (
                    abs(source_nhanes)
                    if pd.notna(source_nhanes)
                    else np.nan
                ),
                "abs_kaggle_vs_nhanes": (
                    abs(kaggle_nhanes)
                    if pd.notna(kaggle_nhanes)
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        "\nLargest Source vs NHANES differences:"
    )

    print(
        result[
            [
                "feature",
                "feature_type",
                "metric",
                "source_vs_nhanes",
            ]
        ]
        .sort_values(
            "source_vs_nhanes",
            key=lambda series:
                series.abs(),
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    return result


# =============================================================================
# Save outputs
# =============================================================================


def save_outputs(
    numeric_summary: pd.DataFrame,
    categorical_frequencies: pd.DataFrame,
    categorical_values: pd.DataFrame,
    target_summary: pd.DataFrame,
    feature_compatibility: pd.DataFrame,
    complete_case_summary: pd.DataFrame,
) -> None:
    print_header("SAVING OUTPUTS")

    outputs = (
        (
            NUMERIC_SUMMARY_PATH,
            numeric_summary,
        ),
        (
            CATEGORICAL_FREQUENCIES_PATH,
            categorical_frequencies,
        ),
        (
            CATEGORICAL_VALUES_PATH,
            categorical_values,
        ),
        (
            TARGET_SUMMARY_PATH,
            target_summary,
        ),
        (
            FEATURE_COMPATIBILITY_PATH,
            feature_compatibility,
        ),
        (
            COMPLETE_CASE_SUMMARY_PATH,
            complete_case_summary,
        ),
    )

    for path, df in outputs:
        df.to_csv(
            path,
            index=False,
        )

        print(
            path.relative_to(ROOT)
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_output_directory()

    print_header(
        "THREE-DOMAIN HARMONIZATION AUDIT"
    )

    print(
        f"Feature set: COMMON_20 "
        f"({len(COMMON_20)} features)"
    )

    datasets = load_datasets()

    validate_schema(
        datasets
    )

    target_summary = (
        build_target_summary(
            datasets
        )
    )

    complete_case_summary = (
        build_complete_case_summary(
            datasets
        )
    )

    numeric_summary = (
        build_numeric_summary(
            datasets
        )
    )

    categorical_values = (
        build_categorical_values(
            datasets
        )
    )

    categorical_frequencies = (
        build_categorical_frequencies(
            datasets
        )
    )

    feature_compatibility = (
        build_feature_compatibility(
            datasets
        )
    )

    save_outputs(
        numeric_summary,
        categorical_frequencies,
        categorical_values,
        target_summary,
        feature_compatibility,
        complete_case_summary,
    )

    print_header(
        "HARMONIZATION AUDIT COMPLETE"
    )

    print(
        "No datasets were modified."
    )

    print(
        "\nReview feature_compatibility.csv and numeric_summary.csv "
        "before freezing the final three-domain feature space."
    )


if __name__ == "__main__":
    main()
