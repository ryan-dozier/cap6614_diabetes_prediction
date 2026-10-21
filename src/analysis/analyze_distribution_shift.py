from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    ks_2samp,
    spearmanr,
    wasserstein_distance,
)

from src.data import (
    load_common,
    load_metadata,
    load_schema,
)
from src.paths import (
    DISTRIBUTION_SHIFT_DIR as OUTPUT_DIR,
    DOMAIN_CLASSIFIER_DIR,
    PROJECT_ROOT as ROOT,
)


# =============================================================================
# Paths
# =============================================================================

DOMAIN_IMPORTANCE_PATH = (
    DOMAIN_CLASSIFIER_DIR / "feature_importance.csv"
)

FIGURES_DIR = OUTPUT_DIR / "figures"

NUMERIC_SHIFT_PATH = (
    OUTPUT_DIR / "numeric_shift.csv"
)

CATEGORICAL_SHIFT_PATH = (
    OUTPUT_DIR / "categorical_shift.csv"
)

CATEGORICAL_FREQUENCIES_PATH = (
    OUTPUT_DIR / "categorical_frequencies.csv"
)

COMBINED_SHIFT_PATH = (
    OUTPUT_DIR / "combined_feature_shift.csv"
)

METRIC_CORRELATIONS_PATH = (
    OUTPUT_DIR / "metric_correlations.csv"
)

METADATA_PATH = (
    OUTPUT_DIR / "metadata.json"
)


# =============================================================================
# Configuration
# =============================================================================

SEED = 42

# Statistical calculations use every row.
#
# Plotting uses a deterministic sample because rendering hundreds of thousands
# of points or histogram observations provides little visual benefit.
PLOT_SAMPLE_SIZE = 100_000

TOP_DISTRIBUTION_PLOTS = 5

QUANTILES = [
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
]


# These columns are represented numerically but semantically behave as binary
# categorical indicators. They remain in the numeric analysis, but we also
# report their positive-class prevalence explicitly.
BINARY_NUMERIC_FEATURES = {
    "cardiovascular_history",
    "family_history_diabetes",
    "hypertension_history",
}


# =============================================================================
# Utility functions
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n"
            f"  {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def safe_filename(value: str) -> str:
    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return np.nan

    return numerator / denominator


def safe_standardized_difference(
    mean_source: float,
    mean_kaggle: float,
    std_source: float,
    std_kaggle: float,
) -> float:
    pooled_std = np.sqrt(
        (
            std_source ** 2
            + std_kaggle ** 2
        )
        / 2
    )

    if pooled_std == 0:
        return 0.0

    return (
        mean_kaggle - mean_source
    ) / pooled_std


def pooled_standard_deviation(
    std_source: float,
    std_kaggle: float,
) -> float:
    return float(
        np.sqrt(
            (
                std_source ** 2
                + std_kaggle ** 2
            )
            / 2
        )
    )


def variance_distortion(
    std_ratio: float,
) -> float:
    """
    Symmetric measure of variance change.

    Example:
        std ratio = 0.5
        std ratio = 2.0

    both produce the same distortion magnitude:
        abs(log(2))

    This makes variance contraction and expansion comparable.
    """

    if (
        not np.isfinite(std_ratio)
        or std_ratio <= 0
    ):
        return np.nan

    return float(
        abs(np.log(std_ratio))
    )


# =============================================================================
# Input loading
# =============================================================================


def load_domain_importance() -> pd.DataFrame:
    require_file(DOMAIN_IMPORTANCE_PATH)

    importance = pd.read_csv(
        DOMAIN_IMPORTANCE_PATH
    )

    required_columns = {
        "feature",
        "importance",
    }

    missing = (
        required_columns
        - set(importance.columns)
    )

    if missing:
        raise ValueError(
            "Domain feature-importance file is missing "
            "required columns: "
            + ", ".join(sorted(missing))
        )

    if importance["feature"].duplicated().any():
        raise ValueError(
            "Domain feature-importance file contains "
            "duplicate feature names."
        )

    schema = load_schema()

    expected_features = set(
        schema.features
    )

    observed_features = set(
        importance["feature"]
    )

    if observed_features != expected_features:
        missing_features = sorted(
            expected_features - observed_features
        )

        unexpected_features = sorted(
            observed_features - expected_features
        )

        message = [
            "Domain feature-importance feature set "
            "does not match the prepared schema."
        ]

        if missing_features:
            message.append(
                "Missing: "
                + ", ".join(missing_features)
            )

        if unexpected_features:
            message.append(
                "Unexpected: "
                + ", ".join(unexpected_features)
            )

        raise ValueError(
            "\n".join(message)
        )

    importance = importance[
        [
            "feature",
            "importance",
        ]
    ].copy()

    importance = importance.rename(
        columns={
            "importance":
                "domain_importance",
        }
    )

    return importance


def load_analysis_inputs():
    print_header("LOADING ANALYSIS INPUTS")

    source = load_common("source")
    kaggle = load_common("kaggle")
    importance = load_domain_importance()
    schema = load_schema()

    print(
        f"Source rows: "
        f"{len(source):,}"
    )

    print(
        f"Kaggle rows: "
        f"{len(kaggle):,}"
    )

    print(
        f"Shared features: "
        f"{len(schema.features)}"
    )

    print(
        f"Domain importance rows: "
        f"{len(importance):,}"
    )

    return (
        source,
        kaggle,
        importance,
        schema,
    )


# =============================================================================
# Numeric shift analysis
# =============================================================================


def analyze_numeric_feature(
    feature: str,
    source: pd.Series,
    kaggle: pd.Series,
) -> dict:
    source_values = source.to_numpy(
        dtype=np.float64
    )

    kaggle_values = kaggle.to_numpy(
        dtype=np.float64
    )

    source_mean = float(
        np.mean(source_values)
    )

    kaggle_mean = float(
        np.mean(kaggle_values)
    )

    source_std = float(
        np.std(
            source_values,
            ddof=1,
        )
    )

    kaggle_std = float(
        np.std(
            kaggle_values,
            ddof=1,
        )
    )

    pooled_std = (
        pooled_standard_deviation(
            source_std,
            kaggle_std,
        )
    )

    mean_difference = (
        kaggle_mean
        - source_mean
    )

    standardized_difference = (
        safe_standardized_difference(
            source_mean,
            kaggle_mean,
            source_std,
            kaggle_std,
        )
    )

    std_ratio = safe_ratio(
        kaggle_std,
        source_std,
    )

    ks_result = ks_2samp(
        source_values,
        kaggle_values,
        alternative="two-sided",
        method="auto",
    )

    wasserstein = float(
        wasserstein_distance(
            source_values,
            kaggle_values,
        )
    )

    normalized_wasserstein = (
        wasserstein / pooled_std
        if pooled_std > 0
        else 0.0
    )

    source_quantiles = np.quantile(
        source_values,
        QUANTILES,
    )

    kaggle_quantiles = np.quantile(
        kaggle_values,
        QUANTILES,
    )

    row = {
        "feature": feature,
        "is_binary_numeric":
            feature
            in BINARY_NUMERIC_FEATURES,

        "source_mean":
            source_mean,
        "kaggle_mean":
            kaggle_mean,

        "mean_difference":
            mean_difference,

        "standardized_mean_difference":
            standardized_difference,

        "abs_standardized_mean_difference":
            abs(standardized_difference),

        "source_std":
            source_std,
        "kaggle_std":
            kaggle_std,

        "std_ratio_kaggle_to_source":
            std_ratio,

        "variance_distortion":
            variance_distortion(
                std_ratio
            ),

        "ks_statistic":
            float(
                ks_result.statistic
            ),

        "ks_pvalue":
            float(
                ks_result.pvalue
            ),

        "wasserstein_distance":
            wasserstein,

        "normalized_wasserstein_distance":
            normalized_wasserstein,

        "source_min":
            float(
                np.min(source_values)
            ),

        "kaggle_min":
            float(
                np.min(kaggle_values)
            ),

        "source_max":
            float(
                np.max(source_values)
            ),

        "kaggle_max":
            float(
                np.max(kaggle_values)
            ),
    }

    for quantile, source_value, kaggle_value in zip(
        QUANTILES,
        source_quantiles,
        kaggle_quantiles,
    ):
        suffix = int(
            round(quantile * 100)
        )

        row[
            f"source_q{suffix:02d}"
        ] = float(source_value)

        row[
            f"kaggle_q{suffix:02d}"
        ] = float(kaggle_value)

    # -------------------------------------------------------------------------
    # Binary indicators are numerically represented but are more naturally
    # interpreted through prevalence.
    # -------------------------------------------------------------------------

    if feature in BINARY_NUMERIC_FEATURES:
        source_positive_rate = (
            source_mean
        )

        kaggle_positive_rate = (
            kaggle_mean
        )

        prevalence_difference = (
            kaggle_positive_rate
            - source_positive_rate
        )

        row[
            "source_positive_rate"
        ] = source_positive_rate

        row[
            "kaggle_positive_rate"
        ] = kaggle_positive_rate

        row[
            "positive_rate_difference"
        ] = prevalence_difference

        row[
            "abs_positive_rate_difference"
        ] = abs(
            prevalence_difference
        )

    else:
        row[
            "source_positive_rate"
        ] = np.nan

        row[
            "kaggle_positive_rate"
        ] = np.nan

        row[
            "positive_rate_difference"
        ] = np.nan

        row[
            "abs_positive_rate_difference"
        ] = np.nan

    return row


def analyze_numeric_features(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    numeric_features: tuple[str, ...],
) -> pd.DataFrame:
    print_header("ANALYZING NUMERIC FEATURES")

    rows = []

    for feature in numeric_features:
        rows.append(
            analyze_numeric_feature(
                feature,
                source[feature],
                kaggle[feature],
            )
        )

    result = pd.DataFrame(rows)

    result["rank_ks"] = (
        result["ks_statistic"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    result["rank_abs_smd"] = (
        result[
            "abs_standardized_mean_difference"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    result[
        "rank_normalized_wasserstein"
    ] = (
        result[
            "normalized_wasserstein_distance"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    result[
        "rank_variance_distortion"
    ] = (
        result[
            "variance_distortion"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    result = result.sort_values(
        "ks_statistic",
        ascending=False,
    ).reset_index(drop=True)

    print(
        f"Analyzed "
        f"{len(result)} numeric features."
    )

    return result


# =============================================================================
# Categorical shift analysis
# =============================================================================


def analyze_categorical_feature(
    feature: str,
    source: pd.Series,
    kaggle: pd.Series,
) -> tuple[dict, list[dict]]:
    source_values = (
        source.astype(str)
    )

    kaggle_values = (
        kaggle.astype(str)
    )

    categories = sorted(
        set(source_values.unique())
        | set(kaggle_values.unique())
    )

    source_counts = (
        source_values
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    kaggle_counts = (
        kaggle_values
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    source_frequencies = (
        source_counts
        / len(source_values)
    )

    kaggle_frequencies = (
        kaggle_counts
        / len(kaggle_values)
    )

    p = source_frequencies.to_numpy(
        dtype=np.float64
    )

    q = kaggle_frequencies.to_numpy(
        dtype=np.float64
    )

    absolute_differences = np.abs(
        p - q
    )

    total_variation = float(
        0.5
        * np.sum(
            absolute_differences
        )
    )

    # scipy returns Jensen-Shannon DISTANCE.
    # Squaring it gives Jensen-Shannon divergence.
    js_distance = float(
        jensenshannon(
            p,
            q,
            base=2.0,
        )
    )

    js_divergence = (
        js_distance ** 2
    )

    max_difference_index = int(
        np.argmax(
            absolute_differences
        )
    )

    summary = {
        "feature": feature,
        "category_count":
            len(categories),

        "total_variation_distance":
            total_variation,

        "js_distance":
            js_distance,

        "js_divergence":
            js_divergence,

        "max_abs_category_difference":
            float(
                absolute_differences[
                    max_difference_index
                ]
            ),

        "most_shifted_category":
            categories[
                max_difference_index
            ],
    }

    details = []

    for category in categories:
        source_frequency = float(
            source_frequencies[
                category
            ]
        )

        kaggle_frequency = float(
            kaggle_frequencies[
                category
            ]
        )

        difference = (
            kaggle_frequency
            - source_frequency
        )

        details.append(
            {
                "feature":
                    feature,

                "category":
                    category,

                "source_count":
                    int(
                        source_counts[
                            category
                        ]
                    ),

                "kaggle_count":
                    int(
                        kaggle_counts[
                            category
                        ]
                    ),

                "source_frequency":
                    source_frequency,

                "kaggle_frequency":
                    kaggle_frequency,

                "frequency_difference":
                    difference,

                "abs_frequency_difference":
                    abs(
                        difference
                    ),
            }
        )

    return (
        summary,
        details,
    )


def analyze_categorical_features(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    categorical_features: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    print_header(
        "ANALYZING CATEGORICAL FEATURES"
    )

    summaries = []
    details = []

    for feature in categorical_features:
        (
            summary,
            category_details,
        ) = analyze_categorical_feature(
            feature,
            source[feature],
            kaggle[feature],
        )

        summaries.append(summary)
        details.extend(category_details)

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df["rank_tv"] = (
        summary_df[
            "total_variation_distance"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    summary_df["rank_js"] = (
        summary_df[
            "js_divergence"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    summary_df = (
        summary_df
        .sort_values(
            "total_variation_distance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    details_df = pd.DataFrame(
        details
    )

    print(
        f"Analyzed "
        f"{len(summary_df)} "
        f"categorical features."
    )

    return (
        summary_df,
        details_df,
    )


# =============================================================================
# Merge statistical metrics with learned domain importance
# =============================================================================


def merge_feature_results(
    importance: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    schema,
) -> pd.DataFrame:
    rows = []

    for feature in schema.features:
        if feature in schema.numeric_features:
            feature_type = (
                "binary_numeric"
                if feature
                in BINARY_NUMERIC_FEATURES
                else "numeric"
            )
        else:
            feature_type = (
                "categorical"
            )

        rows.append(
            {
                "feature": feature,
                "feature_type":
                    feature_type,
            }
        )

    combined = pd.DataFrame(rows)

    combined = combined.merge(
        importance,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    numeric_columns = [
        "feature",
        "standardized_mean_difference",
        "abs_standardized_mean_difference",
        "source_std",
        "kaggle_std",
        "std_ratio_kaggle_to_source",
        "variance_distortion",
        "ks_statistic",
        "ks_pvalue",
        "wasserstein_distance",
        "normalized_wasserstein_distance",
        "source_positive_rate",
        "kaggle_positive_rate",
        "positive_rate_difference",
        "abs_positive_rate_difference",
    ]

    combined = combined.merge(
        numeric[
            numeric_columns
        ],
        on="feature",
        how="left",
        validate="one_to_one",
    )

    categorical_columns = [
        "feature",
        "total_variation_distance",
        "js_distance",
        "js_divergence",
        "max_abs_category_difference",
        "most_shifted_category",
    ]

    combined = combined.merge(
        categorical[
            categorical_columns
        ],
        on="feature",
        how="left",
        validate="one_to_one",
    )

    combined[
        "rank_domain_importance"
    ] = (
        combined[
            "domain_importance"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    ranking_columns = {
        "rank_abs_smd":
            "abs_standardized_mean_difference",

        "rank_ks":
            "ks_statistic",

        "rank_normalized_wasserstein":
            "normalized_wasserstein_distance",

        "rank_variance_distortion":
            "variance_distortion",

        "rank_total_variation":
            "total_variation_distance",

        "rank_js_divergence":
            "js_divergence",
    }

    for rank_column, metric_column in (
        ranking_columns.items()
    ):
        combined[rank_column] = (
            combined[metric_column]
            .rank(
                method="min",
                ascending=False,
                na_option="keep",
            )
            .astype("Int64")
        )

    combined = (
        combined
        .sort_values(
            "domain_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return combined


# =============================================================================
# Metric correlations
# =============================================================================


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[
    int,
    float,
    float,
]:
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    n = len(frame)

    if n < 3:
        return (
            n,
            np.nan,
            np.nan,
        )

    if (
        frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return (
            n,
            np.nan,
            np.nan,
        )

    result = spearmanr(
        frame["x"],
        frame["y"],
    )

    return (
        n,
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def calculate_metric_correlations(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    print_header(
        "CALCULATING METRIC CORRELATIONS"
    )

    rows = []

    numeric = combined[
        combined["feature_type"].isin(
            [
                "numeric",
                "binary_numeric",
            ]
        )
    ]

    numeric_metrics = [
        (
            "abs_standardized_mean_difference",
            "|standardized mean difference|",
        ),
        (
            "ks_statistic",
            "KS statistic",
        ),
        (
            "normalized_wasserstein_distance",
            "normalized Wasserstein distance",
        ),
        (
            "variance_distortion",
            "variance distortion",
        ),
    ]

    for (
        metric_column,
        metric_label,
    ) in numeric_metrics:
        n, rho, pvalue = safe_spearman(
            numeric[
                "domain_importance"
            ],
            numeric[
                metric_column
            ],
        )

        rows.append(
            {
                "feature_scope":
                    "numeric",

                "metric_x":
                    "domain_importance",

                "metric_y":
                    metric_column,

                "metric_y_label":
                    metric_label,

                "n_features":
                    n,

                "spearman_rho":
                    rho,

                "pvalue":
                    pvalue,
            }
        )

    categorical = combined[
        combined["feature_type"]
        == "categorical"
    ]

    categorical_metrics = [
        (
            "total_variation_distance",
            "total variation distance",
        ),
        (
            "js_divergence",
            "Jensen-Shannon divergence",
        ),
    ]

    for (
        metric_column,
        metric_label,
    ) in categorical_metrics:
        n, rho, pvalue = safe_spearman(
            categorical[
                "domain_importance"
            ],
            categorical[
                metric_column
            ],
        )

        rows.append(
            {
                "feature_scope":
                    "categorical",

                "metric_x":
                    "domain_importance",

                "metric_y":
                    metric_column,

                "metric_y_label":
                    metric_label,

                "n_features":
                    n,

                "spearman_rho":
                    rho,

                "pvalue":
                    pvalue,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================


def save_horizontal_bar_plot(
    data: pd.DataFrame,
    feature_column: str,
    metric_column: str,
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    plot_data = (
        data
        .sort_values(
            metric_column,
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.barh(
        plot_data[
            feature_column
        ],
        plot_data[
            metric_column
        ],
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plot_data = (
        data[
            [
                "feature",
                x_column,
                y_column,
            ]
        ]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        plot_data[x_column],
        plot_data[y_column],
    )

    for _, row in plot_data.iterrows():
        ax.annotate(
            row["feature"],
            (
                row[x_column],
                row[y_column],
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def sample_for_plot(
    series: pd.Series,
    seed: int,
) -> pd.Series:
    if len(series) <= PLOT_SAMPLE_SIZE:
        return series

    return series.sample(
        n=PLOT_SAMPLE_SIZE,
        random_state=seed,
    )


def save_numeric_distribution_plot(
    feature: str,
    source: pd.Series,
    kaggle: pd.Series,
    path: Path,
    seed: int,
) -> None:
    source_sample = sample_for_plot(
        source,
        seed,
    )

    kaggle_sample = sample_for_plot(
        kaggle,
        seed + 1,
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    # Binary variables are clearer as prevalence plots.
    if feature in BINARY_NUMERIC_FEATURES:
        source_frequencies = (
            source_sample
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        kaggle_frequencies = (
            kaggle_sample
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        categories = [0, 1]

        source_values = [
            source_frequencies.get(
                value,
                0.0,
            )
            for value in categories
        ]

        kaggle_values = [
            kaggle_frequencies.get(
                value,
                0.0,
            )
            for value in categories
        ]

        width = 0.35

        x = np.arange(
            len(categories)
        )

        ax.bar(
            x - width / 2,
            source_values,
            width,
            label="Source",
        )

        ax.bar(
            x + width / 2,
            kaggle_values,
            width,
            label="Kaggle",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            categories
        )

        ax.set_ylabel(
            "Proportion"
        )

    else:
        ax.hist(
            [
                source_sample,
                kaggle_sample,
            ],
            bins=50,
            density=True,
            alpha=0.6,
            label=[
                "Source",
                "Kaggle",
            ],
        )

        ax.set_ylabel(
            "Density"
        )

    ax.set_xlabel(feature)

    ax.set_title(
        f"Source vs. Kaggle: {feature}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_figures(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    combined: pd.DataFrame,
) -> list[str]:
    print_header("CREATING FIGURES")

    output_files = []

    importance_path = (
        FIGURES_DIR
        / "domain_feature_importance.png"
    )

    save_horizontal_bar_plot(
        combined,
        "feature",
        "domain_importance",
        "CatBoost domain importance",
        "Domain Classifier Feature Importance",
        importance_path,
    )

    output_files.append(
        str(
            importance_path.relative_to(
                ROOT
            )
        )
    )

    numeric = combined[
        combined["feature_type"].isin(
            [
                "numeric",
                "binary_numeric",
            ]
        )
    ].copy()

    ks_path = (
        FIGURES_DIR
        / "numeric_ks_statistics.png"
    )

    save_horizontal_bar_plot(
        numeric,
        "feature",
        "ks_statistic",
        "KS statistic",
        "Numeric Distribution Shift",
        ks_path,
    )

    output_files.append(
        str(
            ks_path.relative_to(ROOT)
        )
    )

    importance_vs_ks_path = (
        FIGURES_DIR
        / "importance_vs_ks.png"
    )

    save_scatter_plot(
        numeric,
        "ks_statistic",
        "domain_importance",
        "KS statistic",
        "CatBoost domain importance",
        "Domain Importance vs. KS Shift",
        importance_vs_ks_path,
    )

    output_files.append(
        str(
            importance_vs_ks_path
            .relative_to(ROOT)
        )
    )

    importance_vs_smd_path = (
        FIGURES_DIR
        / "importance_vs_abs_smd.png"
    )

    save_scatter_plot(
        numeric,
        "abs_standardized_mean_difference",
        "domain_importance",
        "|Standardized mean difference|",
        "CatBoost domain importance",
        (
            "Domain Importance vs. "
            "Standardized Mean Shift"
        ),
        importance_vs_smd_path,
    )

    output_files.append(
        str(
            importance_vs_smd_path
            .relative_to(ROOT)
        )
    )

    # -------------------------------------------------------------------------
    # Distribution plots for the most important NUMERIC domain features.
    # -------------------------------------------------------------------------

    top_numeric = (
        numeric
        .sort_values(
            "domain_importance",
            ascending=False,
        )
        .head(
            TOP_DISTRIBUTION_PLOTS
        )
    )

    distribution_dir = (
        FIGURES_DIR
        / "top_numeric_distributions"
    )

    distribution_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, feature in enumerate(
        top_numeric["feature"]
    ):
        filename = (
            f"{index + 1:02d}_"
            f"{safe_filename(feature)}.png"
        )

        path = (
            distribution_dir
            / filename
        )

        save_numeric_distribution_plot(
            feature,
            source[feature],
            kaggle[feature],
            path,
            SEED + index * 10,
        )

        output_files.append(
            str(
                path.relative_to(ROOT)
            )
        )

    print(
        f"Created "
        f"{len(output_files)} figures."
    )

    return output_files


# =============================================================================
# Saving results
# =============================================================================


def save_analysis_results(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    categorical_frequencies: pd.DataFrame,
    combined: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    numeric.to_csv(
        NUMERIC_SHIFT_PATH,
        index=False,
    )

    categorical.to_csv(
        CATEGORICAL_SHIFT_PATH,
        index=False,
    )

    categorical_frequencies.to_csv(
        CATEGORICAL_FREQUENCIES_PATH,
        index=False,
    )

    combined.to_csv(
        COMBINED_SHIFT_PATH,
        index=False,
    )

    correlations.to_csv(
        METRIC_CORRELATIONS_PATH,
        index=False,
    )


def save_metadata(
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    schema,
    figure_paths: list[str],
) -> None:
    preparation_metadata = (
        load_metadata()
    )

    metadata = {
        "analysis":
            "distribution_shift",

        "analysis_version":
            1,

        "seed":
            SEED,

        "datasets": {
            "source_rows":
                int(len(source)),

            "kaggle_rows":
                int(len(kaggle)),
        },

        "features": {
            "total":
                len(schema.features),

            "numeric":
                list(
                    schema.numeric_features
                ),

            "categorical":
                list(
                    schema.categorical_features
                ),

            "binary_numeric":
                sorted(
                    BINARY_NUMERIC_FEATURES
                ),
        },

        "numeric_metrics": [
            "standardized_mean_difference",
            "std_ratio_kaggle_to_source",
            "variance_distortion",
            "ks_statistic",
            "wasserstein_distance",
            "normalized_wasserstein_distance",
            "quantiles",
        ],

        "categorical_metrics": [
            "total_variation_distance",
            "jensen_shannon_distance",
            "jensen_shannon_divergence",
            "category_frequency_difference",
        ],

        "domain_classifier_input": {
            "path":
                str(
                    DOMAIN_IMPORTANCE_PATH
                    .relative_to(ROOT)
                ),

            "sha256":
                sha256_file(
                    DOMAIN_IMPORTANCE_PATH
                ),
        },

        "preparation": {
            "schema_version":
                preparation_metadata[
                    "schema_version"
                ],

            "random_seed":
                preparation_metadata[
                    "random_seed"
                ],

            "raw_files":
                preparation_metadata[
                    "raw_files"
                ],
        },

        "plotting": {
            "plot_sample_size":
                PLOT_SAMPLE_SIZE,

            "top_distribution_plots":
                TOP_DISTRIBUTION_PLOTS,

            "figures":
                figure_paths,
        },

        "outputs": {
            "numeric_shift":
                str(
                    NUMERIC_SHIFT_PATH
                    .relative_to(ROOT)
                ),

            "categorical_shift":
                str(
                    CATEGORICAL_SHIFT_PATH
                    .relative_to(ROOT)
                ),

            "categorical_frequencies":
                str(
                    CATEGORICAL_FREQUENCIES_PATH
                    .relative_to(ROOT)
                ),

            "combined_feature_shift":
                str(
                    COMBINED_SHIFT_PATH
                    .relative_to(ROOT)
                ),

            "metric_correlations":
                str(
                    METRIC_CORRELATIONS_PATH
                    .relative_to(ROOT)
                ),
        },
    }

    with METADATA_PATH.open(
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )


# =============================================================================
# Console summaries
# =============================================================================


def print_numeric_summary(
    numeric: pd.DataFrame,
) -> None:
    print_header(
        "TOP NUMERIC SHIFTS BY KS STATISTIC"
    )

    columns = [
        "feature",
        "ks_statistic",
        "abs_standardized_mean_difference",
        "std_ratio_kaggle_to_source",
        "normalized_wasserstein_distance",
    ]

    print(
        numeric[
            columns
        ]
        .sort_values(
            "ks_statistic",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print_header(
        "TOP NUMERIC VARIANCE DISTORTIONS"
    )

    columns = [
        "feature",
        "source_std",
        "kaggle_std",
        "std_ratio_kaggle_to_source",
        "variance_distortion",
    ]

    print(
        numeric[
            columns
        ]
        .sort_values(
            "variance_distortion",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_categorical_summary(
    categorical: pd.DataFrame,
) -> None:
    print_header(
        "CATEGORICAL DISTRIBUTION SHIFT"
    )

    columns = [
        "feature",
        "total_variation_distance",
        "js_divergence",
        "most_shifted_category",
        "max_abs_category_difference",
    ]

    print(
        categorical[
            columns
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def print_combined_summary(
    combined: pd.DataFrame,
) -> None:
    print_header(
        "DOMAIN IMPORTANCE AND STATISTICAL SHIFT"
    )

    columns = [
        "feature",
        "feature_type",
        "domain_importance",
        "ks_statistic",
        "abs_standardized_mean_difference",
        "std_ratio_kaggle_to_source",
        "total_variation_distance",
    ]

    print(
        combined[
            columns
        ]
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_correlation_summary(
    correlations: pd.DataFrame,
) -> None:
    print_header(
        "METRIC RANK CORRELATIONS"
    )

    columns = [
        "feature_scope",
        "metric_y_label",
        "n_features",
        "spearman_rho",
        "pvalue",
    ]

    print(
        correlations[
            columns
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_output_directories()

    (
        source,
        kaggle,
        importance,
        schema,
    ) = load_analysis_inputs()

    # -------------------------------------------------------------------------
    # Numeric shift analysis.
    # -------------------------------------------------------------------------

    numeric = analyze_numeric_features(
        source,
        kaggle,
        schema.numeric_features,
    )

    # -------------------------------------------------------------------------
    # Categorical shift analysis.
    # -------------------------------------------------------------------------

    (
        categorical,
        categorical_frequencies,
    ) = analyze_categorical_features(
        source,
        kaggle,
        schema.categorical_features,
    )

    # -------------------------------------------------------------------------
    # Combine statistics with CatBoost's learned domain importance.
    # -------------------------------------------------------------------------

    print_header(
        "MERGING WITH DOMAIN CLASSIFIER RESULTS"
    )

    combined = merge_feature_results(
        importance,
        numeric,
        categorical,
        schema,
    )

    print(
        f"Combined feature rows: "
        f"{len(combined)}"
    )

    # -------------------------------------------------------------------------
    # Examine whether simple statistical shift measures agree with the learned
    # domain classifier.
    # -------------------------------------------------------------------------

    correlations = (
        calculate_metric_correlations(
            combined
        )
    )

    # -------------------------------------------------------------------------
    # Save tabular outputs.
    # -------------------------------------------------------------------------

    print_header(
        "SAVING ANALYSIS TABLES"
    )

    save_analysis_results(
        numeric,
        categorical,
        categorical_frequencies,
        combined,
        correlations,
    )

    print(
        NUMERIC_SHIFT_PATH.relative_to(
            ROOT
        )
    )

    print(
        CATEGORICAL_SHIFT_PATH.relative_to(
            ROOT
        )
    )

    print(
        CATEGORICAL_FREQUENCIES_PATH.relative_to(
            ROOT
        )
    )

    print(
        COMBINED_SHIFT_PATH.relative_to(
            ROOT
        )
    )

    print(
        METRIC_CORRELATIONS_PATH.relative_to(
            ROOT
        )
    )

    # -------------------------------------------------------------------------
    # Generate figures.
    # -------------------------------------------------------------------------

    figure_paths = create_figures(
        source,
        kaggle,
        combined,
    )

    # -------------------------------------------------------------------------
    # Save reproducibility metadata.
    # -------------------------------------------------------------------------

    save_metadata(
        source,
        kaggle,
        schema,
        figure_paths,
    )

    # -------------------------------------------------------------------------
    # Console results.
    # -------------------------------------------------------------------------

    print_numeric_summary(
        numeric
    )

    print_categorical_summary(
        categorical
    )

    print_combined_summary(
        combined
    )

    print_correlation_summary(
        correlations
    )

    print_header(
        "DISTRIBUTION SHIFT ANALYSIS COMPLETE"
    )

    print(
        "Statistics were calculated using the "
        "complete prepared source and Kaggle datasets."
    )

    print(
        f"Plots use at most "
        f"{PLOT_SAMPLE_SIZE:,} rows per domain."
    )

    print(
        "\nImportant:"
    )

    print(
        "  With 100,000 source rows and 700,000 Kaggle rows, "
        "very small differences may have extremely small p-values."
    )

    print(
        "  Interpret effect sizes such as KS statistic, "
        "total variation distance, normalized Wasserstein distance, "
        "and variance distortion rather than p-values alone."
    )


if __name__ == "__main__":
    main()