import pandas as pd

from src.paths import (
    KAGGLE_RAW_PATH as KAGGLE_PATH,
    SOURCE_RAW_PATH as SOURCE_PATH,
)

TARGET = "diagnosed_diabetes"


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    source = pd.read_csv(SOURCE_PATH)
    kaggle = pd.read_csv(KAGGLE_PATH)

    # Normalize target types so both datasets use the same representation.
    source[TARGET] = source[TARGET].astype("int64")
    kaggle[TARGET] = kaggle[TARGET].astype("int64")

    print_header("DATASET SHAPES")
    print(f"Source: {source.shape}")
    print(f"Kaggle: {kaggle.shape}")

    # -------------------------------------------------------------------------
    # Determine shared and dataset-specific features.
    # -------------------------------------------------------------------------

    source_features = set(source.columns) - {TARGET}
    kaggle_features = set(kaggle.columns) - {
        "id",
        TARGET,
    }

    shared = sorted(source_features & kaggle_features)
    source_only = sorted(source_features - kaggle_features)
    kaggle_only = sorted(kaggle_features - source_features)

    print_header("SHARED FEATURES")

    for feature in shared:
        print(feature)

    print(f"\nTotal shared features: {len(shared)}")

    print_header("SOURCE-ONLY FEATURES")

    for feature in source_only:
        print(feature)

    print_header("KAGGLE-ONLY FEATURES")

    if kaggle_only:
        for feature in kaggle_only:
            print(feature)
    else:
        print("None")

    # -------------------------------------------------------------------------
    # Target distributions.
    # -------------------------------------------------------------------------

    print_header("TARGET DISTRIBUTIONS")

    print("\nSource:")
    print(
        source[TARGET]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nKaggle:")
    print(
        kaggle[TARGET]
        .value_counts(normalize=True)
        .sort_index()
    )

    # -------------------------------------------------------------------------
    # Missing-value checks.
    # -------------------------------------------------------------------------

    print_header("MISSING VALUES")

    print("\nSource:")
    print(
        source[shared]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nKaggle:")
    print(
        kaggle[shared]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    # -------------------------------------------------------------------------
    # Data types.
    # -------------------------------------------------------------------------

    print_header("DATA TYPES")

    print("\nSource:")
    print(source[shared].dtypes)

    print("\nKaggle:")
    print(kaggle[shared].dtypes)

    # -------------------------------------------------------------------------
    # Identify categorical and numeric features.
    # -------------------------------------------------------------------------

    categorical_features = [
        feature
        for feature in shared
        if (
            not pd.api.types.is_numeric_dtype(source[feature])
            or not pd.api.types.is_numeric_dtype(kaggle[feature])
        )
    ]

    numeric_features = [
        feature
        for feature in shared
        if feature not in categorical_features
    ]

    print_header("FEATURE TYPES")

    print("\nCategorical features:")
    for feature in categorical_features:
        print(f"  {feature}")

    print(f"\nTotal categorical features: {len(categorical_features)}")

    print("\nNumeric features:")
    for feature in numeric_features:
        print(f"  {feature}")

    print(f"\nTotal numeric features: {len(numeric_features)}")

    # -------------------------------------------------------------------------
    # Compare categorical values between source and Kaggle.
    # -------------------------------------------------------------------------

    print_header("CATEGORICAL VALUE COMPARISON")

    categorical_mismatches = []

    for feature in categorical_features:
        source_values = sorted(
            source[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        kaggle_values = sorted(
            kaggle[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        print(f"\n{feature}")
        print(f"  Source: {source_values}")
        print(f"  Kaggle: {kaggle_values}")

        if source_values == kaggle_values:
            print("  MATCH")
        else:
            print("  *** DIFFER ***")
            categorical_mismatches.append(feature)

    print("\nCategorical comparison summary:")

    if categorical_mismatches:
        print(
            "The following categorical features differ between datasets:"
        )

        for feature in categorical_mismatches:
            print(f"  {feature}")
    else:
        print("All categorical features use matching values.")

    # -------------------------------------------------------------------------
    # Compare numeric distributions.
    #
    # Standardized mean difference:
    #
    #     (Kaggle mean - Source mean) / pooled standard deviation
    #
    # This gives a rough measure of distribution shift on a common scale.
    # -------------------------------------------------------------------------

    rows = []

    for feature in numeric_features:
        source_mean = source[feature].mean()
        kaggle_mean = kaggle[feature].mean()

        source_std = source[feature].std()
        kaggle_std = kaggle[feature].std()

        pooled_std = (
            (source_std ** 2 + kaggle_std ** 2) / 2
        ) ** 0.5

        if pooled_std > 0:
            standardized_difference = (
                (kaggle_mean - source_mean) / pooled_std
            )
        else:
            standardized_difference = 0.0

        rows.append(
            {
                "feature": feature,
                "source_mean": source_mean,
                "kaggle_mean": kaggle_mean,
                "source_std": source_std,
                "kaggle_std": kaggle_std,
                "standardized_difference": standardized_difference,
                "abs_standardized_difference":
                    abs(standardized_difference),
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "abs_standardized_difference",
        ascending=False,
    )

    print_header("NUMERIC DISTRIBUTION DIFFERENCES")

    print(
        comparison[
            [
                "feature",
                "source_mean",
                "kaggle_mean",
                "source_std",
                "kaggle_std",
                "standardized_difference",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    print_header("SUMMARY")

    print(f"Source rows:             {len(source):,}")
    print(f"Kaggle rows:             {len(kaggle):,}")
    print(f"Shared features:         {len(shared)}")
    print(f"Source-only features:    {len(source_only)}")
    print(f"Kaggle-only features:    {len(kaggle_only)}")
    print(f"Categorical features:    {len(categorical_features)}")
    print(f"Numeric features:        {len(numeric_features)}")
    print(
        f"Categorical mismatches:  {len(categorical_mismatches)}"
    )

    print("\nLargest numeric distribution shifts:")

    print(
        comparison[
            [
                "feature",
                "standardized_difference",
            ]
        ]
        .head(5)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()