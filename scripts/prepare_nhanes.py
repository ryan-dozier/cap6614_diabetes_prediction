from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import (
    NHANES_PREPARATION_DIR as OUTPUT_DIR,
    NHANES_RAW_DIR as NHANES_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT as ROOT,
    SOURCE_RAW_PATH,
    SPLITS_DIR,
)

# =============================================================================
# Paths
# =============================================================================

HARMONIZED_PATH = (
    PROCESSED_DIR / "nhanes_harmonized.csv"
)

COMMON_16_PATH = (
    PROCESSED_DIR / "nhanes_common_16.csv"
)

COMMON_20_PATH = (
    PROCESSED_DIR / "nhanes_common_20.csv"
)

COMMON_22_PATH = (
    PROCESSED_DIR / "nhanes_common_22.csv"
)

WEIGHTS_PATH = (
    PROCESSED_DIR / "nhanes_weights.csv"
)

MISSINGNESS_PATH = (
    OUTPUT_DIR / "feature_missingness.csv"
)

METADATA_PATH = (
    OUTPUT_DIR / "metadata.json"
)

SPLIT_ASSIGNMENTS_PATH = (
    SPLITS_DIR / "nhanes_split_assignments.csv"
)


# =============================================================================
# Configuration
# =============================================================================

MINIMUM_AGE = 20

TARGET = "diagnosed_diabetes"

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


# 24 original Source/Kaggle features minus:
#
#   diet_score
#   waist_to_hip_ratio
#
# which do not currently have defensible NHANES equivalents.

COMMON_22 = (
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
    "ldl_cholesterol",
    "physical_activity_minutes_per_week",
    "screen_time_hours_per_day",
    "sleep_hours_per_day",
    "smoking_status",
    "systolic_bp",
    "triglycerides",
)


# Higher-coverage feature set that additionally excludes fasting-only
# LDL and triglycerides.

COMMON_20 = tuple(
    feature
    for feature in COMMON_22
    if feature not in {
        "ldl_cholesterol",
        "triglycerides",
    }
)


# Primary three-domain feature set.
#
# These four variables are excluded because their NHANES representations are
# more approximate than the remaining shared features:
#
#   education_level
#   employment_status
#   income_level
#   screen_time_hours_per_day
#
# The broader COMMON_20 set remains useful as a sensitivity analysis.

STRICT_COMMON_16 = tuple(
    feature
    for feature in COMMON_20
    if feature not in {
        "education_level",
        "employment_status",
        "income_level",
        "screen_time_hours_per_day",
    }
)


CATEGORICAL_FEATURES = (
    "education_level",
    "employment_status",
    "ethnicity",
    "gender",
    "income_level",
    "smoking_status",
)


# =============================================================================
# Raw NHANES variables
# =============================================================================

COMPONENT_COLUMNS = {
    "DEMO_I": (
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDHHIN2",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ),

    "BMX_I": (
        "SEQN",
        "BMXBMI",
    ),

    "BPX_I": (
        "SEQN",
        "BPXPLS",
        "BPXSY1",
        "BPXSY2",
        "BPXSY3",
        "BPXSY4",
        "BPXDI1",
        "BPXDI2",
        "BPXDI3",
        "BPXDI4",
    ),

    "TCHOL_I": (
        "SEQN",
        "LBXTC",
    ),

    "HDL_I": (
        "SEQN",
        "LBDHDD",
    ),

    "TRIGLY_I": (
        "SEQN",
        "WTSAF2YR",
        "LBXTR",
        "LBDLDL",
    ),

    "DIQ_I": (
        "SEQN",
        "DIQ010",
    ),

    "PAQ_I": (
        "SEQN",

        "PAQ605",
        "PAQ610",
        "PAD615",

        "PAQ620",
        "PAQ625",
        "PAD630",

        "PAQ635",
        "PAQ640",
        "PAD645",

        "PAQ650",
        "PAQ655",
        "PAD660",

        "PAQ665",
        "PAQ670",
        "PAD675",

        "PAQ710",
        "PAQ715",
    ),

    "SLQ_I": (
        "SEQN",
        "SLD012",
    ),

    "SMQ_I": (
        "SEQN",
        "SMQ020",
        "SMQ040",
    ),

    "ALQ_I": (
        "SEQN",
        "ALQ101",
        "ALQ110",
        "ALQ120Q",
        "ALQ120U",
        "ALQ130",
    ),

    "OCQ_I": (
        "SEQN",
        "OCD150",
        "OCQ380",
    ),

    "BPQ_I": (
        "SEQN",
        "BPQ020",
    ),

    "MCQ_I": (
        "SEQN",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
        "MCQ300C",
    ),
}


# =============================================================================
# Utilities
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_output_directories() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def component_path(component: str) -> Path:
    return NHANES_DIR / f"{component}.XPT"


def normalize_sas_zero_artifacts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pandas/XPORT decoding can represent numeric SAS zero as an extremely
    small non-zero floating-point value (for example ~5.4e-79).

    Convert such values back to exact zero.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        values = df[column]

        mask = (
            values.notna()
            & values.ne(0)
            & values.abs().lt(1e-50)
        )

        if mask.any():
            df.loc[mask, column] = 0.0

    return df


def load_component(
    component: str,
) -> pd.DataFrame:
    path = component_path(component)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing NHANES component:\n"
            f"  {path.relative_to(ROOT)}"
        )

    df = pd.read_sas(
        path,
        format="xport",
    )

    df.columns = [
        str(column).upper()
        for column in df.columns
    ]

    df = normalize_sas_zero_artifacts(df)

    required = set(
        COMPONENT_COLUMNS[component]
    )

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{component} is missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = df[
        list(COMPONENT_COLUMNS[component])
    ].copy()

    if result["SEQN"].isna().any():
        raise ValueError(
            f"{component} contains missing SEQN."
        )

    if result["SEQN"].duplicated().any():
        raise ValueError(
            f"{component} contains duplicate SEQN."
        )

    return result


# =============================================================================
# Loading / merging
# =============================================================================


def load_and_merge_components() -> pd.DataFrame:
    print_header("LOADING NHANES COMPONENTS")

    components = {}

    for component in COMPONENT_COLUMNS:
        df = load_component(component)

        components[component] = df

        print(
            f"{component:<10} "
            f"rows={len(df):>6,}"
        )

    # -------------------------------------------------------------------------
    # Start with the adult demographic population.
    # -------------------------------------------------------------------------

    merged = components["DEMO_I"].loc[
        components["DEMO_I"]["RIDAGEYR"] >= MINIMUM_AGE
    ].copy()

    print(
        f"\nAdults age >= {MINIMUM_AGE}: "
        f"{len(merged):,}"
    )

    # -------------------------------------------------------------------------
    # Left joins preserve the entire adult base population.
    #
    # Missing component participation remains missing rather than silently
    # deleting participants during merging.
    # -------------------------------------------------------------------------

    for component, df in components.items():
        if component == "DEMO_I":
            continue

        merged = merged.merge(
            df,
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

    print(
        f"Merged adult rows: "
        f"{len(merged):,}"
    )

    return merged


# =============================================================================
# Generic recoding helpers
# =============================================================================


def map_codes(
    series: pd.Series,
    mapping: dict,
) -> pd.Series:
    return series.map(mapping)


def binary_yes_no(
    series: pd.Series,
) -> pd.Series:
    """
    Standard NHANES yes/no coding:

        1 -> yes
        2 -> no

    Refused / don't know / missing remain NA.
    """

    return (
        series
        .map(
            {
                1.0: 1,
                2.0: 0,
            }
        )
        .astype("Int64")
    )


# =============================================================================
# Demographic features
# =============================================================================


def derive_gender(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["RIAGENDR"]
        .map(
            {
                1.0: "Male",
                2.0: "Female",
            }
        )
        .astype("string")
    )


def derive_ethnicity(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Harmonize NHANES RIDRETH3 to the five Source/Kaggle categories.
    """

    return (
        df["RIDRETH3"]
        .map(
            {
                1.0: "Hispanic",  # Mexican American
                2.0: "Hispanic",  # Other Hispanic
                3.0: "White",
                4.0: "Black",
                6.0: "Asian",
                7.0: "Other",
            }
        )
        .astype("string")
    )


def derive_education(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Approximate harmonization to the Source/Kaggle four-category vocabulary.

    NHANES:
        1  Less than 9th grade
        2  9-11th grade
        3  High-school graduate / GED
        4  Some college / AA
        5  College graduate or above

    This mapping is intentionally recorded as approximate.
    """

    return (
        df["DMDEDUC2"]
        .map(
            {
                1.0: "No formal",
                2.0: "Highschool",
                3.0: "Highschool",
                4.0: "Graduate",
                5.0: "Postgraduate",
            }
        )
        .astype("string")
    )


def derive_income(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Approximate mapping of NHANES annual household-income ranges to the
    five Source/Kaggle income categories.

    Code 12 ("$20,000 and over") is too broad to place reliably and
    therefore remains missing.
    """

    mapping = {
        # Under $25k
        1.0: "Low",
        2.0: "Low",
        3.0: "Low",
        4.0: "Low",
        5.0: "Low",
        13.0: "Low",

        # $25k-$44,999
        6.0: "Lower-Middle",
        7.0: "Lower-Middle",

        # $45k-$64,999
        8.0: "Middle",
        9.0: "Middle",

        # $65k-$99,999
        10.0: "Upper-Middle",
        14.0: "Upper-Middle",

        # $100k+
        15.0: "High",
    }

    return (
        df["INDHHIN2"]
        .map(mapping)
        .astype("string")
    )


def derive_employment(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Harmonize employment to:

        Employed
        Retired
        Student
        Unemployed

    NHANES respondents who are not working and are neither students nor
    retired are grouped into the project's broader Unemployed category.
    """

    status = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    ocd150 = df["OCD150"]
    reason = df["OCQ380"]

    # Working or temporarily absent from a job.
    status.loc[
        ocd150.isin([1.0, 2.0])
    ] = "Employed"

    # Explicitly looking for work.
    status.loc[
        ocd150.eq(3.0)
    ] = "Unemployed"

    not_working = ocd150.eq(4.0)

    status.loc[
        not_working
        & reason.eq(2.0)
    ] = "Student"

    status.loc[
        not_working
        & reason.eq(3.0)
    ] = "Retired"

    # Homemaker, unable to work, layoff, disabled, or other non-work reason.
    status.loc[
        not_working
        & reason.isin(
            [
                1.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ]
        )
    ] = "Unemployed"

    return status


# =============================================================================
# Smoking / alcohol
# =============================================================================


def derive_smoking_status(
    df: pd.DataFrame,
) -> pd.Series:
    status = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    ever = df["SMQ020"]
    current = df["SMQ040"]

    status.loc[
        ever.eq(2.0)
    ] = "Never"

    status.loc[
        ever.eq(1.0)
        & current.isin([1.0, 2.0])
    ] = "Current"

    status.loc[
        ever.eq(1.0)
        & current.eq(3.0)
    ] = "Former"

    return status


def derive_alcohol_per_week(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Approximate average alcoholic drinks per week.

    ALQ120Q provides drinking frequency.
    ALQ120U specifies week/month/year.
    ALQ130 provides average drinks per drinking day.
    """

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=np.float64,
    )

    quantity = pd.to_numeric(
        df["ALQ120Q"],
        errors="coerce",
    )

    unit = pd.to_numeric(
        df["ALQ120U"],
        errors="coerce",
    )

    drinks = pd.to_numeric(
        df["ALQ130"],
        errors="coerce",
    )

    # Explicitly reported never in the past year.
    result.loc[
        quantity.eq(0)
    ] = 0.0

    # Fewer than 12 drinks over the respondent's lifetime is effectively
    # zero on the weekly scale used by the synthetic dataset.
    result.loc[
        df["ALQ110"].eq(2.0)
    ] = 0.0

    valid_quantity = quantity.between(
        1,
        365,
        inclusive="both",
    )

    valid_drinks = drinks.between(
        1,
        14,
        inclusive="both",
    )

    valid = (
        valid_quantity
        & unit.isin([1.0, 2.0, 3.0])
        & valid_drinks
    )

    days_per_week = pd.Series(
        np.nan,
        index=df.index,
        dtype=np.float64,
    )

    days_per_week.loc[
        valid & unit.eq(1.0)
    ] = quantity

    days_per_week.loc[
        valid & unit.eq(2.0)
    ] = (
        quantity * 12.0 / 52.1429
    )

    days_per_week.loc[
        valid & unit.eq(3.0)
    ] = (
        quantity / 52.1429
    )

    result.loc[valid] = (
        days_per_week.loc[valid]
        * drinks.loc[valid]
    )

    return result


# =============================================================================
# Physical activity
# =============================================================================


def derive_activity_component(
    gate: pd.Series,
    days: pd.Series,
    minutes: pd.Series,
) -> pd.Series:
    """
    Convert one NHANES GPAQ activity section to minutes/week.

    Gate:
        1 -> participates
        2 -> does not participate

    Follow-up missingness after a "No" response is structural and becomes 0.
    """

    result = pd.Series(
        np.nan,
        index=gate.index,
        dtype=np.float64,
    )

    result.loc[
        gate.eq(2.0)
    ] = 0.0

    valid_days = days.between(
        1,
        7,
        inclusive="both",
    )

    valid_minutes = minutes.between(
        10,
        1440,
        inclusive="both",
    )

    active_valid = (
        gate.eq(1.0)
        & valid_days
        & valid_minutes
    )

    result.loc[active_valid] = (
        days.loc[active_valid]
        * minutes.loc[active_valid]
    )

    return result


def derive_physical_activity(
    df: pd.DataFrame,
) -> pd.Series:
    components = [
        derive_activity_component(
            df["PAQ605"],
            df["PAQ610"],
            df["PAD615"],
        ),

        derive_activity_component(
            df["PAQ620"],
            df["PAQ625"],
            df["PAD630"],
        ),

        derive_activity_component(
            df["PAQ635"],
            df["PAQ640"],
            df["PAD645"],
        ),

        derive_activity_component(
            df["PAQ650"],
            df["PAQ655"],
            df["PAD660"],
        ),

        derive_activity_component(
            df["PAQ665"],
            df["PAQ670"],
            df["PAD675"],
        ),
    ]

    frame = pd.concat(
        components,
        axis=1,
    )

    # Require every section to be interpretable.
    return frame.sum(
        axis=1,
        min_count=len(components),
    )


# =============================================================================
# Screen time
# =============================================================================


def screen_category_to_hours(
    series: pd.Series,
) -> pd.Series:
    """
    Convert NHANES screen-time categories to approximate hours/day.

    0 = <1 hour      -> midpoint 0.5
    1-4              -> reported hours
    5 = 5+ hours     -> lower-bound value 5
    8 = none         -> 0
    """

    return series.map(
        {
            0.0: 0.5,
            1.0: 1.0,
            2.0: 2.0,
            3.0: 3.0,
            4.0: 4.0,
            5.0: 5.0,
            8.0: 0.0,
        }
    )


def derive_screen_time(
    df: pd.DataFrame,
) -> pd.Series:
    tv = screen_category_to_hours(
        df["PAQ710"]
    )

    computer = screen_category_to_hours(
        df["PAQ715"]
    )

    return pd.concat(
        [tv, computer],
        axis=1,
    ).sum(
        axis=1,
        min_count=2,
    )


# =============================================================================
# Medical-history features
# =============================================================================


def derive_cardiovascular_history(
    df: pd.DataFrame,
) -> pd.Series:
    columns = [
        "MCQ160B",  # heart failure
        "MCQ160C",  # coronary heart disease
        "MCQ160D",  # angina
        "MCQ160E",  # heart attack
        "MCQ160F",  # stroke
    ]

    values = df[columns]

    result = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    any_yes = (
        values.eq(1.0)
        .any(axis=1)
    )

    all_no = (
        values.eq(2.0)
        .all(axis=1)
    )

    result.loc[any_yes] = 1
    result.loc[all_no] = 0

    return result


# =============================================================================
# Blood pressure
# =============================================================================


def mean_available_measurements(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    values = df[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return values.mean(
        axis=1,
        skipna=True,
    )


# =============================================================================
# Target
# =============================================================================


def derive_diabetes_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    DIQ010:

        1 -> diagnosed diabetes
        2 -> no diagnosed diabetes
        3 -> borderline

    Borderline responses are intentionally left missing rather than being
    assigned to either binary class.
    """

    return (
        df["DIQ010"]
        .map(
            {
                1.0: 1,
                2.0: 0,
            }
        )
        .astype("Int64")
    )


# =============================================================================
# Harmonized dataset construction
# =============================================================================


def build_harmonized_dataset(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    print_header("BUILDING HARMONIZED NHANES DATASET")

    output = pd.DataFrame(
        {
            "seqn":
                raw["SEQN"].astype("Int64"),

            "age":
                raw["RIDAGEYR"],

            "gender":
                derive_gender(raw),

            "ethnicity":
                derive_ethnicity(raw),

            "education_level":
                derive_education(raw),

            "income_level":
                derive_income(raw),

            "employment_status":
                derive_employment(raw),

            "smoking_status":
                derive_smoking_status(raw),

            "alcohol_consumption_per_week":
                derive_alcohol_per_week(raw),

            "physical_activity_minutes_per_week":
                derive_physical_activity(raw),

            "sleep_hours_per_day":
                raw["SLD012"],

            "screen_time_hours_per_day":
                derive_screen_time(raw),

            "family_history_diabetes":
                binary_yes_no(
                    raw["MCQ300C"]
                ),

            "hypertension_history":
                binary_yes_no(
                    raw["BPQ020"]
                ),

            "cardiovascular_history":
                derive_cardiovascular_history(raw),

            "bmi":
                raw["BMXBMI"],

            "systolic_bp":
                mean_available_measurements(
                    raw,
                    [
                        "BPXSY1",
                        "BPXSY2",
                        "BPXSY3",
                        "BPXSY4",
                    ],
                ),

            "diastolic_bp":
                mean_available_measurements(
                    raw,
                    [
                        "BPXDI1",
                        "BPXDI2",
                        "BPXDI3",
                        "BPXDI4",
                    ],
                ),

            "heart_rate":
                raw["BPXPLS"],

            "cholesterol_total":
                raw["LBXTC"],

            "hdl_cholesterol":
                raw["LBDHDD"],

            "ldl_cholesterol":
                raw["LBDLDL"],

            "triglycerides":
                raw["LBXTR"],

            TARGET:
                derive_diabetes_target(raw),
        }
    )

    # Stable column order.
    output = output[
        [
            "seqn",
            *COMMON_22,
            TARGET,
        ]
    ]

    print(
        f"Rows: {len(output):,}"
    )

    print(
        f"Candidate features: "
        f"{len(COMMON_22)}"
    )

    return output


# =============================================================================
# Validation
# =============================================================================


def validate_categorical_vocabularies(
    nhanes: pd.DataFrame,
) -> None:
    """
    Ensure harmonized categories are valid subsets of the raw Source-domain
    vocabularies.

    Reading the raw source CSV keeps NHANES preparation independent of
    prepare_data.py and data.py, which allows setup to prepare NHANES first.
    """

    if not SOURCE_RAW_PATH.exists():
        raise FileNotFoundError(
            "Missing source dataset required for categorical-vocabulary "
            "validation:\n"
            f"  {SOURCE_RAW_PATH.relative_to(ROOT)}"
        )

    source = pd.read_csv(
        SOURCE_RAW_PATH,
        usecols=list(CATEGORICAL_FEATURES),
    )

    for feature in CATEGORICAL_FEATURES:
        expected = set(
            source[feature]
            .dropna()
            .astype(str)
            .unique()
        )

        observed = set(
            nhanes[feature]
            .dropna()
            .astype(str)
            .unique()
        )

        unexpected = observed - expected

        if unexpected:
            raise ValueError(
                f"{feature}: harmonization generated "
                f"unexpected categories: "
                f"{sorted(unexpected)}"
            )

    print(
        "Categorical vocabularies: OK"
    )


def validate_identifier(
    nhanes: pd.DataFrame,
) -> None:
    if nhanes["seqn"].isna().any():
        raise ValueError(
            "Prepared NHANES data contains missing SEQN."
        )

    if nhanes["seqn"].duplicated().any():
        raise ValueError(
            "Prepared NHANES data contains duplicate SEQN."
        )



# =============================================================================
# Experiment-ready complete-case datasets
# =============================================================================


def build_complete_case_dataset(
    nhanes: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.DataFrame:
    columns = [
        "seqn",
        *features,
        TARGET,
    ]

    result = (
        nhanes[columns]
        .dropna(
            subset=[
                *features,
                TARGET,
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    result[TARGET] = (
        result[TARGET]
        .astype("int64")
    )

    return result


def validate_complete_case_dataset(
    df: pd.DataFrame,
    features: tuple[str, ...],
    name: str,
) -> None:
    expected_columns = [
        "seqn",
        *features,
        TARGET,
    ]

    if df.columns.tolist() != expected_columns:
        raise ValueError(
            f"{name}: unexpected column order/schema."
        )

    if df.isna().any().any():
        raise ValueError(
            f"{name}: complete-case dataset still contains missing values."
        )

    if df["seqn"].duplicated().any():
        raise ValueError(
            f"{name}: duplicate SEQN values found."
        )

    observed_target = set(
        df[TARGET]
        .unique()
        .tolist()
    )

    if not observed_target.issubset({0, 1}):
        raise ValueError(
            f"{name}: target contains values outside {{0, 1}}: "
            f"{sorted(observed_target)}"
        )


# =============================================================================
# Fixed train / validation / test splits
# =============================================================================


def create_stratified_split_indices(
    df: pd.DataFrame,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Create deterministic 70/15/15 stratified row-index splits.

    The authoritative split is created from STRICT_COMMON_16. The broader
    COMMON_20 and COMMON_22 datasets are subsets of this cohort and therefore
    reuse the same participant (SEQN) assignments.
    """

    if not np.isclose(
        TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION,
        1.0,
    ):
        raise ValueError(
            "Train/validation/test fractions must sum to 1."
        )

    total_rows = len(df)

    train_size = int(
        round(total_rows * TRAIN_FRACTION)
    )

    val_size = int(
        round(total_rows * VAL_FRACTION)
    )

    test_size = (
        total_rows
        - train_size
        - val_size
    )

    indices = np.arange(
        total_rows
    )

    train_indices, remaining_indices = (
        train_test_split(
            indices,
            train_size=train_size,
            test_size=val_size + test_size,
            random_state=seed,
            stratify=df[TARGET],
        )
    )

    remaining_targets = (
        df.iloc[remaining_indices][TARGET]
    )

    val_indices, test_indices = (
        train_test_split(
            remaining_indices,
            train_size=val_size,
            test_size=test_size,
            random_state=seed + 1,
            stratify=remaining_targets,
        )
    )

    return {
        "train": np.asarray(
            train_indices,
            dtype=np.int64,
        ),
        "val": np.asarray(
            val_indices,
            dtype=np.int64,
        ),
        "test": np.asarray(
            test_indices,
            dtype=np.int64,
        ),
    }


def validate_split_indices(
    split_indices: dict[str, np.ndarray],
    total_rows: int,
) -> None:
    split_sets = {
        name: set(
            indices.tolist()
        )
        for name, indices
        in split_indices.items()
    }

    if split_sets["train"] & split_sets["val"]:
        raise RuntimeError(
            "NHANES train and validation splits overlap."
        )

    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError(
            "NHANES train and test splits overlap."
        )

    if split_sets["val"] & split_sets["test"]:
        raise RuntimeError(
            "NHANES validation and test splits overlap."
        )

    observed = (
        split_sets["train"]
        | split_sets["val"]
        | split_sets["test"]
    )

    expected = set(
        range(total_rows)
    )

    if observed != expected:
        raise RuntimeError(
            "NHANES split assignments do not cover the "
            "STRICT_COMMON_16 dataset exactly."
        )


def class_distribution(
    target: pd.Series,
) -> dict:
    counts = (
        target
        .value_counts()
        .sort_index()
    )

    proportions = (
        target
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    return {
        "rows": int(
            len(target)
        ),
        "class_counts": {
            str(int(label)): int(count)
            for label, count
            in counts.items()
        },
        "class_proportions": {
            str(int(label)): float(proportion)
            for label, proportion
            in proportions.items()
        },
    }


def summarize_splits(
    df: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
) -> dict:
    summary = {}

    for split_name, indices in split_indices.items():
        summary[split_name] = (
            class_distribution(
                df.iloc[indices][TARGET]
            )
        )

    return summary


def print_split_summary(
    summary: dict,
) -> None:
    print_header(
        "CREATING FIXED NHANES SPLITS"
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        info = summary[split_name]

        positive_fraction = (
            info["class_proportions"]
            .get("1", 0.0)
        )

        print(
            f"  {split_name:<5} "
            f"rows={info['rows']:>6,}  "
            f"positive={positive_fraction:.6f}"
        )


def save_split_outputs(
    strict_common_16: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
) -> None:
    """
    Save the authoritative SEQN-to-split assignment and convenience split CSVs.

    The split assignment is based on STRICT_COMMON_16. Sensitivity feature sets
    reuse these SEQN assignments in data.py, ensuring the same participant
    never changes split when the feature set changes.
    """

    split_labels = np.empty(
        len(strict_common_16),
        dtype=object,
    )

    for split_name, indices in split_indices.items():
        split_labels[
            indices
        ] = split_name

    assignments = pd.DataFrame(
        {
            "row_index":
                np.arange(
                    len(strict_common_16)
                ),

            "seqn":
                strict_common_16[
                    "seqn"
                ].to_numpy(),

            "split":
                split_labels,
        }
    )

    assignments.to_csv(
        SPLIT_ASSIGNMENTS_PATH,
        index=False,
    )

    for split_name, indices in split_indices.items():
        path = (
            SPLITS_DIR
            / f"nhanes_{split_name}.csv"
        )

        split_df = (
            strict_common_16
            .iloc[indices]
            .reset_index(drop=True)
        )

        split_df.to_csv(
            path,
            index=False,
        )


# =============================================================================
# Diagnostics
# =============================================================================


def build_missingness_table(
    nhanes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in [
        *COMMON_22,
        TARGET,
    ]:
        nonmissing = int(
            nhanes[column]
            .notna()
            .sum()
        )

        missing = int(
            nhanes[column]
            .isna()
            .sum()
        )

        rows.append(
            {
                "feature": column,
                "rows": len(nhanes),
                "nonmissing": nonmissing,
                "missing": missing,
                "missing_fraction":
                    missing / len(nhanes),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
    )


def complete_case_summary(
    nhanes: pd.DataFrame,
    features: tuple[str, ...],
) -> dict:
    required = [
        *features,
        TARGET,
    ]

    complete = nhanes.dropna(
        subset=required
    )

    positive_rate = (
        float(
            complete[TARGET]
            .astype(int)
            .mean()
        )
        if len(complete)
        else None
    )

    return {
        "feature_count": len(features),
        "rows": len(complete),
        "positive_rows": (
            int(
                complete[TARGET]
                .astype(int)
                .sum()
            )
            if len(complete)
            else 0
        ),
        "positive_fraction":
            positive_rate,
    }


def print_preparation_summary(
    nhanes: pd.DataFrame,
    missingness: pd.DataFrame,
    strict_common_16_summary: dict,
    common_20_summary: dict,
    common_22_summary: dict,
) -> None:
    print_header("NHANES PREPARATION SUMMARY")

    print(
        f"Adult rows: "
        f"{len(nhanes):,}"
    )

    target_valid = int(
        nhanes[TARGET]
        .notna()
        .sum()
    )

    print(
        f"Rows with binary diabetes target: "
        f"{target_valid:,}"
    )

    print(
        "\nExperiment-ready complete-case datasets:"
    )

    print(
        f"  STRICT_COMMON_16: "
        f"{strict_common_16_summary['rows']:,} rows, "
        f"positive="
        f"{strict_common_16_summary['positive_fraction']:.4f} "
        f"[PRIMARY]"
        if strict_common_16_summary["positive_fraction"] is not None
        else "  STRICT_COMMON_16: 0 rows"
    )

    print(
        f"  COMMON_20:       "
        f"{common_20_summary['rows']:,} rows, "
        f"positive="
        f"{common_20_summary['positive_fraction']:.4f} "
        f"[EXTENDED HARMONIZATION]"
        if common_20_summary["positive_fraction"] is not None
        else "  COMMON_20: 0 rows"
    )

    print(
        f"  COMMON_22:       "
        f"{common_22_summary['rows']:,} rows, "
        f"positive="
        f"{common_22_summary['positive_fraction']:.4f} "
        f"[FASTING-LIPID SENSITIVITY]"
        if common_22_summary["positive_fraction"] is not None
        else "  COMMON_22: 0 rows"
    )

    print(
        "\nHighest feature missingness:"
    )

    print(
        missingness.head(10).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


# =============================================================================
# Save outputs
# =============================================================================


def save_outputs(
    raw: pd.DataFrame,
    nhanes: pd.DataFrame,
    strict_common_16: pd.DataFrame,
    common_20: pd.DataFrame,
    common_22: pd.DataFrame,
    missingness: pd.DataFrame,
    strict_common_16_summary: dict,
    common_20_summary: dict,
    common_22_summary: dict,
    split_summary: dict,
) -> None:
    nhanes.to_csv(
        HARMONIZED_PATH,
        index=False,
    )

    strict_common_16.to_csv(
        COMMON_16_PATH,
        index=False,
    )

    common_20.to_csv(
        COMMON_20_PATH,
        index=False,
    )

    common_22.to_csv(
        COMMON_22_PATH,
        index=False,
    )

    weights = pd.DataFrame(
        {
            "seqn":
                raw["SEQN"].astype("Int64"),

            "wtmec2yr":
                raw["WTMEC2YR"],

            "wtsaf2yr":
                raw["WTSAF2YR"],

            "sdmvpsu":
                raw["SDMVPSU"],

            "sdmvstra":
                raw["SDMVSTRA"],
        }
    )

    weights.to_csv(
        WEIGHTS_PATH,
        index=False,
    )

    missingness.to_csv(
        MISSINGNESS_PATH,
        index=False,
    )

    metadata = {
        "dataset":
            "NHANES 2015-2016",

        "minimum_age":
            MINIMUM_AGE,

        "target":
            TARGET,

        "adult_rows":
            len(nhanes),

        "primary_feature_set":
            "strict_common_16",

        "random_seed":
            SEED,

        "split_fractions": {
            "train": TRAIN_FRACTION,
            "val": VAL_FRACTION,
            "test": TEST_FRACTION,
        },

        "primary_splits":
            split_summary,

        "feature_sets": {
            "strict_common_16": {
                "role":
                    "Primary three-domain comparison.",
                "feature_names":
                    list(STRICT_COMMON_16),
                **strict_common_16_summary,
            },

            "common_20": {
                "role":
                    "Extended harmonization sensitivity analysis.",
                "feature_names":
                    list(COMMON_20),
                **common_20_summary,
            },

            "common_22": {
                "role":
                    "Fasting-lipid sensitivity analysis.",
                "feature_names":
                    list(COMMON_22),
                **common_22_summary,
            },
        },

        "strict_common_16_exclusions": {
            "education_level":
                "NHANES education categories require approximate mapping.",
            "employment_status":
                "NHANES work-status categories require approximate grouping.",
            "income_level":
                "NHANES dollar brackets require approximate mapping to project categories.",
            "screen_time_hours_per_day":
                "NHANES uses categorical/top-coded TV and computer-use responses.",
        },

        "excluded_source_kaggle_features": {
            "diet_score":
                "No directly comparable NHANES variable "
                "in the selected components.",

            "waist_to_hip_ratio":
                "Waist circumference is available but a "
                "matching hip circumference was not identified.",
        },

        "harmonization_notes": {
            "education_level":
                "Approximate category mapping.",

            "income_level":
                "Approximate income-bracket mapping.",

            "employment_status":
                "Non-working respondents who are neither "
                "students nor retired are grouped as Unemployed.",

            "alcohol_consumption_per_week":
                "Derived approximate average drinks/week.",

            "physical_activity_minutes_per_week":
                "Derived from NHANES GPAQ work, transport, "
                "and recreational activity.",

            "screen_time_hours_per_day":
                "Derived from categorical TV/video and "
                "computer-use responses.",

            "systolic_bp":
                "Mean of available NHANES systolic readings.",

            "diastolic_bp":
                "Mean of available NHANES diastolic readings.",

            TARGET:
                "DIQ010 Yes -> 1, No -> 0; "
                "Borderline/refused/don't-know excluded.",
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
# Main
# =============================================================================


def main() -> None:
    ensure_output_directories()

    print_header(
        "PREPARING NHANES 2015-2016"
    )

    raw = load_and_merge_components()

    nhanes = build_harmonized_dataset(
        raw
    )

    validate_identifier(nhanes)
    validate_categorical_vocabularies(
        nhanes
    )

    missingness = (
        build_missingness_table(
            nhanes
        )
    )

    strict_common_16 = (
        build_complete_case_dataset(
            nhanes,
            STRICT_COMMON_16,
        )
    )

    common_20 = (
        build_complete_case_dataset(
            nhanes,
            COMMON_20,
        )
    )

    common_22 = (
        build_complete_case_dataset(
            nhanes,
            COMMON_22,
        )
    )

    validate_complete_case_dataset(
        strict_common_16,
        STRICT_COMMON_16,
        "STRICT_COMMON_16",
    )

    validate_complete_case_dataset(
        common_20,
        COMMON_20,
        "COMMON_20",
    )

    validate_complete_case_dataset(
        common_22,
        COMMON_22,
        "COMMON_22",
    )

    strict_common_16_summary = (
        complete_case_summary(
            nhanes,
            STRICT_COMMON_16,
        )
    )

    common_20_summary = (
        complete_case_summary(
            nhanes,
            COMMON_20,
        )
    )

    common_22_summary = (
        complete_case_summary(
            nhanes,
            COMMON_22,
        )
    )

    split_indices = (
        create_stratified_split_indices(
            strict_common_16,
            SEED,
        )
    )

    validate_split_indices(
        split_indices,
        len(strict_common_16),
    )

    primary_split_summary = (
        summarize_splits(
            strict_common_16,
            split_indices,
        )
    )

    print_split_summary(
        primary_split_summary
    )

    print_preparation_summary(
        nhanes,
        missingness,
        strict_common_16_summary,
        common_20_summary,
        common_22_summary,
    )

    save_outputs(
        raw,
        nhanes,
        strict_common_16,
        common_20,
        common_22,
        missingness,
        strict_common_16_summary,
        common_20_summary,
        common_22_summary,
        primary_split_summary,
    )

    save_split_outputs(
        strict_common_16,
        split_indices,
    )

    print_header(
        "NHANES PREPARATION COMPLETE"
    )

    print(
        HARMONIZED_PATH.relative_to(ROOT)
    )

    print(
        COMMON_16_PATH.relative_to(ROOT)
    )

    print(
        COMMON_20_PATH.relative_to(ROOT)
    )

    print(
        COMMON_22_PATH.relative_to(ROOT)
    )

    print(
        WEIGHTS_PATH.relative_to(ROOT)
    )

    print(
        SPLIT_ASSIGNMENTS_PATH.relative_to(ROOT)
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        print(
            (
                SPLITS_DIR
                / f"nhanes_{split_name}.csv"
            ).relative_to(ROOT)
        )

    print(
        MISSINGNESS_PATH.relative_to(ROOT)
    )

    print(
        METADATA_PATH.relative_to(ROOT)
    )

    print(
        "\nSTRICT_COMMON_16 is the primary three-domain feature set."
    )

    print(
        "COMMON_20 and COMMON_22 are retained as sensitivity analyses."
    )


if __name__ == "__main__":
    main()