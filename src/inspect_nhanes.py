from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

NHANES_DIR = ROOT / "data/raw/nhanes_2015_2016"

OUTPUT_DIR = ROOT / "results/nhanes_inspection"

VALUE_CODE_AUDIT_PATH = (
    OUTPUT_DIR / "value_code_audit.csv"
)

DERIVATION_SUMMARY_PATH = (
    OUTPUT_DIR / "derivation_numeric_summary.csv"
)

COMPONENT_SUMMARY_PATH = OUTPUT_DIR / "component_summary.csv"
COMPONENT_COLUMNS_PATH = OUTPUT_DIR / "component_columns.csv"
VARIABLE_SUMMARY_PATH = OUTPUT_DIR / "candidate_variable_summary.csv"
MERGE_SURVIVAL_PATH = OUTPUT_DIR / "merge_survival.csv"
FEATURE_MAPPING_PATH = OUTPUT_DIR / "feature_mapping_candidates.csv"




# =============================================================================
# NHANES components
# =============================================================================

EXPECTED_COMPONENTS = (
    "DEMO_I",
    "BMX_I",
    "BPX_I",
    "TCHOL_I",
    "HDL_I",
    "TRIGLY_I",
    "GHB_I",
    "GLU_I",
    "INS_I",
    "OGTT_I",
    "DIQ_I",
    "PAQ_I",
    "SLQ_I",
    "SMQ_I",
    "ALQ_I",
    "OCQ_I",
    "BPQ_I",
    "MCQ_I",
)


# Candidate variables are intentionally broader than the final feature set.
#
# This is an inspection script. The goal is to confirm what is actually present
# in the downloaded NHANES files before we freeze any mappings or derived
# variables.

CANDIDATE_VARIABLES = {
    "DEMO_I": (
        # Identifiers / survey design
        "SEQN",
        "WTINT2YR",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",

        # Candidate project features
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDHHIN2",
        "INDFMPIR",
    ),

    "BMX_I": (
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
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

    "GHB_I": (
        "SEQN",
        "LBXGH",
    ),

    "GLU_I": (
        "SEQN",
        "WTSAF2YR",
        "LBXGLU",
    ),

    "INS_I": (
        "SEQN",
        "WTSAF2YR",
        "LBXIN",
    ),

    "OGTT_I": (
        "SEQN",
        "WTSAF2YR",
        "LBXGLT",
    ),

    "DIQ_I": (
        "SEQN",
        "DIQ010",
    ),

    "PAQ_I": (
        "SEQN",

        # Vigorous work
        "PAQ605",
        "PAQ610",
        "PAD615",

        # Moderate work
        "PAQ620",
        "PAQ625",
        "PAD630",

        # Walking / bicycling for transportation
        "PAQ635",
        "PAQ640",
        "PAD645",

        # Vigorous recreation
        "PAQ650",
        "PAQ655",
        "PAD660",

        # Moderate recreation
        "PAQ665",
        "PAQ670",
        "PAD675",

        # Screen time
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
        "ALQ120Q",
        "ALQ120U",
        "ALQ130",
    ),

    "OCQ_I": (
        "SEQN",
        "OCD150",
        "OCQ180",
    ),

    "BPQ_I": (
        "SEQN",
        "BPQ020",
    ),

    "MCQ_I": (
        "SEQN",

        # Cardiovascular history candidates
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",

        # Family history of diabetes
        "MCQ300C",
    ),
}


# =============================================================================
# Value-code inspection
# =============================================================================

# These variables represent categorical/questionnaire responses whose numeric
# codes must be understood before they are recoded into the project schema.

CODE_AUDIT_VARIABLES = {
    "DEMO_I": (
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDHHIN2",
    ),

    "DIQ_I": (
        "DIQ010",
    ),

    "OCQ_I": (
        "OCD150",
        "OCQ180",
    ),

    "SMQ_I": (
        "SMQ020",
        "SMQ040",
    ),

    "ALQ_I": (
        "ALQ120U",
    ),

    "PAQ_I": (
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
        "PAQ710",
        "PAQ715",
    ),

    "BPQ_I": (
        "BPQ020",
    ),

    "MCQ_I": (
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
        "MCQ300C",
    ),
}


# Numeric questionnaire values used when constructing derived features.
#
# These are summarized rather than value-counted because variables such as
# activity duration can contain many distinct values.

DERIVATION_NUMERIC_VARIABLES = {
    "ALQ_I": (
        "ALQ120Q",
        "ALQ130",
    ),

    "PAQ_I": (
        "PAQ610",
        "PAD615",
        "PAQ625",
        "PAD630",
        "PAQ640",
        "PAD645",
        "PAQ655",
        "PAD660",
        "PAQ670",
        "PAD675",
    ),
}

# =============================================================================
# Candidate project-feature mappings
# =============================================================================


@dataclass(frozen=True)
class FeatureCandidate:
    project_feature: str
    component: str
    nhanes_variables: tuple[str, ...]
    mapping_type: str
    notes: str


FEATURE_CANDIDATES = (
    FeatureCandidate(
        "age",
        "DEMO_I",
        ("RIDAGEYR",),
        "direct",
        "Age in years.",
    ),

    FeatureCandidate(
        "gender",
        "DEMO_I",
        ("RIAGENDR",),
        "recode",
        "NHANES categorical code must be mapped to project categories.",
    ),

    FeatureCandidate(
        "ethnicity",
        "DEMO_I",
        ("RIDRETH3",),
        "recode",
        "NHANES race/Hispanic-origin categories require harmonization.",
    ),

    FeatureCandidate(
        "education_level",
        "DEMO_I",
        ("DMDEDUC2",),
        "recode",
        "Adult education categories require harmonization.",
    ),

    FeatureCandidate(
        "income_level",
        "DEMO_I",
        ("INDHHIN2", "INDFMPIR"),
        "choose/recode",
        "Two candidate representations: household income category or poverty ratio.",
    ),

    FeatureCandidate(
        "employment_status",
        "OCQ_I",
        ("OCD150",),
        "recode",
        "Type of work activity during the previous week.",
    ),

    FeatureCandidate(
        "smoking_status",
        "SMQ_I",
        ("SMQ020", "SMQ040"),
        "derived",
        "Can derive never/former/current smoking status.",
    ),

    FeatureCandidate(
        "alcohol_consumption_per_week",
        "ALQ_I",
        ("ALQ120Q", "ALQ120U"),
        "derived",
        "Convert reported frequency and unit to a common weekly frequency.",
    ),

    FeatureCandidate(
        "physical_activity_minutes_per_week",
        "PAQ_I",
        (
            "PAQ610",
            "PAD615",
            "PAQ625",
            "PAD630",
            "PAQ640",
            "PAD645",
            "PAQ655",
            "PAD660",
            "PAQ670",
            "PAD675",
        ),
        "derived",
        "Potential weekly activity total from work, transport, and recreation.",
    ),

    FeatureCandidate(
        "diet_score",
        "",
        (),
        "unavailable",
        "No direct equivalent in currently downloaded components.",
    ),

    FeatureCandidate(
        "sleep_hours_per_day",
        "SLQ_I",
        ("SLD012",),
        "direct",
        "Usual weekday/workday sleep duration.",
    ),

    FeatureCandidate(
        "screen_time_hours_per_day",
        "PAQ_I",
        ("PAQ710", "PAQ715"),
        "derived",
        "TV/video and computer-use categories; not a direct continuous match.",
    ),

    FeatureCandidate(
        "family_history_diabetes",
        "MCQ_I",
        ("MCQ300C",),
        "recode",
        "Close biological relative reported to have diabetes.",
    ),

    FeatureCandidate(
        "hypertension_history",
        "BPQ_I",
        ("BPQ020",),
        "recode",
        "Ever told by a health professional of hypertension.",
    ),

    FeatureCandidate(
        "cardiovascular_history",
        "MCQ_I",
        (
            "MCQ160B",
            "MCQ160C",
            "MCQ160D",
            "MCQ160E",
        ),
        "derived",
        "Candidate composite of major cardiovascular diagnoses.",
    ),

    FeatureCandidate(
        "bmi",
        "BMX_I",
        ("BMXBMI",),
        "direct",
        "Measured body mass index.",
    ),

    FeatureCandidate(
        "waist_to_hip_ratio",
        "BMX_I",
        ("BMXWAIST",),
        "incomplete",
        "Waist circumference exists, but no matching hip circumference has been selected.",
    ),

    FeatureCandidate(
        "systolic_bp",
        "BPX_I",
        (
            "BPXSY1",
            "BPXSY2",
            "BPXSY3",
            "BPXSY4",
        ),
        "derived",
        "Use an average of available valid systolic readings.",
    ),

    FeatureCandidate(
        "diastolic_bp",
        "BPX_I",
        (
            "BPXDI1",
            "BPXDI2",
            "BPXDI3",
            "BPXDI4",
        ),
        "derived",
        "Use an average of available valid diastolic readings.",
    ),

    FeatureCandidate(
        "heart_rate",
        "BPX_I",
        ("BPXPLS",),
        "direct",
        "Measured 60-second pulse.",
    ),

    FeatureCandidate(
        "cholesterol_total",
        "TCHOL_I",
        ("LBXTC",),
        "direct",
        "Measured total cholesterol in mg/dL.",
    ),

    FeatureCandidate(
        "hdl_cholesterol",
        "HDL_I",
        ("LBDHDD",),
        "direct",
        "Measured direct HDL cholesterol in mg/dL.",
    ),

    FeatureCandidate(
        "ldl_cholesterol",
        "TRIGLY_I",
        ("LBDLDL",),
        "direct",
        "Derived LDL cholesterol in mg/dL from the fasting subsample.",
    ),

    FeatureCandidate(
        "triglycerides",
        "TRIGLY_I",
        ("LBXTR",),
        "direct",
        "Measured triglycerides in mg/dL from the fasting subsample.",
    ),

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    FeatureCandidate(
        "diagnosed_diabetes",
        "DIQ_I",
        ("DIQ010",),
        "recode",
        "Doctor/health-professional diagnosed diabetes; pregnancy-only diabetes excluded.",
    ),

    # -------------------------------------------------------------------------
    # Source-only clinical variables that NHANES may also support
    # -------------------------------------------------------------------------

    FeatureCandidate(
        "glucose_fasting",
        "GLU_I",
        ("LBXGLU",),
        "direct",
        "Measured fasting plasma glucose in mg/dL.",
    ),

    FeatureCandidate(
        "glucose_postprandial",
        "OGTT_I",
        ("LBXGLT",),
        "approximate",
        "Two-hour oral glucose tolerance test measurement.",
    ),

    FeatureCandidate(
        "insulin_level",
        "INS_I",
        ("LBXIN",),
        "direct",
        "Measured fasting insulin.",
    ),

    FeatureCandidate(
        "hba1c",
        "GHB_I",
        ("LBXGH",),
        "direct",
        "Measured glycohemoglobin percentage.",
    ),
)


# =============================================================================
# Merge order
# =============================================================================

# Broad/full-sample components first.
#
# Fasting-subsample components are intentionally placed last so the output
# makes their effect on participant coverage obvious.

MERGE_ORDER = (
    "DEMO_I",
    "DIQ_I",
    "BMX_I",
    "BPX_I",
    "TCHOL_I",
    "HDL_I",
    "BPQ_I",
    "MCQ_I",
    "SMQ_I",
    "ALQ_I",
    "SLQ_I",
    "OCQ_I",
    "PAQ_I",
    "GHB_I",
    "TRIGLY_I",
    "GLU_I",
    "INS_I",
    "OGTT_I",
)


# =============================================================================
# Utility functions
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


def component_path(component: str) -> Path:
    return NHANES_DIR / f"{component}.XPT"


def require_input_files() -> None:
    missing = [
        component_path(component)
        for component in EXPECTED_COMPONENTS
        if not component_path(component).exists()
    ]

    if missing:
        message = [
            "Missing required NHANES component files:",
            "",
        ]

        message.extend(
            f"  {path.relative_to(ROOT)}"
            for path in missing
        )

        raise FileNotFoundError(
            "\n".join(message)
        )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize NHANES/SAS variable names to uppercase strings.

    NHANES variable names are conventionally uppercase. Keeping this
    normalization here prevents accidental case mismatches during inspection.
    """

    df = df.copy()

    df.columns = [
        str(column).upper()
        for column in df.columns
    ]

    return df


# =============================================================================
# Loading
# =============================================================================


def load_component(component: str) -> pd.DataFrame:
    path = component_path(component)

    df = pd.read_sas(
        path,
        format="xport",
    )

    df = normalize_column_names(df)

    if "SEQN" not in df.columns:
        raise ValueError(
            f"{component} does not contain SEQN."
        )

    return df


def load_all_components() -> dict[str, pd.DataFrame]:
    print_header("LOADING NHANES COMPONENTS")

    components = {}

    for component in EXPECTED_COMPONENTS:
        path = component_path(component)

        print(
            f"{component:<10} "
            f"{path.relative_to(ROOT)}"
        )

        df = load_component(component)

        components[component] = df

        print(
            f"           rows={len(df):,}, "
            f"columns={len(df.columns):,}"
        )

    return components


# =============================================================================
# SEQN validation
# =============================================================================


def validate_seqn(
    component: str,
    df: pd.DataFrame,
) -> dict:
    seqn = df["SEQN"]

    missing_seqn = int(seqn.isna().sum())
    duplicate_seqn = int(seqn.duplicated().sum())
    unique_seqn = int(seqn.nunique(dropna=True))

    return {
        "component": component,
        "rows": len(df),
        "columns": len(df.columns),
        "unique_seqn": unique_seqn,
        "missing_seqn": missing_seqn,
        "duplicate_seqn": duplicate_seqn,
    }


def analyze_components(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("COMPONENT / SEQN VALIDATION")

    rows = []

    for component in EXPECTED_COMPONENTS:
        summary = validate_seqn(
            component,
            components[component],
        )

        rows.append(summary)

        status = (
            "OK"
            if (
                summary["missing_seqn"] == 0
                and summary["duplicate_seqn"] == 0
            )
            else "CHECK"
        )

        print(
            f"{component:<10} "
            f"rows={summary['rows']:>6,}  "
            f"unique={summary['unique_seqn']:>6,}  "
            f"missing={summary['missing_seqn']:>4,}  "
            f"duplicates={summary['duplicate_seqn']:>4,}  "
            f"{status}"
        )

    result = pd.DataFrame(rows)

    bad = result.loc[
        (result["missing_seqn"] > 0)
        | (result["duplicate_seqn"] > 0)
    ]

    if not bad.empty:
        raise ValueError(
            "One or more NHANES components have invalid SEQN values. "
            "See component_summary.csv."
        )

    return result


# =============================================================================
# Column inventory
# =============================================================================


def build_component_column_inventory(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for component, df in components.items():
        for column in df.columns:
            rows.append(
                {
                    "component": component,
                    "column": column,
                    "dtype": str(df[column].dtype),
                    "rows": len(df),
                    "nonmissing": int(
                        df[column].notna().sum()
                    ),
                    "missing": int(
                        df[column].isna().sum()
                    ),
                    "nonmissing_fraction": float(
                        df[column].notna().mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Candidate-variable inspection
# =============================================================================


def inspect_candidate_variables(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("CANDIDATE VARIABLE INVENTORY")

    rows = []

    for component in EXPECTED_COMPONENTS:
        df = components[component]

        print(f"\n{component}")

        expected = CANDIDATE_VARIABLES.get(
            component,
            ("SEQN",),
        )

        for variable in expected:
            exists = variable in df.columns

            if exists:
                nonmissing = int(
                    df[variable].notna().sum()
                )

                missing = int(
                    df[variable].isna().sum()
                )

                nonmissing_fraction = float(
                    df[variable].notna().mean()
                )

                dtype = str(
                    df[variable].dtype
                )

                print(
                    f"  [OK]      {variable:<12} "
                    f"nonmissing={nonmissing:>6,} "
                    f"({nonmissing_fraction:>6.1%})"
                )

            else:
                nonmissing = 0
                missing = len(df)
                nonmissing_fraction = 0.0
                dtype = ""

                print(
                    f"  [MISSING] {variable}"
                )

            rows.append(
                {
                    "component": component,
                    "variable": variable,
                    "exists": exists,
                    "dtype": dtype,
                    "rows": len(df),
                    "nonmissing": nonmissing,
                    "missing": missing,
                    "nonmissing_fraction":
                        nonmissing_fraction,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Feature-mapping candidate report
# =============================================================================


def build_feature_mapping_report(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print_header("PROJECT FEATURE MAPPING CANDIDATES")

    rows = []

    for candidate in FEATURE_CANDIDATES:
        if not candidate.component:
            found = []
            missing = []
            all_present = False

        else:
            columns = set(
                components[
                    candidate.component
                ].columns
            )

            found = [
                variable
                for variable
                in candidate.nhanes_variables
                if variable in columns
            ]

            missing = [
                variable
                for variable
                in candidate.nhanes_variables
                if variable not in columns
            ]

            all_present = (
                len(candidate.nhanes_variables) > 0
                and len(missing) == 0
            )

        if candidate.mapping_type == "unavailable":
            status = "UNAVAILABLE"
        elif all_present:
            status = "INPUTS PRESENT"
        elif found:
            status = "PARTIAL"
        else:
            status = "MISSING"

        print(
            f"{candidate.project_feature:<38} "
            f"{status:<16} "
            f"{candidate.mapping_type}"
        )

        rows.append(
            {
                "project_feature":
                    candidate.project_feature,

                "component":
                    candidate.component,

                "nhanes_variables":
                    ",".join(
                        candidate.nhanes_variables
                    ),

                "found_variables":
                    ",".join(found),

                "missing_variables":
                    ",".join(missing),

                "mapping_type":
                    candidate.mapping_type,

                "status":
                    status,

                "notes":
                    candidate.notes,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Component overlap / merge survival
# =============================================================================


def adult_seqn_sets(
    demo: pd.DataFrame,
) -> tuple[set, set, set]:
    if "RIDAGEYR" not in demo.columns:
        raise ValueError(
            "DEMO_I does not contain RIDAGEYR."
        )

    all_seqn = set(
        demo["SEQN"].dropna().tolist()
    )

    adult_18_seqn = set(
        demo.loc[
            demo["RIDAGEYR"] >= 18,
            "SEQN",
        ]
        .dropna()
        .tolist()
    )

    adult_20_seqn = set(
        demo.loc[
            demo["RIDAGEYR"] >= 20,
            "SEQN",
        ]
        .dropna()
        .tolist()
    )

    return (
        all_seqn,
        adult_18_seqn,
        adult_20_seqn,
    )


def analyze_merge_survival(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Report participant overlap as NHANES components are progressively added.

    This uses only SEQN membership.

    It deliberately does NOT perform complete-case filtering on questionnaire
    variables because NHANES skip patterns can produce structurally missing
    values that must be interpreted during later recoding.
    """

    print_header("PROGRESSIVE COMPONENT OVERLAP")

    (
        all_remaining,
        adults_18_remaining,
        adults_20_remaining,
    ) = adult_seqn_sets(
        components["DEMO_I"]
    )

    initial_all = len(all_remaining)
    initial_18 = len(adults_18_remaining)
    initial_20 = len(adults_20_remaining)

    rows = []

    for step, component in enumerate(
        MERGE_ORDER,
        start=1,
    ):
        component_seqn = set(
            components[component][
                "SEQN"
            ]
            .dropna()
            .tolist()
        )

        all_remaining &= component_seqn
        adults_18_remaining &= component_seqn
        adults_20_remaining &= component_seqn

        row = {
            "step": step,
            "component": component,

            "all_rows_remaining":
                len(all_remaining),

            "age18plus_rows_remaining":
                len(adults_18_remaining),

            "age20plus_rows_remaining":
                len(adults_20_remaining),

            "all_fraction_of_demo":
                (
                    len(all_remaining)
                    / initial_all
                    if initial_all
                    else 0.0
                ),

            "age18plus_fraction":
                (
                    len(adults_18_remaining)
                    / initial_18
                    if initial_18
                    else 0.0
                ),

            "age20plus_fraction":
                (
                    len(adults_20_remaining)
                    / initial_20
                    if initial_20
                    else 0.0
                ),
        }

        rows.append(row)

        print(
            f"{step:>2}. {component:<10} "
            f"all={len(all_remaining):>6,}  "
            f"18+={len(adults_18_remaining):>6,}  "
            f"20+={len(adults_20_remaining):>6,}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# Important-variable summary
# =============================================================================


def print_high_value_variables(
    components: dict[str, pd.DataFrame],
) -> None:
    print_header("HIGH-VALUE VARIABLE COVERAGE")

    checks = (
        ("DIQ_I", "DIQ010", "diagnosed diabetes"),
        ("BMX_I", "BMXBMI", "BMI"),
        ("BPX_I", "BPXPLS", "heart rate"),
        ("TCHOL_I", "LBXTC", "total cholesterol"),
        ("HDL_I", "LBDHDD", "HDL cholesterol"),
        ("TRIGLY_I", "LBDLDL", "LDL cholesterol"),
        ("TRIGLY_I", "LBXTR", "triglycerides"),
        ("GHB_I", "LBXGH", "HbA1c"),
        ("GLU_I", "LBXGLU", "fasting glucose"),
        ("INS_I", "LBXIN", "fasting insulin"),
        ("OGTT_I", "LBXGLT", "2-hour glucose"),
        ("SLQ_I", "SLD012", "sleep duration"),
        ("MCQ_I", "MCQ300C", "family diabetes history"),
        ("BPQ_I", "BPQ020", "hypertension history"),
    )

    for component, variable, description in checks:
        df = components[component]

        if variable not in df.columns:
            print(
                f"{description:<28} "
                f"{component}/{variable:<12} MISSING"
            )
            continue

        nonmissing = int(
            df[variable].notna().sum()
        )

        print(
            f"{description:<28} "
            f"{component}/{variable:<12} "
            f"{nonmissing:>6,} nonmissing"
        )


# =============================================================================
# Save outputs
# =============================================================================

def save_results(
    component_summary: pd.DataFrame,
    component_columns: pd.DataFrame,
    variable_summary: pd.DataFrame,
    merge_survival: pd.DataFrame,
    feature_mapping: pd.DataFrame,
    value_code_audit: pd.DataFrame,
    derivation_summary: pd.DataFrame,
) -> None:
    print_header("SAVING INSPECTION OUTPUTS")

    component_summary.to_csv(
        COMPONENT_SUMMARY_PATH,
        index=False,
    )

    component_columns.to_csv(
        COMPONENT_COLUMNS_PATH,
        index=False,
    )

    variable_summary.to_csv(
        VARIABLE_SUMMARY_PATH,
        index=False,
    )

    merge_survival.to_csv(
        MERGE_SURVIVAL_PATH,
        index=False,
    )

    feature_mapping.to_csv(
        FEATURE_MAPPING_PATH,
        index=False,
    )

    value_code_audit.to_csv(
        VALUE_CODE_AUDIT_PATH,
        index=False,
    )

    derivation_summary.to_csv(
        DERIVATION_SUMMARY_PATH,
        index=False,
    )

    for path in (
        COMPONENT_SUMMARY_PATH,
        COMPONENT_COLUMNS_PATH,
        VARIABLE_SUMMARY_PATH,
        MERGE_SURVIVAL_PATH,
        FEATURE_MAPPING_PATH,
        VALUE_CODE_AUDIT_PATH,
        DERIVATION_SUMMARY_PATH,
    ):
        print(path.relative_to(ROOT))

# =============================================================================
# Value-code audit
# =============================================================================


def build_value_code_audit(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Record observed response codes for categorical NHANES questionnaire fields.

    No interpretation or recoding is performed here. The purpose of this
    report is to expose the raw codes and their frequencies before mappings
    are frozen in prepare_nhanes.py.
    """

    print_header("NHANES VALUE-CODE AUDIT")

    rows = []

    for component, variables in CODE_AUDIT_VARIABLES.items():
        df = components[component]

        print(f"\n{component}")

        for variable in variables:
            if variable not in df.columns:
                print(f"\n  {variable}: MISSING")

                rows.append(
                    {
                        "component": component,
                        "variable": variable,
                        "value": "<VARIABLE MISSING>",
                        "count": 0,
                        "fraction_of_nonmissing": 0.0,
                        "missing_count": len(df),
                    }
                )

                continue

            series = df[variable]

            nonmissing = series.dropna()

            missing_count = int(
                series.isna().sum()
            )

            counts = (
                nonmissing
                .value_counts(dropna=False)
                .sort_index()
            )

            total_nonmissing = len(nonmissing)

            print(
                f"\n  {variable}"
                f"  nonmissing={total_nonmissing:,}"
                f"  missing={missing_count:,}"
            )

            for value, count in counts.items():
                fraction = (
                    count / total_nonmissing
                    if total_nonmissing
                    else 0.0
                )

                print(
                    f"    {value!s:<12} "
                    f"{count:>6,} "
                    f"({fraction:>6.2%})"
                )

                rows.append(
                    {
                        "component": component,
                        "variable": variable,
                        "value": value,
                        "count": int(count),
                        "fraction_of_nonmissing":
                            float(fraction),
                        "missing_count":
                            missing_count,
                    }
                )

            # Explicit missing row so missingness is retained in the saved audit.
            if missing_count:
                rows.append(
                    {
                        "component": component,
                        "variable": variable,
                        "value": "<MISSING>",
                        "count": missing_count,
                        "fraction_of_nonmissing": None,
                        "missing_count": missing_count,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Numeric derivation-variable audit
# =============================================================================


def build_derivation_numeric_summary(
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize numeric questionnaire variables that may be combined into
    derived project features.

    Examples include activity frequency/duration and alcohol frequency.
    """

    print_header("DERIVATION NUMERIC VARIABLE SUMMARY")

    rows = []

    for component, variables in DERIVATION_NUMERIC_VARIABLES.items():
        df = components[component]

        print(f"\n{component}")

        for variable in variables:
            if variable not in df.columns:
                print(f"  {variable:<12} MISSING")

                rows.append(
                    {
                        "component": component,
                        "variable": variable,
                        "rows": len(df),
                        "nonmissing": 0,
                        "missing": len(df),
                        "unique_values": 0,
                        "min": None,
                        "median": None,
                        "mean": None,
                        "max": None,
                    }
                )

                continue

            series = pd.to_numeric(
                df[variable],
                errors="coerce",
            )

            valid = series.dropna()

            row = {
                "component": component,
                "variable": variable,
                "rows": len(df),
                "nonmissing": int(valid.size),
                "missing": int(series.isna().sum()),
                "unique_values": int(valid.nunique()),
                "min": (
                    float(valid.min())
                    if not valid.empty
                    else None
                ),
                "median": (
                    float(valid.median())
                    if not valid.empty
                    else None
                ),
                "mean": (
                    float(valid.mean())
                    if not valid.empty
                    else None
                ),
                "max": (
                    float(valid.max())
                    if not valid.empty
                    else None
                ),
            }

            rows.append(row)

            print(
                f"  {variable:<12} "
                f"n={row['nonmissing']:>6,}  "
                f"unique={row['unique_values']:>4,}  "
                f"min={row['min']}  "
                f"median={row['median']}  "
                f"max={row['max']}"
            )

    return pd.DataFrame(rows)



# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_output_directory()

    print_header("NHANES 2015-2016 INSPECTION")

    print(
        f"Input directory:\n"
        f"  {NHANES_DIR.relative_to(ROOT)}"
    )

    require_input_files()

    components = load_all_components()

    component_summary = analyze_components(
        components
    )

    component_columns = (
        build_component_column_inventory(
            components
        )
    )

    variable_summary = (
        inspect_candidate_variables(
            components
        )
    )

    feature_mapping = (
        build_feature_mapping_report(
            components
        )
    )

    merge_survival = (
        analyze_merge_survival(
            components
        )
    )

    value_code_audit = (
        build_value_code_audit(
            components
        )
    )

    derivation_summary = (
        build_derivation_numeric_summary(
            components
        )
    )

    print_high_value_variables(
        components
    )

    save_results(
        component_summary,
        component_columns,
        variable_summary,
        merge_survival,
        feature_mapping,
        value_code_audit,
        derivation_summary,
    )

    print_header("NHANES INSPECTION COMPLETE")

    print(
        "No NHANES values were recoded, imputed, "
        "or written to a prepared dataset."
    )

    print(
        "\nThe next step is to use these diagnostics to freeze "
        "the NHANES-to-project feature mappings before creating "
        "a canonical NHANES dataset."
    )


if __name__ == "__main__":
    main()