from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import ks_2samp

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    raise ImportError(
        "CatBoost is required for the domain feature-ablation experiment.\n"
        "Install it with:\n"
        "  python3 -m pip install catboost"
    ) from exc

from data import load_common, load_schema


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DOMAIN_CLASSIFIER_DIR = ROOT / "results/domain_classifier"
OUTPUT_DIR = ROOT / "results/domain_feature_ablation"
FIGURES_DIR = OUTPUT_DIR / "figures"

DOMAIN_METRICS_PATH = DOMAIN_CLASSIFIER_DIR / "metrics.json"
DOMAIN_IMPORTANCE_PATH = DOMAIN_CLASSIFIER_DIR / "feature_importance.csv"
SAMPLING_PROVENANCE_PATH = DOMAIN_CLASSIFIER_DIR / "sampling_provenance.csv.gz"

ABLATION_RESULTS_PATH = OUTPUT_DIR / "ablation_results.csv"
REMOVED_FEATURES_PATH = OUTPUT_DIR / "removed_features.csv"
TRAINING_RANKING_METRICS_PATH = OUTPUT_DIR / "training_ranking_metrics.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
AUC_FIGURE_PATH = FIGURES_DIR / "auc_vs_removed_features.png"
ACCURACY_FIGURE_PATH = FIGURES_DIR / "accuracy_vs_removed_features.png"


# =============================================================================
# Configuration
# =============================================================================

Split = Literal["train", "val", "test"]

DOMAIN_SOURCE = 0
DOMAIN_KAGGLE = 1

ABLATION_COUNTS = (1, 3, 5, 10)

# Rankings are fixed before the ablation models are trained and are not
# recomputed after every removal step. The learned domain-importance ranking
# comes from the original domain classifier. Statistical rankings (KS and
# variance distortion) are computed ONLY from the reconstructed training split,
# so the held-out domain test split cannot influence feature selection.
STRATEGIES = (
    "domain_importance",
    "ks_statistic",
    "variance_distortion",
)

BASELINE_AUC_TOLERANCE = 1e-4


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class DomainDataset:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass
class ExperimentInputs:
    datasets: dict[Split, DomainDataset]
    schema: object
    original_metrics: dict
    model_parameters: dict
    early_stopping_rounds: int
    rankings: dict[str, list[str]]
    ranking_metrics: pd.DataFrame


# =============================================================================
# Utility functions
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)



def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)



def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist:\n  {path}\n"
            "Run the earlier domain-classifier script first."
        )



def normalize_catboost_inputs(
    X: pd.DataFrame,
    categorical_features: tuple[str, ...],
) -> pd.DataFrame:
    X = X.copy()

    for feature in categorical_features:
        if feature in X.columns:
            X[feature] = X[feature].astype(str)

    return X


# =============================================================================
# Reconstruct the exact original balanced domain datasets
# =============================================================================


def validate_provenance(
    provenance: pd.DataFrame,
    source_rows: int,
    kaggle_rows: int,
) -> None:
    required_columns = {
        "split",
        "original_domain",
        "original_row_index",
        "domain_label",
    }

    missing = required_columns - set(provenance.columns)

    if missing:
        raise ValueError(
            "sampling_provenance.csv.gz is missing required columns: "
            + ", ".join(sorted(missing))
        )

    valid_splits = {"train", "val", "test"}
    observed_splits = set(provenance["split"].astype(str).unique())

    if observed_splits != valid_splits:
        raise ValueError(
            "Unexpected split labels in sampling provenance. "
            f"Expected {sorted(valid_splits)}, observed {sorted(observed_splits)}."
        )

    valid_domains = {"source", "kaggle"}
    observed_domains = set(provenance["original_domain"].astype(str).unique())

    if observed_domains != valid_domains:
        raise ValueError(
            "Unexpected domain labels in sampling provenance. "
            f"Expected {sorted(valid_domains)}, observed {sorted(observed_domains)}."
        )

    expected_labels = provenance["original_domain"].map(
        {
            "source": DOMAIN_SOURCE,
            "kaggle": DOMAIN_KAGGLE,
        }
    )

    if not np.array_equal(
        expected_labels.to_numpy(dtype=np.int64),
        provenance["domain_label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "Domain labels in sampling provenance are inconsistent with "
            "original_domain."
        )

    row_indices = provenance["original_row_index"].to_numpy(dtype=np.int64)

    if np.any(row_indices < 0):
        raise ValueError("Sampling provenance contains negative row indices.")

    source_mask = provenance["original_domain"].eq("source").to_numpy()
    kaggle_mask = provenance["original_domain"].eq("kaggle").to_numpy()

    if np.any(row_indices[source_mask] >= source_rows):
        raise ValueError("Sampling provenance contains out-of-range source indices.")

    if np.any(row_indices[kaggle_mask] >= kaggle_rows):
        raise ValueError("Sampling provenance contains out-of-range Kaggle indices.")

    duplicate_keys = provenance.duplicated(
        subset=["split", "original_domain", "original_row_index"]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Sampling provenance contains duplicate rows within a split/domain."
        )



def reconstruct_domain_split(
    split: Split,
    provenance: pd.DataFrame,
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: tuple[str, ...],
) -> DomainDataset:
    split_provenance = (
        provenance.loc[provenance["split"].eq(split)]
        .reset_index(drop=True)
        .copy()
    )

    if split_provenance.empty:
        raise ValueError(f"No provenance rows found for split '{split}'.")

    source_mask = split_provenance["original_domain"].eq("source").to_numpy()
    kaggle_mask = split_provenance["original_domain"].eq("kaggle").to_numpy()

    source_positions = np.flatnonzero(source_mask)
    kaggle_positions = np.flatnonzero(kaggle_mask)

    source_indices = split_provenance.loc[
        source_mask,
        "original_row_index",
    ].to_numpy(dtype=np.int64)

    kaggle_indices = split_provenance.loc[
        kaggle_mask,
        "original_row_index",
    ].to_numpy(dtype=np.int64)

    source_rows = source.iloc[source_indices][feature_columns].copy()
    source_rows.insert(0, "_position", source_positions)

    kaggle_rows = kaggle.iloc[kaggle_indices][feature_columns].copy()
    kaggle_rows.insert(0, "_position", kaggle_positions)

    X = (
        pd.concat([source_rows, kaggle_rows], ignore_index=True)
        .sort_values("_position")
        .drop(columns="_position")
        .reset_index(drop=True)
    )

    X = normalize_catboost_inputs(
        X,
        categorical_features,
    )

    y = split_provenance["domain_label"].astype(np.int64).reset_index(drop=True)

    source_count = int((y == DOMAIN_SOURCE).sum())
    kaggle_count = int((y == DOMAIN_KAGGLE).sum())

    if source_count != kaggle_count:
        raise ValueError(
            f"Reconstructed {split} split is not balanced: "
            f"source={source_count:,}, kaggle={kaggle_count:,}."
        )

    return DomainDataset(
        X=X,
        y=y,
        metadata=split_provenance,
    )



def reconstruct_domain_datasets(
    provenance: pd.DataFrame,
    source: pd.DataFrame,
    kaggle: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: tuple[str, ...],
) -> dict[Split, DomainDataset]:
    return {
        split: reconstruct_domain_split(
            split,
            provenance,
            source,
            kaggle,
            feature_columns,
            categorical_features,
        )
        for split in ("train", "val", "test")
    }


# =============================================================================
# Feature-ranking inputs
# =============================================================================


def calculate_variance_distortion(
    source_std: float,
    kaggle_std: float,
) -> tuple[float, float]:
    """
    Return (std_ratio, symmetric variance-distortion magnitude).

    variance_distortion = abs(log(kaggle_std / source_std))

    A 2x expansion and a 2x contraction therefore receive the same magnitude.
    """

    if source_std <= 0:
        return np.nan, np.nan

    std_ratio = kaggle_std / source_std

    if not np.isfinite(std_ratio) or std_ratio <= 0:
        return std_ratio, np.nan

    return (
        float(std_ratio),
        float(abs(np.log(std_ratio))),
    )


def calculate_training_ranking_metrics(
    train: DomainDataset,
    numeric_features: tuple[str, ...],
) -> pd.DataFrame:
    """
    Compute statistical feature-selection metrics from TRAINING DATA ONLY.

    analyze_distribution_shift.py may still use the complete prepared datasets
    for descriptive analysis. Those full-dataset statistics are intentionally
    not used here because the held-out domain test distribution must not
    influence which features are removed before test evaluation.
    """

    source_mask = train.y.eq(DOMAIN_SOURCE).to_numpy()
    kaggle_mask = train.y.eq(DOMAIN_KAGGLE).to_numpy()

    rows = []

    for feature in numeric_features:
        source_values = train.X.loc[
            source_mask,
            feature,
        ].to_numpy(dtype=np.float64)

        kaggle_values = train.X.loc[
            kaggle_mask,
            feature,
        ].to_numpy(dtype=np.float64)

        if len(source_values) == 0 or len(kaggle_values) == 0:
            raise ValueError(
                f"Training split has no observations for one domain while "
                f"ranking feature '{feature}'."
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

        std_ratio, distortion = calculate_variance_distortion(
            source_std,
            kaggle_std,
        )

        ks_result = ks_2samp(
            source_values,
            kaggle_values,
            alternative="two-sided",
            method="auto",
        )

        rows.append(
            {
                "feature": feature,
                "source_train_rows": int(len(source_values)),
                "kaggle_train_rows": int(len(kaggle_values)),
                "source_std": source_std,
                "kaggle_std": kaggle_std,
                "std_ratio_kaggle_to_source": std_ratio,
                "variance_distortion": distortion,
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
            }
        )

    metrics = pd.DataFrame(rows)

    metrics["ks_rank"] = (
        metrics["ks_statistic"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    metrics["variance_distortion_rank"] = (
        metrics["variance_distortion"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    return metrics


def load_feature_rankings(
    all_features: list[str],
    train: DomainDataset,
    numeric_features: tuple[str, ...],
) -> tuple[
    dict[str, list[str]],
    pd.DataFrame,
]:
    """
    Build fixed ablation rankings without using the held-out test distribution.

    domain_importance:
        Loaded from the original CatBoost domain classifier, which was trained
        on the training split with validation used for early stopping.

    ks_statistic / variance_distortion:
        Recomputed from the exact reconstructed balanced TRAIN split used by
        the domain-classifier experiment.
    """

    require_file(DOMAIN_IMPORTANCE_PATH)

    importance = pd.read_csv(
        DOMAIN_IMPORTANCE_PATH
    )

    expected = set(all_features)

    if set(importance["feature"]) != expected:
        raise ValueError(
            "feature_importance.csv does not contain exactly the prepared feature set."
        )

    domain_importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        ["feature"]
        .tolist()
    )

    ranking_metrics = calculate_training_ranking_metrics(
        train,
        numeric_features,
    )

    expected_numeric = set(numeric_features)

    if set(ranking_metrics["feature"]) != expected_numeric:
        raise ValueError(
            "Train-only statistical ranking metrics do not contain exactly "
            "the prepared numeric feature set."
        )

    ks_ranking = (
        ranking_metrics
        .sort_values(
            "ks_statistic",
            ascending=False,
        )
        ["feature"]
        .tolist()
    )

    variance_ranking = (
        ranking_metrics
        .sort_values(
            "variance_distortion",
            ascending=False,
        )
        ["feature"]
        .tolist()
    )

    rankings = {
        "domain_importance": domain_importance,
        "ks_statistic": ks_ranking,
        "variance_distortion": variance_ranking,
    }

    for strategy, ranking in rankings.items():
        if len(ranking) < max(ABLATION_COUNTS):
            raise ValueError(
                f"Ranking '{strategy}' only contains {len(ranking)} features, "
                f"but remove_n={max(ABLATION_COUNTS)} was requested."
            )

        if len(ranking) != len(set(ranking)):
            raise ValueError(
                f"Ranking '{strategy}' contains duplicate features."
            )

        unexpected = set(ranking) - expected

        if unexpected:
            raise ValueError(
                f"Ranking '{strategy}' contains unexpected features: "
                + ", ".join(sorted(unexpected))
            )

    return (
        rankings,
        ranking_metrics,
    )


# =============================================================================
# Experiment input loading
# =============================================================================


def load_experiment_inputs() -> ExperimentInputs:
    print_header("LOADING ABLATION INPUTS")

    require_file(DOMAIN_METRICS_PATH)
    require_file(SAMPLING_PROVENANCE_PATH)

    schema = load_schema()
    source = load_common("source")
    kaggle = load_common("kaggle")

    with DOMAIN_METRICS_PATH.open("r") as file:
        original_metrics = json.load(file)

    provenance = pd.read_csv(SAMPLING_PROVENANCE_PATH)

    validate_provenance(
        provenance,
        source_rows=len(source),
        kaggle_rows=len(kaggle),
    )

    feature_columns = list(schema.features)

    datasets = reconstruct_domain_datasets(
        provenance,
        source,
        kaggle,
        feature_columns,
        tuple(schema.categorical_features),
    )

    rankings, ranking_metrics = load_feature_rankings(
        feature_columns,
        datasets["train"],
        tuple(schema.numeric_features),
    )

    try:
        model_parameters = dict(original_metrics["model"]["parameters"])
        early_stopping_rounds = int(
            original_metrics["model"]["early_stopping_rounds"]
        )
    except KeyError as exc:
        raise ValueError(
            "metrics.json does not contain the model configuration required "
            "to reproduce the original domain classifier."
        ) from exc

    print(f"Features: {len(feature_columns)}")

    for split in ("train", "val", "test"):
        dataset = datasets[split]
        print(
            f"{split.capitalize():5s}: "
            f"{len(dataset.y):,} rows "
            f"({int((dataset.y == 0).sum()):,} source, "
            f"{int((dataset.y == 1).sum()):,} Kaggle)"
        )

    print("\nFixed removal rankings loaded:")
    print("  domain_importance: original CatBoost model (no test evaluation used)")
    print("  ks_statistic: train-only reconstructed domain split")
    print("  variance_distortion: train-only reconstructed domain split")

    for strategy, ranking in rankings.items():
        print(f"  {strategy}: {ranking[:5]}")

    return ExperimentInputs(
        datasets=datasets,
        schema=schema,
        original_metrics=original_metrics,
        model_parameters=model_parameters,
        early_stopping_rounds=early_stopping_rounds,
        rankings=rankings,
        ranking_metrics=ranking_metrics,
    )


# =============================================================================
# CatBoost training and evaluation
# =============================================================================


def create_pool(
    dataset: DomainDataset,
    selected_features: list[str],
    categorical_features: tuple[str, ...],
) -> Pool:
    selected_categorical = [
        feature
        for feature in categorical_features
        if feature in selected_features
    ]

    return Pool(
        data=dataset.X[selected_features],
        label=dataset.y,
        cat_features=selected_categorical,
    )



def train_model(
    train: DomainDataset,
    validation: DomainDataset,
    selected_features: list[str],
    categorical_features: tuple[str, ...],
    model_parameters: dict,
    early_stopping_rounds: int,
) -> CatBoostClassifier:
    train_pool = create_pool(
        train,
        selected_features,
        categorical_features,
    )

    validation_pool = create_pool(
        validation,
        selected_features,
        categorical_features,
    )

    model = CatBoostClassifier(**model_parameters)

    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
    )

    return model



def evaluate_model(
    model: CatBoostClassifier,
    dataset: DomainDataset,
    selected_features: list[str],
    categorical_features: tuple[str, ...],
) -> dict:
    pool = create_pool(
        dataset,
        selected_features,
        categorical_features,
    )

    probabilities = model.predict_proba(pool)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    return {
        "roc_auc": float(
            roc_auc_score(dataset.y, probabilities)
        ),
        "accuracy": float(
            accuracy_score(dataset.y, predictions)
        ),
        "log_loss": float(
            log_loss(dataset.y, probabilities)
        ),
        "precision": float(
            precision_score(dataset.y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(dataset.y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(dataset.y, predictions, zero_division=0)
        ),
    }


# =============================================================================
# Ablation experiment
# =============================================================================


def run_single_experiment(
    inputs: ExperimentInputs,
    strategy: str,
    remove_n: int,
    removed_features: list[str],
) -> dict:
    all_features = list(inputs.schema.features)
    removed_set = set(removed_features)

    selected_features = [
        feature
        for feature in all_features
        if feature not in removed_set
    ]

    if not selected_features:
        raise ValueError("Ablation removed every feature.")

    print_header(
        f"ABLATION: {strategy} / REMOVE {remove_n}"
    )

    if removed_features:
        print("Removed features:")
        for feature in removed_features:
            print(f"  - {feature}")
    else:
        print("Removed features: none")

    print(f"Remaining features: {len(selected_features)}")

    model = train_model(
        inputs.datasets["train"],
        inputs.datasets["val"],
        selected_features,
        tuple(inputs.schema.categorical_features),
        inputs.model_parameters,
        inputs.early_stopping_rounds,
    )

    validation_metrics = evaluate_model(
        model,
        inputs.datasets["val"],
        selected_features,
        tuple(inputs.schema.categorical_features),
    )

    test_metrics = evaluate_model(
        model,
        inputs.datasets["test"],
        selected_features,
        tuple(inputs.schema.categorical_features),
    )

    best_iteration = model.get_best_iteration()
    if best_iteration >= 0:
        best_iteration += 1

    print(
        f"Validation ROC-AUC: {validation_metrics['roc_auc']:.6f}\n"
        f"Test ROC-AUC:       {test_metrics['roc_auc']:.6f}\n"
        f"Test accuracy:      {test_metrics['accuracy']:.6f}\n"
        f"Best iteration:     {best_iteration}"
    )

    return {
        "strategy": strategy,
        "remove_n": remove_n,
        "remaining_features": len(selected_features),
        "removed_features": ";".join(removed_features),
        "best_iteration": best_iteration,
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_log_loss": validation_metrics["log_loss"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_log_loss": test_metrics["log_loss"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
    }



def check_baseline_consistency(
    baseline_result: dict,
    original_metrics: dict,
) -> None:
    try:
        original_test_auc = float(
            original_metrics["test_metrics"]["roc_auc"]
        )
    except KeyError as exc:
        raise ValueError(
            "Original metrics.json does not contain test_metrics.roc_auc."
        ) from exc

    reproduced_test_auc = float(
        baseline_result["test_roc_auc"]
    )

    difference = abs(
        reproduced_test_auc - original_test_auc
    )

    print_header("BASELINE CONSISTENCY CHECK")
    print(f"Original test ROC-AUC:   {original_test_auc:.9f}")
    print(f"Reproduced test ROC-AUC: {reproduced_test_auc:.9f}")
    print(f"Absolute difference:     {difference:.9f}")

    if difference > BASELINE_AUC_TOLERANCE:
        raise RuntimeError(
            "The reconstructed baseline does not match the original domain "
            "classifier closely enough. Refusing to continue the ablation "
            "because sampling/training conditions may differ."
        )

    print("Baseline reproduction passed.")



def run_ablation_experiments(
    inputs: ExperimentInputs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    removed_feature_rows = []

    # -------------------------------------------------------------------------
    # Shared baseline: train once using all 24 features.
    # -------------------------------------------------------------------------

    baseline = run_single_experiment(
        inputs,
        strategy="baseline",
        remove_n=0,
        removed_features=[],
    )

    check_baseline_consistency(
        baseline,
        inputs.original_metrics,
    )

    results.append(baseline)

    # -------------------------------------------------------------------------
    # Progressive removal using each fixed ranking.
    # -------------------------------------------------------------------------

    for strategy in STRATEGIES:
        ranking = inputs.rankings[strategy]

        for remove_n in ABLATION_COUNTS:
            removed_features = ranking[:remove_n]

            result = run_single_experiment(
                inputs,
                strategy=strategy,
                remove_n=remove_n,
                removed_features=removed_features,
            )

            results.append(result)

            for removal_rank, feature in enumerate(
                removed_features,
                start=1,
            ):
                removed_feature_rows.append(
                    {
                        "strategy": strategy,
                        "remove_n": remove_n,
                        "removal_rank": removal_rank,
                        "feature": feature,
                    }
                )

    results_df = pd.DataFrame(results)

    baseline_test_auc = float(
        results_df.loc[
            results_df["strategy"].eq("baseline"),
            "test_roc_auc",
        ].iloc[0]
    )

    baseline_validation_auc = float(
        results_df.loc[
            results_df["strategy"].eq("baseline"),
            "validation_roc_auc",
        ].iloc[0]
    )

    results_df["test_auc_drop_from_baseline"] = (
        baseline_test_auc - results_df["test_roc_auc"]
    )

    results_df["validation_auc_drop_from_baseline"] = (
        baseline_validation_auc - results_df["validation_roc_auc"]
    )

    removed_features_df = pd.DataFrame(removed_feature_rows)

    return results_df, removed_features_df


# =============================================================================
# Plotting
# =============================================================================


def make_plot_data(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results.loc[
        results["strategy"].eq("baseline")
    ]

    if len(baseline) != 1:
        raise ValueError("Expected exactly one baseline result.")

    rows = []

    for strategy in STRATEGIES:
        strategy_rows = results.loc[
            results["strategy"].eq(strategy)
        ].copy()

        strategy_baseline = baseline.copy()
        strategy_baseline["strategy"] = strategy

        rows.append(
            pd.concat(
                [strategy_baseline, strategy_rows],
                ignore_index=True,
            )
        )

    return pd.concat(rows, ignore_index=True)



def plot_metric_vs_removed_features(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plot_data = make_plot_data(results)

    fig, ax = plt.subplots(figsize=(7, 5))

    for strategy in STRATEGIES:
        strategy_data = (
            plot_data.loc[
                plot_data["strategy"].eq(strategy)
            ]
            .sort_values("remove_n")
        )

        ax.plot(
            strategy_data["remove_n"],
            strategy_data[metric],
            marker="o",
            label=strategy.replace("_", " "),
        )

    if metric.endswith("roc_auc"):
        ax.axhline(
            0.5,
            linestyle="--",
            linewidth=1,
            label="chance ROC-AUC",
        )

    ax.set_xlabel("Number of removed features")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([0, *ABLATION_COUNTS])
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def create_figures(results: pd.DataFrame) -> None:
    print_header("CREATING FIGURES")

    plot_metric_vs_removed_features(
        results,
        metric="test_roc_auc",
        ylabel="Test ROC-AUC",
        title="Domain Separability After Feature Removal",
        path=AUC_FIGURE_PATH,
    )

    plot_metric_vs_removed_features(
        results,
        metric="test_accuracy",
        ylabel="Test accuracy",
        title="Domain Classification Accuracy After Feature Removal",
        path=ACCURACY_FIGURE_PATH,
    )

    print(AUC_FIGURE_PATH.relative_to(ROOT))
    print(ACCURACY_FIGURE_PATH.relative_to(ROOT))


# =============================================================================
# Output and summaries
# =============================================================================


def save_metadata(
    inputs: ExperimentInputs,
    results: pd.DataFrame,
) -> None:
    original_auc = float(
        inputs.original_metrics["test_metrics"]["roc_auc"]
    )

    metadata = {
        "experiment": "domain_feature_ablation",
        "description": (
            "Progressive feature removal from a balanced source-vs-Kaggle "
            "domain classifier using fixed rankings selected without access "
            "to the held-out domain test distribution."
        ),
        "ablation_counts": list(ABLATION_COUNTS),
        "strategies": list(STRATEGIES),
        "baseline_original_test_roc_auc": original_auc,
        "baseline_auc_tolerance": BASELINE_AUC_TOLERANCE,
        "ranking_policy": {
            "general": (
                "Rankings are fixed before ablation training and are not "
                "recomputed after feature removal."
            ),
            "domain_importance": (
                "Loaded from the original CatBoost domain classifier. "
                "The classifier was trained on the domain-training split with "
                "validation used for early stopping; test data is not used "
                "to construct this ranking."
            ),
            "ks_statistic": (
                "Computed from the exact reconstructed balanced domain-training "
                "split only."
            ),
            "variance_distortion": (
                "Computed from the exact reconstructed balanced domain-training "
                "split only."
            ),
        },
        "sampling_policy": (
            "Exact original balanced samples are reconstructed from "
            "results/domain_classifier/sampling_provenance.csv.gz."
        ),
        "model": {
            "class": "CatBoostClassifier",
            "parameters": inputs.model_parameters,
            "early_stopping_rounds": inputs.early_stopping_rounds,
        },
        "features": {
            "total": len(inputs.schema.features),
            "names": list(inputs.schema.features),
            "categorical": list(inputs.schema.categorical_features),
            "numeric": list(inputs.schema.numeric_features),
        },
        "input_files": {
            "domain_metrics": str(DOMAIN_METRICS_PATH.relative_to(ROOT)),
            "domain_feature_importance": str(
                DOMAIN_IMPORTANCE_PATH.relative_to(ROOT)
            ),
            "sampling_provenance": str(
                SAMPLING_PROVENANCE_PATH.relative_to(ROOT)
            ),
        },
        "runs": int(len(results)),
        "outputs": {
            "ablation_results": str(ABLATION_RESULTS_PATH.relative_to(ROOT)),
            "removed_features": str(REMOVED_FEATURES_PATH.relative_to(ROOT)),
            "training_ranking_metrics": str(
                TRAINING_RANKING_METRICS_PATH.relative_to(ROOT)
            ),
            "auc_figure": str(AUC_FIGURE_PATH.relative_to(ROOT)),
            "accuracy_figure": str(ACCURACY_FIGURE_PATH.relative_to(ROOT)),
        },
    }

    with METADATA_PATH.open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)



def print_results_summary(results: pd.DataFrame) -> None:
    print_header("DOMAIN FEATURE ABLATION RESULTS")

    summary = results[
        [
            "strategy",
            "remove_n",
            "remaining_features",
            "best_iteration",
            "validation_roc_auc",
            "test_roc_auc",
            "test_accuracy",
        ]
    ].copy()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    baseline_auc = float(
        results.loc[
            results["strategy"].eq("baseline"),
            "test_roc_auc",
        ].iloc[0]
    )

    nonbaseline = results.loc[
        ~results["strategy"].eq("baseline")
    ].copy()

    nonbaseline["auc_drop_from_baseline"] = (
        baseline_auc - nonbaseline["test_roc_auc"]
    )

    print_header("LARGEST REDUCTIONS IN DOMAIN ROC-AUC")

    print(
        nonbaseline[
            [
                "strategy",
                "remove_n",
                "remaining_features",
                "test_roc_auc",
                "auc_drop_from_baseline",
                "removed_features",
            ]
        ]
        .sort_values(
            "auc_drop_from_baseline",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_output_directories()

    inputs = load_experiment_inputs()

    results, removed_features = run_ablation_experiments(inputs)

    results.to_csv(
        ABLATION_RESULTS_PATH,
        index=False,
    )

    removed_features.to_csv(
        REMOVED_FEATURES_PATH,
        index=False,
    )

    inputs.ranking_metrics.to_csv(
        TRAINING_RANKING_METRICS_PATH,
        index=False,
    )

    create_figures(results)
    save_metadata(inputs, results)
    print_results_summary(results)

    print_header("DOMAIN FEATURE ABLATION COMPLETE")

    print(ABLATION_RESULTS_PATH.relative_to(ROOT))
    print(REMOVED_FEATURES_PATH.relative_to(ROOT))
    print(TRAINING_RANKING_METRICS_PATH.relative_to(ROOT))
    print(METADATA_PATH.relative_to(ROOT))

    print(
        "\nInterpretation: lower domain ROC-AUC after removal means the "
        "remaining feature set contains less information about whether a row "
        "originated from the source or Kaggle-generated distribution."
    )


if __name__ == "__main__":
    main()
