from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    raise ImportError(
        "CatBoost is required for the domain-classifier experiment.\n"
        "Install it with:\n"
        "  pip install catboost"
    ) from exc

from src.data import load_schema, load_split
from src.paths import DOMAIN_CLASSIFIER_DIR as OUTPUT_DIR, PROJECT_ROOT as ROOT


# =============================================================================
# Configuration
# =============================================================================

METRICS_PATH = OUTPUT_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
SAMPLING_PROVENANCE_PATH = OUTPUT_DIR / "sampling_provenance.csv.gz"
MODEL_PATH = OUTPUT_DIR / "domain_classifier.cbm"


SEED = 42

DOMAIN_SOURCE = 0
DOMAIN_KAGGLE = 1

Split = Literal["train", "val", "test"]


MODEL_PARAMS = {
    "iterations": 1000,
    "depth": 7,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "verbose": 100,
    "allow_writing_files": False,
    "thread_count": -1,
}


# =============================================================================
# Data container
# =============================================================================


@dataclass
class DomainDataset:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


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


# =============================================================================
# Domain dataset construction
# =============================================================================


def normalize_catboost_inputs(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize categorical representation for CatBoost.

    This does not learn anything from the data and does not perform encoding.
    It simply ensures that categorical values are represented consistently as
    Python strings.
    """
    schema = load_schema()

    X = X.copy()

    for feature in schema.categorical_features:
        X[feature] = X[feature].astype(str)

    return X


def create_balanced_domain_split(
    split: Split,
    seed: int,
) -> DomainDataset:
    """
    Build a balanced source-vs-Kaggle classification dataset.

    All source examples in the prepared split are used. Kaggle examples are
    deterministically downsampled to the same number of rows.

    Domain labels:
        0 -> source
        1 -> Kaggle-generated

    diagnosed_diabetes is intentionally excluded from X.

    Kaggle id is retained only in metadata and is never passed to the model.
    """

    schema = load_schema()

    source = load_split(
        "source",
        split,
        include_row_index=True,
    )

    kaggle = load_split(
        "kaggle",
        split,
        include_row_index=True,
        include_id=True,
    )

    if len(kaggle) < len(source):
        raise ValueError(
            f"Kaggle {split} split contains fewer rows than "
            f"the source {split} split."
        )

    # -------------------------------------------------------------------------
    # Deterministically select exactly as many Kaggle rows as source rows.
    # -------------------------------------------------------------------------

    kaggle = kaggle.sample(
        n=len(source),
        replace=False,
        random_state=seed,
    ).copy()

    # -------------------------------------------------------------------------
    # Extract predictors.
    #
    # By selecting schema.features explicitly, neither:
    #
    #   diagnosed_diabetes
    #   id
    #   row_index
    #
    # can accidentally enter the classifier.
    # -------------------------------------------------------------------------

    feature_columns = list(schema.features)

    X_source = source[
        feature_columns
    ].copy()

    X_kaggle = kaggle[
        feature_columns
    ].copy()

    X_source = normalize_catboost_inputs(X_source)
    X_kaggle = normalize_catboost_inputs(X_kaggle)

    # -------------------------------------------------------------------------
    # Artificial domain targets.
    # -------------------------------------------------------------------------

    y_source = pd.Series(
        np.full(
            len(X_source),
            DOMAIN_SOURCE,
            dtype=np.int64,
        ),
        name="domain",
    )

    y_kaggle = pd.Series(
        np.full(
            len(X_kaggle),
            DOMAIN_KAGGLE,
            dtype=np.int64,
        ),
        name="domain",
    )

    # -------------------------------------------------------------------------
    # Provenance metadata.
    #
    # This allows predictions to be traced back to the canonical prepared
    # datasets without exposing provenance columns to the classifier.
    # -------------------------------------------------------------------------

    source_metadata = pd.DataFrame(
        {
            "split": split,
            "original_domain": "source",
            "original_row_index":
                source["row_index"].to_numpy(),
            "kaggle_id": pd.array(
                [pd.NA] * len(source),
                dtype="Int64",
            ),
        }
    )

    kaggle_metadata = pd.DataFrame(
        {
            "split": split,
            "original_domain": "kaggle",
            "original_row_index":
                kaggle["row_index"].to_numpy(),
            "kaggle_id": pd.array(
                kaggle["id"].to_numpy(),
                dtype="Int64",
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Combine domains.
    # -------------------------------------------------------------------------

    X = pd.concat(
        [
            X_source,
            X_kaggle,
        ],
        ignore_index=True,
    )

    y = pd.concat(
        [
            y_source,
            y_kaggle,
        ],
        ignore_index=True,
    )

    metadata = pd.concat(
        [
            source_metadata,
            kaggle_metadata,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Shuffle the combined domain dataset deterministically.
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(len(X))

    X = (
        X.iloc[permutation]
        .reset_index(drop=True)
    )

    y = (
        y.iloc[permutation]
        .reset_index(drop=True)
    )

    metadata = (
        metadata.iloc[permutation]
        .reset_index(drop=True)
    )

    return DomainDataset(
        X=X,
        y=y,
        metadata=metadata,
    )


def build_domain_datasets() -> dict[Split, DomainDataset]:
    """
    Build train, validation, and test domain datasets.

    Separate sampling seeds are used for clarity even though the prepared
    source/Kaggle splits are already mutually exclusive.
    """

    return {
        "train": create_balanced_domain_split(
            "train",
            SEED,
        ),
        "val": create_balanced_domain_split(
            "val",
            SEED + 1,
        ),
        "test": create_balanced_domain_split(
            "test",
            SEED + 2,
        ),
    }


# =============================================================================
# CatBoost pools
# =============================================================================


def create_pool(
    dataset: DomainDataset,
) -> Pool:
    schema = load_schema()

    return Pool(
        data=dataset.X,
        label=dataset.y,
        cat_features=list(
            schema.categorical_features
        ),
    )


# =============================================================================
# Model training
# =============================================================================


def train_model(
    train: DomainDataset,
    validation: DomainDataset,
) -> CatBoostClassifier:
    print_header("TRAINING DOMAIN CLASSIFIER")

    train_pool = create_pool(train)
    validation_pool = create_pool(validation)

    model = CatBoostClassifier(
        **MODEL_PARAMS,
    )

    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=75,
    )

    best_iteration = model.get_best_iteration()

    if best_iteration >= 0:
        print(
            f"\nBest iteration: "
            f"{best_iteration + 1}"
        )

    return model


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(
    model: CatBoostClassifier,
    dataset: DomainDataset,
) -> tuple[dict, np.ndarray, np.ndarray]:
    pool = create_pool(dataset)

    probabilities = (
        model.predict_proba(pool)[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    metrics = {
        "roc_auc": float(
            roc_auc_score(
                dataset.y,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                dataset.y,
                predictions,
            )
        ),
        "log_loss": float(
            log_loss(
                dataset.y,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                dataset.y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                dataset.y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                dataset.y,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                dataset.y,
                predictions,
            )
            .astype(int)
            .tolist()
        ),
    }

    return (
        metrics,
        probabilities,
        predictions,
    )


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print_header(title)

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Log loss:  "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1:        "
        f"{metrics['f1']:.6f}"
    )

    matrix = metrics["confusion_matrix"]

    print("\nConfusion matrix:")
    print("                 Pred source   Pred Kaggle")
    print(
        f"True source      "
        f"{matrix[0][0]:>11,}   "
        f"{matrix[0][1]:>11,}"
    )
    print(
        f"True Kaggle      "
        f"{matrix[1][0]:>11,}   "
        f"{matrix[1][1]:>11,}"
    )


# =============================================================================
# Feature importance
# =============================================================================


def get_feature_importance(
    model: CatBoostClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    importance = model.get_feature_importance(
        type="PredictionValuesChange",
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


def print_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 15,
) -> None:
    print_header(
        "TOP DOMAIN-DISCRIMINATIVE FEATURES"
    )

    print(
        importance
        .head(top_n)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# Experiment metadata
# =============================================================================


def summarize_domain_dataset(
    dataset: DomainDataset,
) -> dict:
    counts = (
        dataset.y
        .value_counts()
        .sort_index()
    )

    return {
        "rows": int(len(dataset.y)),
        "source_rows": int(
            counts.get(
                DOMAIN_SOURCE,
                0,
            )
        ),
        "kaggle_rows": int(
            counts.get(
                DOMAIN_KAGGLE,
                0,
            )
        ),
        "features": int(
            dataset.X.shape[1]
        ),
    }


def print_dataset_summary(
    datasets: dict[Split, DomainDataset],
) -> None:
    print_header(
        "BALANCED DOMAIN DATASETS"
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        summary = summarize_domain_dataset(
            datasets[split]
        )

        print(f"\n{split.capitalize()}:")

        print(
            f"  Source: "
            f"{summary['source_rows']:,}"
        )

        print(
            f"  Kaggle: "
            f"{summary['kaggle_rows']:,}"
        )

        print(
            f"  Total:  "
            f"{summary['rows']:,}"
        )


# =============================================================================
# Saving outputs
# =============================================================================


def save_predictions(
    dataset: DomainDataset,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    output = dataset.metadata.copy()

    output["domain_true"] = (
        dataset.y.to_numpy()
    )

    output["domain_probability"] = (
        probabilities
    )

    output["domain_predicted"] = (
        predictions
    )

    output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )


def save_sampling_provenance(
    datasets: dict[Split, DomainDataset],
) -> None:
    """
    Save the exact rows selected for each balanced domain split.

    This is metadata only; predictor values are intentionally not duplicated.
    """

    frames = []

    for split in (
        "train",
        "val",
        "test",
    ):
        dataset = datasets[split]

        frame = dataset.metadata.copy()

        frame["domain_label"] = (
            dataset.y.to_numpy()
        )

        frames.append(frame)

    provenance = pd.concat(
        frames,
        ignore_index=True,
    )

    provenance.to_csv(
        SAMPLING_PROVENANCE_PATH,
        index=False,
        compression="gzip",
    )


def save_metrics(
    model: CatBoostClassifier,
    datasets: dict[Split, DomainDataset],
    validation_metrics: dict,
    test_metrics: dict,
) -> None:
    schema = load_schema()

    best_iteration = (
        model.get_best_iteration()
    )

    if best_iteration >= 0:
        best_iteration += 1

    output = {
        "experiment": "domain_classifier",
        "description": (
            "Balanced binary classification of source "
            "versus Kaggle-generated samples using the "
            "24 shared predictors."
        ),
        "seed": SEED,
        "domain_labels": {
            "source": DOMAIN_SOURCE,
            "kaggle": DOMAIN_KAGGLE,
        },
        "features": {
            "count": len(
                schema.features
            ),
            "names": list(
                schema.features
            ),
            "categorical": list(
                schema.categorical_features
            ),
            "numeric": list(
                schema.numeric_features
            ),
        },
        "explicitly_excluded_inputs": [
            schema.target,
            "id",
            "row_index",
        ],
        "splits": {
            split: summarize_domain_dataset(
                dataset
            )
            for split, dataset
            in datasets.items()
        },
        "model": {
            "class": "CatBoostClassifier",
            "parameters": MODEL_PARAMS,
            "early_stopping_rounds": 75,
            "best_iteration": (
                best_iteration
            ),
        },
        "validation_metrics":
            validation_metrics,
        "test_metrics":
            test_metrics,
    }

    with METRICS_PATH.open("w") as file:
        json.dump(
            output,
            file,
            indent=2,
            sort_keys=True,
        )


# =============================================================================
# Main experiment
# =============================================================================


def main() -> None:
    ensure_output_directory()

    # -------------------------------------------------------------------------
    # Construct the balanced domain-classification datasets.
    # -------------------------------------------------------------------------

    print_header(
        "BUILDING DOMAIN CLASSIFICATION DATA"
    )

    datasets = build_domain_datasets()

    print_dataset_summary(datasets)

    # -------------------------------------------------------------------------
    # Train CatBoost.
    # -------------------------------------------------------------------------

    model = train_model(
        datasets["train"],
        datasets["val"],
    )

    # -------------------------------------------------------------------------
    # Evaluate validation and untouched domain-test sets.
    # -------------------------------------------------------------------------

    (
        validation_metrics,
        _,
        _,
    ) = evaluate_model(
        model,
        datasets["val"],
    )

    (
        test_metrics,
        test_probabilities,
        test_predictions,
    ) = evaluate_model(
        model,
        datasets["test"],
    )

    print_metrics(
        "VALIDATION RESULTS",
        validation_metrics,
    )

    print_metrics(
        "TEST RESULTS",
        test_metrics,
    )

    # -------------------------------------------------------------------------
    # Global feature importance.
    # -------------------------------------------------------------------------

    feature_importance = (
        get_feature_importance(
            model,
            list(
                datasets[
                    "train"
                ].X.columns
            ),
        )
    )

    print_feature_importance(
        feature_importance
    )

    # -------------------------------------------------------------------------
    # Save all experiment artifacts.
    # -------------------------------------------------------------------------

    print_header(
        "SAVING EXPERIMENT OUTPUTS"
    )

    model.save_model(
        MODEL_PATH,
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    save_predictions(
        datasets["test"],
        test_probabilities,
        test_predictions,
    )

    save_sampling_provenance(
        datasets
    )

    save_metrics(
        model,
        datasets,
        validation_metrics,
        test_metrics,
    )

    print(
        MODEL_PATH.relative_to(ROOT)
    )

    print(
        METRICS_PATH.relative_to(ROOT)
    )

    print(
        FEATURE_IMPORTANCE_PATH.relative_to(
            ROOT
        )
    )

    print(
        TEST_PREDICTIONS_PATH.relative_to(
            ROOT
        )
    )

    print(
        SAMPLING_PROVENANCE_PATH.relative_to(
            ROOT
        )
    )

    print_header(
        "DOMAIN CLASSIFIER COMPLETE"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  0.50 -> domains are effectively "
        "indistinguishable"
    )

    print(
        "  higher -> shared features contain "
        "increasingly strong domain information"
    )


if __name__ == "__main__":
    main()