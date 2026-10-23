from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import data
from src.data import DatasetSchema, Domain, FeatureSet, Split
from src.experiments.config import ExperimentConfig, load_config
from src.experiments.metrics import binary_labels, evaluate_binary
from src.experiments.results import (
    create_run_directory,
    write_evaluations,
    write_json,
    write_predictions,
)
from src.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Partition:
    X: pd.DataFrame
    y: pd.Series
    provenance: pd.DataFrame
    identities: frozenset[tuple[str, int]]
    summary: dict[str, Any]


def load_partition(
    domain: Domain, split: Split, feature_set: FeatureSet, schema: DatasetSchema,
) -> Partition:
    frame = data.load_split(
        domain, split, feature_set=feature_set,
        include_row_index=True, include_seqn=domain == "nhanes",
    )
    identifiers = ["row_index", *(["seqn"] if domain == "nhanes" else [])]
    expected = (*identifiers, *schema.features, schema.target)
    if tuple(frame.columns) != expected:
        raise ValueError(f"{domain}/{split}: columns/order do not match the requested schema.")
    for column in identifiers:
        values = frame[column].to_numpy()
        if values.dtype.kind not in "iu" or frame[column].duplicated().any():
            raise ValueError(f"{domain}/{split}: {column} must contain unique integer identities.")
    identity_column = "seqn" if domain == "nhanes" else "row_index"
    identities = [int(value) for value in frame[identity_column]]
    labels = binary_labels(frame[schema.target])
    identity_lines = [f"{domain}\t{identity_column}\t{value}\n" for value in identities]
    summary = {
        "domain": domain, "split": split, "rows": len(frame),
        "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "identity_column": identity_column,
        "identity_sha256": hashlib.sha256("".join(sorted(identity_lines)).encode()).hexdigest(),
        "ordered_identity_sha256": hashlib.sha256("".join(identity_lines).encode()).hexdigest(),
    }
    return Partition(
        X=frame.loc[:, list(schema.features)].copy(),
        y=frame[schema.target].copy(),
        provenance=frame[identifiers].copy(),
        identities=frozenset((domain, value) for value in identities),
        summary=summary,
    )


def require_disjoint(first: Partition, second: Partition) -> None:
    if first.identities & second.identities:
        raise ValueError(
            f"Identity overlap between {first.summary['domain']}/{first.summary['split']} "
            f"and {second.summary['domain']}/{second.summary['split']}."
        )


def prepared_fingerprints(config: ExperimentConfig) -> dict[str, str]:
    paths = {data.METADATA_PATH}
    if config.feature_set != "full_24":
        paths.add(data.NHANES_METADATA_PATH)
    for domain in (config.train_domain, *config.test_domains):
        paths.add(data.SPLIT_ASSIGNMENT_PATHS[domain])
        paths.add(
            data.NHANES_COMMON_PATHS[config.feature_set]
            if domain == "nhanes" else data.COMMON_PATHS[domain]
        )
    fingerprints = {}
    for path in sorted({path.resolve() for path in paths}):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        label = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        fingerprints[label] = digest.hexdigest()
    return fingerprints


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "catboost"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
            LOGGER.warning("Package version metadata unavailable: %s", package)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def run(config: ExperimentConfig) -> Path:
    started = time.perf_counter()
    output = create_run_directory(config)
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "configuration": config.to_dict(),
        "seed": config.seed,
        "splits": {},
        "prediction_duration_seconds": {},
        "reproducibility": (
            "Fixed splits, seed, identity and prepared-file fingerprints are recorded. "
            "Exact reproducibility across different hardware or package versions is not guaranteed."
        ),
        "selection_policy": (
            "One fit on training-domain train; same-domain val exclusively for early stopping. "
            "Fixed threshold; no target-domain validation, calibration, or test-based selection."
        ),
    }
    try:
        write_json(output / "metadata.json", metadata)
        write_json(output / "config.json", config.to_dict())
        random.seed(config.seed)
        np.random.seed(config.seed)
        metadata["environment"] = environment_metadata()
        schema = data.load_feature_schema(config.feature_set)
        if (
            len(set(schema.features)) != len(schema.features)
            or set(schema.categorical_features) & set(schema.numeric_features)
            or set(schema.features) != set(schema.categorical_features) | set(schema.numeric_features)
        ):
            raise ValueError("The requested feature schema has inconsistent feature types.")
        metadata["schema"] = {
            "target": schema.target,
            "feature_order": list(schema.features),
            "categorical_features": list(schema.categorical_features),
            "numeric_features": list(schema.numeric_features),
        }
        metadata["prepared_file_sha256"] = prepared_fingerprints(config)
        train = load_partition(config.train_domain, "train", config.feature_set, schema)
        val = load_partition(config.validation_domain, "val", config.feature_set, schema)
        require_disjoint(train, val)
        metadata["splits"].update({
            f"{config.train_domain}/train": train.summary,
            f"{config.validation_domain}/val": val.summary,
        })
        write_json(output / "metadata.json", metadata)

        from src.models.catboost import CatBoostAdapter

        adapter = CatBoostAdapter(
            schema.features, schema.categorical_features, schema.target,
            seed=config.seed, model_params=config.model_params,
            early_stopping_rounds=config.early_stopping_rounds,
        )
        fit_started = time.perf_counter()
        adapter.fit(train.X, train.y, val.X, val.y)
        metadata["fit_duration_seconds"] = time.perf_counter() - fit_started
        adapter.save(output / "model.cbm")
        metadata["model"] = adapter.metadata()
        metadata["preprocessing"] = metadata["model"]["preprocessing"]
        metadata["compute_settings"] = metadata["model"]["compute_settings"]
        write_json(output / "metadata.json", metadata)

        evaluations: dict[str, dict[str, Any]] = {}
        for domain in config.test_domains:
            test = load_partition(domain, "test", config.feature_set, schema)
            require_disjoint(train, test)
            require_disjoint(val, test)
            metadata["splits"][f"{domain}/test"] = test.summary
            predict_started = time.perf_counter()
            probability = adapter.predict_proba(test.X)
            metadata["prediction_duration_seconds"][domain] = time.perf_counter() - predict_started
            evaluation = evaluate_binary(test.y, probability, config.threshold, config.calibration_bins)
            for warning in evaluation["warnings"]:
                LOGGER.warning("%s/test: %s", domain, warning)
            predictions = test.provenance.copy()
            predictions.insert(0, "split", "test")
            predictions.insert(0, "domain", domain)
            predictions["y_true"] = test.y.to_numpy()
            predictions["y_probability"] = probability
            predictions["y_pred"] = (probability >= config.threshold).astype(np.int64)
            write_predictions(output, domain, predictions)
            evaluations[domain] = evaluation
            write_evaluations(output, evaluations)
            write_json(output / "metadata.json", metadata)

        print(f"{'domain':<10} {'ROC-AUC':>10} {'avg prec.':>10} {'F1':>10} {'Brier':>10}")
        for domain, evaluation in evaluations.items():
            scores = [evaluation[key] for key in ("roc_auc", "average_precision", "f1", "brier_score")]
            formatted = [f"{score:.6f}" if score is not None else "null" for score in scores]
            print(f"{domain:<10} " + " ".join(f"{value:>10}" for value in formatted))
        print(f"Output: {output}")
        metadata.update(
            status="completed", ended_at_utc=utc_now(),
            total_duration_seconds=time.perf_counter() - started,
        )
        write_json(output / "metadata.json", metadata)
    except BaseException as exc:
        metadata.update(
            status="failed", ended_at_utc=utc_now(),
            total_duration_seconds=time.perf_counter() - started,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        try:
            write_json(output / "metadata.json", metadata)
        except BaseException:
            LOGGER.exception("Could not record failed run status in %s; preserving original error.", output)
        raise

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one predictive model and evaluate fixed cross-domain test splits.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment JSON configuration.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
