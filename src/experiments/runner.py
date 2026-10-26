from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from src import data
from src.data import DatasetSchema, Domain, FeatureSet, Split
from src.experiments.config import ExperimentConfig, load_config
from src.experiments.metrics import evaluate_binary
from src.experiments.results import (
    create_run_directory,
    write_evaluations,
    write_json,
    write_predictions,
    write_training_sample,
)
from src.experiments.sampling import Partition, partition_summary, sampling_metadata, select_training
from src.models.base import ModelAdapter
from src.paths import PROJECT_ROOT
from src.domain_labels import display_domain_name

LOGGER = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    summary = partition_summary(domain, split, frame[identifiers], frame[schema.target])
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


def environment_metadata(model: str = "catboost") -> dict[str, Any]:
    packages = {}
    model_packages = {
        "tabpfn": ("tabpfn", "torch"),
        "tabicl": ("tabicl", "torch", "huggingface-hub", "einops", "psutil"),
        "ft_transformer": ("rtdl-revisiting-models", "torch"),
        "catboost": ("catboost",),
    }[model]
    for package in ("numpy", "pandas", "scikit-learn", "scipy", *model_packages):
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


def selection_policy(model: str) -> str:
    if model in ("tabpfn", "tabicl"):
        return (
            "One pretrained predictor prepared with training context only; validation unused. "
            "No early stopping, validation-based selection, calibration or target-domain adaptation. "
            "Fixed threshold; no test-based selection. Substantial computation occurs at prediction."
        )
    if model == "ft_transformer":
        return (
            "Task-trained FT-Transformer weights use training-domain train only; complete same-domain "
            "validation labels select the highest-ROC-AUC checkpoint and control patience. Fixed "
            "threshold; no target-domain validation, calibration, adaptation, or test-based selection."
        )
    return (
        "One fit on training-domain train; same-domain val exclusively for early stopping. "
        "Fixed threshold; no target-domain validation, calibration, or test-based selection."
    )


def create_adapter(config: ExperimentConfig, schema: DatasetSchema) -> ModelAdapter:
    if config.model == "ft_transformer":
        from src.models.ft_transformer import FTTransformerAdapter

        adapter = FTTransformerAdapter(
            schema.features, schema.categorical_features, schema.target,
            seed=config.seed,
            model_params=config.model_params,
            prediction_batch_size=config.prediction_batch_size or 256,
        )
        if not isinstance(adapter, FTTransformerAdapter):
            raise TypeError("FT-Transformer factory must return an FTTransformerAdapter.")
        return adapter
    if config.model == "tabicl":
        from src.models.tabicl import TabICLAdapter

        return TabICLAdapter(
            schema.features, schema.categorical_features, schema.target,
            seed=config.seed, model_params=config.model_params,
        )
    if config.model == "tabpfn":
        from src.models.tabpfn import TabPFNAdapter

        return TabPFNAdapter(
            schema.features, schema.categorical_features, schema.target,
            seed=config.seed, model_params=config.model_params,
        )
    from src.models.catboost import CatBoostAdapter

    if config.early_stopping_rounds is None:
        raise ValueError("CatBoost requires early_stopping_rounds.")
    return CatBoostAdapter(
        schema.features, schema.categorical_features, schema.target,
        seed=config.seed, model_params=config.model_params,
        early_stopping_rounds=config.early_stopping_rounds,
    )


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
        "selection_policy": selection_policy(config.model),
        "prediction_batch_size": config.prediction_batch_size,
    }
    try:
        write_json(output / "metadata.json", metadata)
        write_json(output / "config.json", config.to_dict())
        random.seed(config.seed)
        np.random.seed(config.seed)
        metadata["environment"] = environment_metadata(config.model)
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
        selected = select_training(train, config.train_sample_size, config.sampling_seed)
        metadata["training_partition_full"] = train.summary
        metadata["training_context"] = selected.summary
        metadata["sampling"] = sampling_metadata(
            train, selected, config.train_sample_size, config.sampling_seed,
        )
        write_training_sample(output, config.train_domain, selected.provenance)
        write_json(output / "metadata.json", metadata)

        adapter = create_adapter(config, schema)
        if config.model == "tabicl":
            from src.models.tabicl import TabICLAdapter

            if not isinstance(adapter, TabICLAdapter):
                raise TypeError("TabICL factory must return a TabICLAdapter.")
            adapter.set_diagnostic_path(output / "native_diagnostics.log")
            print(f"TabICL run: {config.experiment_name}", flush=True)
            print(f"Native diagnostics: {output / 'native_diagnostics.log'}", flush=True)
            print(f"Preparing {len(selected.X):,} training context rows...", flush=True)
        metadata["model_artifact"] = adapter.artifact_name
        adapter.reset_peak_memory()
        adapter.synchronize()
        fit_started = time.perf_counter()
        adapter.fit(selected.X, selected.y, val.X, val.y)
        adapter.synchronize()
        metadata["fit_duration_seconds"] = time.perf_counter() - fit_started
        if config.model == "tabicl":
            print(f"Context prepared in {metadata['fit_duration_seconds']:.2f}s; saving fitted state...", flush=True)
        adapter.save(output / adapter.artifact_name)
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
            if config.model == "tabicl":
                print(f"Predicting {display_domain_name(domain)}: {len(test.X):,} rows...", flush=True)
            adapter.synchronize()
            predict_started = time.perf_counter()
            probability = (
                adapter.predict_proba(test.X) if config.prediction_batch_size is None
                else adapter.predict_proba_batched(test.X, config.prediction_batch_size)
            )
            adapter.synchronize()
            metadata["prediction_duration_seconds"][domain] = time.perf_counter() - predict_started
            if config.model == "ft_transformer":
                metadata.setdefault("unseen_category_counts", {})[domain] = (
                    adapter.unseen_category_counts(test.X)
                )
            if config.model == "tabicl":
                print(f"Prediction completed in {metadata['prediction_duration_seconds'][domain]:.2f}s; writing results...", flush=True)
            evaluation = evaluate_binary(test.y, probability, config.threshold, config.calibration_bins)
            for warning in evaluation["warnings"]:
                LOGGER.warning("%s/test: %s", display_domain_name(domain, include_id=True), warning)
            predictions = test.provenance.copy()
            predictions.insert(0, "split", "test")
            predictions.insert(0, "domain", domain)
            predictions["y_true"] = test.y.to_numpy()
            predictions["y_probability"] = probability
            predictions["y_pred"] = (probability >= config.threshold).astype(np.int64)
            write_predictions(output, domain, predictions)
            evaluations[domain] = evaluation
            write_evaluations(output, evaluations)
            metadata["resources"] = adapter.resource_metadata()
            if config.model in ("tabicl", "ft_transformer"):
                metadata["model"] = adapter.metadata()
            write_json(output / "metadata.json", metadata)

        domain_width = max(
            len(display_domain_name(domain, include_id=True)) for domain in evaluations
        )
        print(f"{'domain':<{domain_width}} {'ROC-AUC':>10} {'avg prec.':>10} {'F1':>10} {'Brier':>10}")
        for domain, evaluation in evaluations.items():
            scores = [evaluation[key] for key in ("roc_auc", "average_precision", "f1", "brier_score")]
            formatted = [f"{score:.6f}" if score is not None else "null" for score in scores]
            print(
                f"{display_domain_name(domain, include_id=True):<{domain_width}} "
                + " ".join(f"{value:>10}" for value in formatted)
            )
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
        if config.model in ("tabicl", "ft_transformer") and "adapter" in locals():
            try:
                metadata["resources"] = adapter.resource_metadata()
            except Exception as resource_error:
                LOGGER.exception(
                    "Could not collect %s failure resources; preserving original error.",
                    config.model,
                )
                exc.add_note(f"Could not collect failure resources: {resource_error}")
        if config.model in ("tabpfn", "tabicl") and "out of memory" in str(exc).lower():
            context = {
                "train_sample_size": config.train_sample_size,
                "prediction_batch_size": config.prediction_batch_size,
                "n_estimators": config.model_params.get("n_estimators", 8 if config.model == "tabicl" else 4),
                "device": config.model_params.get("device", "cuda:0"),
                **({"ensemble_batch_size": config.model_params.get("batch_size", 1)}
                   if config.model == "tabicl" else {}),
                "recovery": (
                    "No project-level retry/fallback; native scheduling/recovery may already have occurred. "
                    "Inspect native_diagnostics.log and review any changed profile in a NEW output directory. "
                    "Do not reduce the n2000 context while claiming matched-budget feasibility."
                    if config.model == "tabicl" else
                    "Use a NEW output directory and explicitly configure a smaller context/batch/ensemble; no automatic retry or fallback."
                ),
            }
            metadata["error"]["out_of_memory_context"] = context
            LOGGER.error("%s out of memory: %s", config.model, context)
            exc.add_note(f"{config.model} configured memory context: {context}")
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
