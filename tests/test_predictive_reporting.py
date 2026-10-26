from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from src.experiments.metrics import evaluate_binary
from src.experiments.sampling import identity_fingerprints
from src.reporting import compare_predictive, figures, tables, validation
from src.reporting.validation import ReportValidationError

PROJECT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = json.loads(
    (PROJECT / "configs/reports/predictive_n2000_seed42.json").read_text(encoding="utf-8")
)
TABICL_MANIFEST = json.loads(
    (PROJECT / "configs/reports/predictive_n2000_seed42_with_tabicl.json").read_text(encoding="utf-8")
)
FEATURES = BASE_MANIFEST["comparison"]["feature_order"]
DOMAINS = BASE_MANIFEST["comparison"]["evaluation_domains"]
TRAIN_IDS = {
    "source": list(range(2000)),
    "kaggle": list(range(10000, 12000)),
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _summary(domain: str, split: str, identities: list[int], labels: list[int]) -> dict:
    identity_column = "seqn" if domain == "nhanes" else "row_index"
    return {
        "domain": domain,
        "split": split,
        "rows": len(labels),
        "class_counts": {"0": labels.count(0), "1": labels.count(1)},
        "identity_column": identity_column,
        **identity_fingerprints(domain, identity_column, identities),
    }


def _write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(domain: str, model: str, reverse: bool = False) -> list[dict]:
    identity_column = "seqn" if domain == "nhanes" else "row_index"
    offset = {"source": 10, "kaggle": 20, "nhanes": 30}[domain]
    indices = list(range(4))
    if reverse:
        indices.reverse()
    result = []
    for index in indices:
        row = {
            "domain": domain,
            "split": "test",
            "row_index": offset + index,
            identity_column: (100 + index) if domain == "nhanes" else offset + index,
            "y_true": [0, 0, 1, 1][index],
            "y_probability": (
                [0.4, 0.6, 0.5, 0.4][index]
                if model == "catboost" else [0.2, 0.4, 0.6, 0.8][index]
            ),
            "y_pred": int(
                (
                    [0.4, 0.6, 0.5, 0.4][index]
                    if model == "catboost" else [0.2, 0.4, 0.6, 0.8][index]
                ) >= 0.5
            ),
        }
        result.append(row)
    return result


def _calibration_rows(metrics: dict, domains: list[str]) -> list[dict]:
    return [
        {"domain": domain, "split": "test", **entry}
        for domain in domains
        for entry in metrics[domain]["calibration_bins"]
    ]


def fixture(root: Path) -> tuple[dict, Path]:
    manifest = json.loads(json.dumps(BASE_MANIFEST))
    run_dirs = {}
    for spec in manifest["runs"]:
        spec["run_id"] = f"test_{spec['role']}"
        spec["artifact_dir"] = f"results/predictive/{spec['role']}"
        run_dir = root / spec["artifact_dir"]
        run_dir.mkdir(parents=True)
        run_dirs[spec["role"]] = run_dir

        model = spec["model"]
        train_domain = spec["train_domain"]
        if model == "catboost":
            model_params = {"task_type": "CPU"}
            early_stopping = 10
            compute = {"task_type": "CPU", "thread_count": 1, "seed": 42}
            batch_size = spec["prediction_batch_size"]
        else:
            model_params = {"device": "cuda:0"}
            early_stopping = None
            compute = {"device": "cuda:0", "device_name": "Synthetic GPU", "seed": 42}
            batch_size = spec["prediction_batch_size"]
        config = {
            "experiment_name": spec["run_id"],
            "model": model,
            "train_domain": train_domain,
            "validation_domain": train_domain,
            "test_domains": DOMAINS,
            "feature_set": "strict_common_16",
            "seed": 42,
            "threshold": 0.5,
            "calibration_bins": 10,
            "model_params": model_params,
            "output_dir": str(run_dir.resolve()),
            "early_stopping_rounds": early_stopping,
            "train_sample_size": 2000,
            "sampling_seed": 42,
            "prediction_batch_size": batch_size,
        }
        training_ids = TRAIN_IDS[train_domain]
        training_fingerprints = identity_fingerprints(
            train_domain, "row_index", training_ids,
        )
        counts = {"0": 1000, "1": 1000}
        sampling = {
            "available_training_rows": 4000,
            "selected_training_rows": 2000,
            "requested_budget": 2000,
            "sampling_seed": 42,
            "algorithm": "synthetic fixture stratified identity sampling",
            "selected_class_counts": counts,
            **training_fingerprints,
        }
        training_context = {
            "domain": train_domain,
            "split": "train",
            "rows": 2000,
            "class_counts": counts,
            "identity_column": "row_index",
            **training_fingerprints,
        }
        full_ids = list(range(4000, 8000)) if train_domain == "source" else list(range(14000, 18000))
        full_summary = _summary(train_domain, "train", full_ids, [0, 1] * 2000)
        val_summary = _summary(train_domain, "val", [8000, 8001], [0, 1])
        sample = [
            {"domain": train_domain, "split": "train", "sample_order": i, "row_index": identity}
            for i, identity in enumerate(training_ids)
        ]
        _write_gzip_csv(
            run_dir / "training_sample.csv.gz",
            ["domain", "split", "sample_order", "row_index"],
            sample,
        )

        saved_metrics = {}
        predictions_by_domain = {}
        split_summaries = {}
        for domain in DOMAINS:
            reversed_order = model == "tabpfn" and domain == "source"
            predictions = _prediction_rows(domain, model, reversed_order)
            predictions_by_domain[domain] = predictions
            pred_fields = list(predictions[0])
            _write_gzip_csv(run_dir / f"predictions_{domain}_test.csv.gz", pred_fields, predictions)
            labels = [row["y_true"] for row in predictions]
            probabilities = [row["y_probability"] for row in predictions]
            saved_metrics[domain] = evaluate_binary(labels, probabilities, 0.5, 10)
            id_column = "seqn" if domain == "nhanes" else "row_index"
            split_summaries[f"{domain}/test"] = _summary(
                domain,
                "test",
                [row[id_column] for row in predictions],
                labels,
            )

        _write_json(run_dir / "metrics.json", saved_metrics)
        cal = _calibration_rows(saved_metrics, DOMAINS)
        _write_gzip_or_plain_calibration(run_dir / "calibration_bins.csv", cal)
        prepared = {"data/prepared.csv": "a" * 64}
        metadata = {
            "status": "completed",
            "configuration": config,
            "seed": 42,
            "prediction_batch_size": batch_size,
            "schema": {
                "target": "diagnosed_diabetes",
                "feature_order": FEATURES,
                "categorical_features": ["ethnicity", "gender", "smoking_status"],
                "numeric_features": [
                    feature for feature in FEATURES
                    if feature not in {"ethnicity", "gender", "smoking_status"}
                ],
            },
            "prepared_file_sha256": prepared,
            "training_partition_full": full_summary,
            "training_context": training_context,
            "sampling": sampling,
            "splits": {
                f"{train_domain}/train": full_summary,
                f"{train_domain}/val": val_summary,
                **split_summaries,
            },
            "prediction_duration_seconds": {domain: 1.0 for domain in DOMAINS},
            "compute_settings": compute,
            "resources": {},
            "fit_duration_seconds": 2.0,
            "total_duration_seconds": 10.0,
            "environment": {"machine": "synthetic"},
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "config.json", config)

    manifest_path = root / "configs/test_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def tabicl_fixture(root: Path) -> tuple[dict, Path]:
    _, _ = fixture(root)
    manifest = json.loads(json.dumps(TABICL_MANIFEST))
    manifest_path = root / "configs/test_manifest_with_tabicl.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in manifest["runs"]:
        role = spec["role"]
        train_domain = spec["train_domain"]
        source_role = role if not role.endswith("_tabicl") else role.removesuffix("_tabicl") + "_tabpfn"
        source_dir = root / "results/predictive" / source_role
        run_dir = root / "results/predictive" / role
        if role.endswith("_tabicl"):
            shutil.copytree(source_dir, run_dir)
        spec["artifact_dir"] = f"results/predictive/{role}"
        config_path = run_dir / "config.json"
        metadata_path = run_dir / "metadata.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        config.update({
            "experiment_name": spec["run_id"],
            "model": spec["model"],
            "train_domain": train_domain,
            "validation_domain": train_domain,
            "prediction_batch_size": spec["prediction_batch_size"],
            "output_dir": str(run_dir.resolve()),
        })
        model = spec["model"]
        if model == "tabpfn":
            config["model_params"] = {
                "n_estimators": 4,
                "device": "cuda:0",
                "fit_mode": "fit_preprocessors",
                "inference_precision": "float32",
                "memory_saving_mode": True,
            }
        elif model == "tabicl":
            checkpoint_relative = (
                "checkpoints/tabicl/fixture/tabicl-classifier-v2-20260212.ckpt"
            )
            config["model_params"] = {
                "model_path": checkpoint_relative,
                "norm_methods": ["none"],
                "device": "cuda:0",
                "n_estimators": 8,
                "batch_size": 1,
                "use_amp": False,
                "use_fa3": False,
                "kv_cache": False,
                "offload_mode": "cpu",
                "n_jobs": 4,
            }
        metadata["configuration"] = config
        metadata["prediction_batch_size"] = spec["prediction_batch_size"]
        metadata["seed"] = 42
        compute = metadata["compute_settings"]
        if model == "catboost":
            model_metadata = {
                "validation_used": True,
                "categorical_features": ["ethnicity", "gender", "smoking_status"],
                "compute_settings": compute,
            }
        elif model == "tabpfn":
            compute.update({
                "forced_dtype": "torch.float32",
                "autocast_enabled": False,
                "n_estimators": 4,
            })
            model_metadata = {
                "checkpoint": {"generation": "TabPFN v3.5 (20260909)"},
                "effective_ensemble_count": 4,
                "effective_init_parameters": {"inference_precision": "torch.float32"},
                "requested_parameters": {"inference_precision": "float32", "n_estimators": 4},
                "validation_policy": "Unused; no early stopping, calibration, tuning, or validation context.",
                "validation_used": False,
                "categorical_features": ["ethnicity", "gender", "smoking_status"],
                "compute_settings": compute,
            }
        else:
            checkpoint_path = (root / config["model_params"]["model_path"]).resolve()
            compute.update({
                "device": "cuda:0",
                "device_name": "Synthetic GPU",
                "n_estimators": 8,
                "ensemble_batch_size": 1,
                "use_amp": False,
                "use_fa3": False,
                "kv_cache": False,
                "cuda_matmul_allow_tf32": False,
                "inference_config": {
                    "COL_CONFIG": {"offload": "cpu", "use_amp": False, "use_fa3": False},
                    "ROW_CONFIG": {"offload": False, "use_amp": False, "use_fa3": False},
                    "ICL_CONFIG": {"offload": False, "use_amp": False, "use_fa3": False},
                },
            })
            requested = dict(config["model_params"])
            effective = dict(requested)
            effective["model_path"] = str(checkpoint_path)
            model_metadata = {
                "checkpoint": {
                    "generation": "TabICLv2 classifier",
                    "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
                    "filename": "tabicl-classifier-v2-20260212.ckpt",
                    "sha256": "a" * 64,
                    "path": str(checkpoint_path),
                },
                "tabicl_version": "2.2.0",
                "experimental_profile": "tabicl_v2_none_fp32_uncached",
                "effective_ensemble_count": 8,
                "effective_init_parameters": effective,
                "requested_parameters": requested,
                "preprocessing": {"normalization_views": ["none"]},
                "validation_policy": "Unused; train-only context/preprocessing, no tuning or calibration.",
                "validation_used": False,
                "categorical_features": ["ethnicity", "gender", "smoking_status"],
                "native_diagnostics": {
                    "recovery_detection": "no_recovery_detected",
                    "swap_fallback_detected": False,
                    "operations": [{"operation": "predict"}],
                    "limitations": "Captured output only; not a complete scheduling trace.",
                },
                "compute_settings": compute,
            }
        metadata["compute_settings"] = compute
        metadata["model"] = model_metadata
        _write_json(config_path, config)
        _write_json(metadata_path, metadata)

    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def _write_gzip_or_plain_calibration(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain", "split", "bin", "lower_bound", "upper_bound", "count",
                "mean_predicted_probability", "observed_positive_rate", "absolute_gap",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


class PredictiveReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, self.manifest_path = fixture(self.root)
        self.patches = [
            patch.object(validation, "PROJECT_ROOT", self.root),
            patch.object(validation, "RESULTS_DIR", self.root / "results"),
            patch.object(compare_predictive, "PROJECT_ROOT", self.root),
            patch.object(compare_predictive, "RESULTS_DIR", self.root / "results"),
        ]
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)

    def validate(self):
        return validation.validate_comparison(self.manifest, self.manifest_path)

    def test_valid_matched_comparison_aligns_order_and_accepts_expected_differences(self):
        runs, result = self.validate()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(runs), 4)
        cross_checks = result["cross_run_population_checks"]
        self.assertEqual(len(cross_checks), 9)
        self.assertTrue(all(check["identities_match"] and check["labels_match"] for check in cross_checks))
        source_alignment = next(
            row for row in result["identity_alignments"]
            if row["training_domain"] == "source" and row["evaluation_domain"] == "source"
        )
        self.assertTrue(source_alignment["identities_match"])
        self.assertTrue(source_alignment["labels_match"])
        self.assertTrue(source_alignment["prediction_order_differs"])
        runtime = tables.build_runtime_rows(runs, self.manifest)
        self.assertEqual([row["prediction_batch_size"] for row in runtime], [256, 256, 256, 1024])
        self.assertIsNone(runtime[0]["pytorch_peak_allocated_bytes"])
        self.assertIsNone(runtime[0]["total_gpu_usage_bytes"])
        self.assertIn("CatBoost runs on CPU", "\n".join(self.manifest["expected_methodological_differences"]))
        report = tables.build_report(runs, self.manifest, result, "timestamp")
        self.assertIn("TabPFN fitting is reported as context preparation", report)
        self.assertIn("No confidence intervals", report)
        self.assertIn("Original Synthetic", report)
        self.assertIn("Playground S5E12", report)
        self.assertIn("NHANES 2015–2016", report)

    def test_tables_have_fixed_row_counts_signs_and_order(self):
        runs, _ = self.validate()
        metric_rows = tables.build_metric_rows(runs, self.manifest)
        difference_rows = tables.build_difference_rows(runs, self.manifest)
        self.assertEqual(len(metric_rows), 12)
        self.assertEqual(len(difference_rows), 6)
        self.assertEqual(
            [(row["train_domain_id"], row["model"], row["evaluation_domain_id"]) for row in metric_rows[:6]],
            [
                ("source", "catboost", "source"),
                ("source", "catboost", "kaggle"),
                ("source", "catboost", "nhanes"),
                ("source", "tabpfn", "source"),
                ("source", "tabpfn", "kaggle"),
                ("source", "tabpfn", "nhanes"),
            ],
        )
        for row in difference_rows:
            self.assertGreater(row["tabpfn_minus_catboost_roc_auc"], 0)
            self.assertGreater(row["tabpfn_minus_catboost_average_precision"], 0)
            self.assertGreater(row["tabpfn_minus_catboost_f1"], 0)
            self.assertLess(row["tabpfn_minus_catboost_brier_score"], 0)
        tables_first = tables.generate_tables(runs, self.manifest)
        tables_second = tables.generate_tables(runs, self.manifest)
        self.assertEqual(tables_first, tables_second)
        self.assertEqual(len(list(csv.DictReader(tables_first["metrics.csv"].splitlines()))), 12)
        self.assertEqual(len(list(csv.DictReader(tables_first["metric_differences.csv"].splitlines()))), 6)

    def test_plot_data_grouping_and_calibration_bins_reuse_validated_metrics(self):
        runs, _ = self.validate()
        metric_points = figures.performance_points(runs, self.manifest, "roc_auc")
        self.assertEqual(len(metric_points), 12)
        self.assertEqual(
            {
                (point["training_domain_id"], point["evaluation_domain_id"], point["model"])
                for point in metric_points
            },
            {
                (train, evaluation, model)
                for train in ("source", "kaggle")
                for evaluation in DOMAINS
                for model in ("catboost", "tabpfn")
            },
        )
        self.assertEqual(
            {point["model_label"] for point in metric_points},
            {"CatBoost", "TabPFN"},
        )
        calibration_rows = figures.calibration_points(runs, self.manifest)
        self.assertEqual(len(calibration_rows), 4 * len(DOMAINS) * 10)
        by_run = {run.spec["run_id"]: run for run in runs}
        for row in calibration_rows:
            expected = by_run[row["run_id"]].metrics[row["evaluation_domain_id"]]["calibration_bins"][row["bin"]]
            self.assertEqual(row["count"], expected["count"])
            self.assertEqual(row["mean_predicted_probability"], expected["mean_predicted_probability"])
            self.assertEqual(row["observed_positive_rate"], expected["observed_positive_rate"])
        self.assertTrue(any(row["count"] == 0 for row in calibration_rows))
        self.assertTrue(any(row["mean_predicted_probability"] is None for row in calibration_rows))

    def test_undefined_auc_and_average_precision_remain_missing(self):
        for spec in self.manifest["runs"]:
            run_dir = self.root / spec["artifact_dir"]
            path = run_dir / "predictions_source_test.csv.gz"
            rows = validation._read_csv(path, spec["run_id"], path.name)
            for row in rows:
                row["y_true"] = "0"
            self._write_predictions(run_dir, "source", rows)
            self._refresh_domain_results(run_dir, "source", rows)
        runs, _ = self.validate()
        metrics = tables.build_metric_rows(runs, self.manifest)
        source_rows = [
            row for row in metrics
            if row["evaluation_domain_id"] == "source"
        ]
        self.assertTrue(all(row["roc_auc"] is None for row in source_rows))
        self.assertTrue(all(row["average_precision"] is None for row in source_rows))
        differences = tables.build_difference_rows(runs, self.manifest)
        source_differences = [
            row for row in differences
            if row["evaluation_domain_id"] == "source"
        ]
        self.assertTrue(all(row["tabpfn_minus_catboost_roc_auc"] is None for row in source_differences))
        self.assertTrue(all(row["tabpfn_minus_catboost_average_precision"] is None for row in source_differences))

    def test_training_sample_identity_mismatch_rejected_between_models(self):
        tab_dir = self.root / self.manifest["runs"][1]["artifact_dir"]
        sample_rows = validation._read_csv(tab_dir / "training_sample.csv.gz", "tab", "sample")
        sample_rows[-1]["row_index"] = str(int(sample_rows[-1]["row_index"]) + 100000)
        identities = [int(row["row_index"]) for row in sample_rows]
        fingerprints = identity_fingerprints("source", "row_index", identities)
        self._write_sample(tab_dir, sample_rows)
        metadata = self._read_json(tab_dir / "metadata.json")
        metadata["training_context"].update(fingerprints)
        metadata["sampling"].update(fingerprints)
        _write_json(tab_dir / "metadata.json", metadata)
        with self.assertRaisesRegex(ReportValidationError, "training identity mismatch"):
            self.validate()

    def test_test_identity_mismatch_rejected_between_models(self):
        tab_dir = self.root / self.manifest["runs"][1]["artifact_dir"]
        prediction_path = tab_dir / "predictions_source_test.csv.gz"
        rows = validation._read_csv(prediction_path, "tab", prediction_path.name)
        rows[0]["row_index"] = "999999"
        self._write_predictions(tab_dir, "source", rows)
        metadata = self._read_json(tab_dir / "metadata.json")
        labels = [int(row["y_true"]) for row in rows]
        metadata["splits"]["source/test"] = _summary(
            "source", "test", [int(row["row_index"]) for row in rows], labels,
        )
        _write_json(tab_dir / "metadata.json", metadata)
        with self.assertRaisesRegex(ReportValidationError, "test identity mismatch"):
            self.validate()

    def test_test_label_mismatch_rejected_between_models(self):
        tab_dir = self.root / self.manifest["runs"][1]["artifact_dir"]
        prediction_path = tab_dir / "predictions_source_test.csv.gz"
        rows = validation._read_csv(prediction_path, "tab", prediction_path.name)
        rows[0]["y_true"] = "0"
        self._write_predictions(tab_dir, "source", rows)
        self._refresh_domain_results(tab_dir, "source", rows)
        with self.assertRaisesRegex(ReportValidationError, "test label mismatch"):
            self.validate()

    def test_threshold_mismatch_against_manifest_is_rejected(self):
        run_dir = self.root / self.manifest["runs"][0]["artifact_dir"]
        config = self._read_json(run_dir / "config.json")
        config["threshold"] = 0.4
        _write_json(run_dir / "config.json", config)
        metadata = self._read_json(run_dir / "metadata.json")
        metadata["configuration"] = config
        _write_json(run_dir / "metadata.json", metadata)
        for domain in DOMAINS:
            path = run_dir / f"predictions_{domain}_test.csv.gz"
            rows = validation._read_csv(path, "fixture", path.name)
            for row in rows:
                row["y_pred"] = str(int(float(row["y_probability"]) >= config["threshold"]))
            self._write_predictions(run_dir, domain, rows)
            self._refresh_domain_results(run_dir, domain, rows, threshold=0.4)
        with self.assertRaisesRegex(ReportValidationError, "config.json.*threshold"):
            self.validate()

    def test_cross_run_population_identity_mismatch_rejected_across_matched_pairs(self):
        self._change_matched_pair_population("identity")
        with self.assertRaisesRegex(ReportValidationError, "cross-run evaluation identity mismatch"):
            self.validate()

    def test_cross_run_population_label_mismatch_rejected_across_matched_pairs(self):
        self._change_matched_pair_population("label")
        with self.assertRaisesRegex(ReportValidationError, "cross-run evaluation label mismatch"):
            self.validate()

    def test_failed_run_missing_artifact_duplicate_role_and_bad_metric(self):
        run_dir = self.root / self.manifest["runs"][0]["artifact_dir"]
        metadata = self._read_json(run_dir / "metadata.json")
        metadata["status"] = "failed"
        _write_json(run_dir / "metadata.json", metadata)
        with self.assertRaisesRegex(ReportValidationError, "status"):
            self.validate()
        metadata["status"] = "completed"
        _write_json(run_dir / "metadata.json", metadata)
        (run_dir / "calibration_bins.csv").unlink()
        with self.assertRaisesRegex(ReportValidationError, "required artifact is missing"):
            self.validate()
        _write_gzip_or_plain_calibration(
            run_dir / "calibration_bins.csv",
            _calibration_rows(self._read_json(run_dir / "metrics.json"), DOMAINS),
        )
        duplicate = json.loads(json.dumps(self.manifest))
        duplicate["runs"][1]["role"] = duplicate["runs"][0]["role"]
        with self.assertRaisesRegex(ReportValidationError, "duplicate comparison role"):
            validation._check_manifest(duplicate)
        metrics = self._read_json(run_dir / "metrics.json")
        metrics["source"]["roc_auc"] = 0.123
        _write_json(run_dir / "metrics.json", metrics)
        with self.assertRaisesRegex(ReportValidationError, "prediction-derived value"):
            self.validate()

    def test_existing_output_and_symlink_escape_are_refused(self):
        reports = self.root / "results/reports"
        reports.mkdir(parents=True)
        existing = reports / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ReportValidationError, "refusing to overwrite"):
            compare_predictive._output_directory(Path("results/reports/existing"), self.manifest)
        alias = reports / "run_alias"
        alias.symlink_to(self.root / self.manifest["runs"][0]["artifact_dir"], target_is_directory=True)
        with self.assertRaises(ReportValidationError):
            compare_predictive._output_directory(Path("results/reports/run_alias"), self.manifest)

    def test_cli_writes_complete_report_and_refuses_reuse(self):
        output = compare_predictive.run_report(
            self.manifest_path,
            Path("results/reports/first"),
        )
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "manifest.json", "metrics.csv", "metric_differences.csv",
                "runtime.csv", "calibration_bins.csv", "validation.json", "report.md",
            },
        )
        with (output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)
        with (output / "metric_differences.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)
        saved_validation = self._read_json(output / "validation.json")
        self.assertEqual(saved_validation["status"], "completed")
        self.assertEqual(saved_validation["input_validation_status"], "passed")
        self.assertEqual(saved_validation["report_generation_status"], "completed")
        self.assertEqual(len(saved_validation["cross_run_population_checks"]), 9)
        report_manifest = self._read_json(output / "manifest.json")
        self.assertEqual(len(report_manifest["resolved_runs"]), 4)
        self.assertFalse(report_manifest["figures_requested"])
        self.assertEqual(report_manifest["figure_files"], [])
        self.assertIsNone(report_manifest["matplotlib_version"])
        with (output / "calibration_bins.csv").open(encoding="utf-8", newline="") as handle:
            calibration_rows = list(csv.DictReader(handle))
        self.assertEqual(len(calibration_rows), 4 * len(DOMAINS) * 10)
        environment = report_manifest["reporting_environment"]
        self.assertIn("numpy", environment["packages"])
        self.assertIn("pandas", environment["packages"])
        self.assertIn("scikit-learn", environment["packages"])
        self.assertIn("available", environment["git"])
        if environment["git"]["available"]:
            self.assertIn("revision", environment["git"])
            self.assertIn("dirty", environment["git"])
        with self.assertRaisesRegex(ReportValidationError, "refusing to overwrite"):
            compare_predictive.run_report(self.manifest_path, Path("results/reports/first"))

    def test_output_verification_rejects_empty_files_and_unresolved_links(self):
        output = self.root / "output-verification"
        output.mkdir()
        (output / "report.md").write_text("[missing](missing.csv)\n", encoding="utf-8")
        (output / "empty.csv").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(OSError, "missing or empty"):
            compare_predictive._verify_outputs(output, {"report.md", "empty.csv"})
        (output / "empty.csv").write_text("value\n1\n", encoding="utf-8")
        with self.assertRaisesRegex(OSError, "unresolved or empty relative links"):
            compare_predictive._verify_outputs(output, {"report.md", "empty.csv"})
        (output / "missing.csv").write_text("ok\n", encoding="utf-8")
        compare_predictive._verify_outputs(output, {"report.md", "empty.csv"})

    def test_unresolved_report_link_leaves_failure_diagnostic(self):
        with patch.object(compare_predictive, "build_report", return_value="[missing](missing.csv)\n"):
            with self.assertRaisesRegex(OSError, "unresolved or empty relative links"):
                compare_predictive.run_report(
                    self.manifest_path,
                    Path("results/reports/unresolved-link"),
                )
        diagnostic = self._read_json(
            self.root / "results/reports/unresolved-link/validation.json",
        )
        self.assertEqual(diagnostic["failed_stage"], "output_validation")
        self.assertEqual(diagnostic["report_generation_status"], "failed")

    def test_figure_report_outputs_png_pdf_and_resolvable_relative_links(self):
        output = compare_predictive.run_report(
            self.manifest_path,
            Path("results/reports/with-figures"),
            with_figures=True,
        )
        report_manifest = self._read_json(output / "manifest.json")
        self.assertTrue(report_manifest["figures_requested"])
        self.assertEqual(report_manifest["matplotlib_version"], version("matplotlib"))
        self.assertEqual(set(report_manifest["figure_files"]), {
            f"figures/{name}" for name in figures.FIGURE_FILES
        })
        self.assertTrue(all((output / name).is_file() for name in report_manifest["figure_files"]))
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("CatBoost", report)
        self.assertIn("TabPFN", report)
        self.assertIn("calibration_bins.csv", report)
        link_paths = re.findall(r"\]\(([^)]+)\)", report)
        self.assertTrue(link_paths)
        self.assertTrue(all((output / link).is_file() for link in link_paths))
        self.assertEqual(
            set(link_paths),
            {f"figures/{name}" for name in figures.FIGURE_FILES}
            | {"calibration_bins.csv"},
        )
        import matplotlib
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_missing_and_zero_runtime_values_are_not_replaced(self):
        for spec in self.manifest["runs"]:
            run_dir = self.root / spec["artifact_dir"]
            metadata = self._read_json(run_dir / "metadata.json")
            metadata["prediction_duration_seconds"]["source"] = 0
            metadata["prediction_duration_seconds"]["kaggle"] = None
            metadata["prediction_duration_seconds"].pop("nhanes")
            _write_json(run_dir / "metadata.json", metadata)
        runs, _ = self.validate()
        points = figures.runtime_points(runs, self.manifest)
        per_run = [point for point in points if point["training_domain_id"] == "source"]
        self.assertEqual(
            {point["duration_seconds"] for point in per_run if point["evaluation_domain_id"] == "source"},
            {0},
        )
        self.assertTrue(all(
            point["duration_seconds"] is None
            for point in per_run if point["evaluation_domain_id"] in ("kaggle", "nhanes")
        ))
        runtime_rows = tables.build_runtime_rows(runs, self.manifest)
        self.assertEqual(runtime_rows[0]["prediction_source_seconds"], 0)
        self.assertIsNone(runtime_rows[0]["prediction_kaggle_seconds"])
        self.assertIsNone(runtime_rows[0]["prediction_nhanes_seconds"])
        rendered = tables.generate_tables(runs, self.manifest)["runtime.csv"]
        rendered_rows = list(csv.DictReader(rendered.splitlines()))
        self.assertEqual(rendered_rows[0]["prediction_source_seconds"], "0")
        self.assertEqual(rendered_rows[0]["prediction_kaggle_seconds"], "")
        self.assertEqual(rendered_rows[0]["prediction_nhanes_seconds"], "")

    def test_failed_validation_leaves_diagnostic_without_success_report(self):
        run_dir = self.root / self.manifest["runs"][0]["artifact_dir"]
        (run_dir / "calibration_bins.csv").unlink()
        with self.assertRaises(ReportValidationError):
            compare_predictive.run_report(
                self.manifest_path,
                Path("results/reports/failed"),
            )
        failed_output = self.root / "results/reports/failed"
        self.assertEqual(self._read_json(failed_output / "validation.json")["status"], "failed")
        self.assertFalse((failed_output / "report.md").exists())

    def test_table_generation_and_output_writing_failures_are_recorded(self):
        with patch.object(compare_predictive, "generate_tables", side_effect=RuntimeError("table failure")):
            with self.assertRaisesRegex(RuntimeError, "table failure"):
                compare_predictive.run_report(self.manifest_path, Path("results/reports/table-failure"))
        table_diagnostic = self._read_json(
            self.root / "results/reports/table-failure/validation.json",
        )
        self.assertEqual(table_diagnostic["input_validation_status"], "passed")
        self.assertEqual(table_diagnostic["report_generation_status"], "failed")
        self.assertEqual(table_diagnostic["failed_stage"], "table_generation")
        self.assertFalse((self.root / "results/reports/table-failure/report.md").exists())

        original_write_json = compare_predictive._write_json

        def fail_manifest_write(path: Path, value: object) -> None:
            if path.name == "manifest.json":
                raise OSError("manifest write failure")
            original_write_json(path, value)

        with patch.object(compare_predictive, "_write_json", side_effect=fail_manifest_write):
            with self.assertRaisesRegex(OSError, "manifest write failure"):
                compare_predictive.run_report(self.manifest_path, Path("results/reports/write-failure"))
        write_diagnostic = self._read_json(
            self.root / "results/reports/write-failure/validation.json",
        )
        self.assertEqual(write_diagnostic["input_validation_status"], "passed")
        self.assertEqual(write_diagnostic["report_generation_status"], "failed")
        self.assertEqual(write_diagnostic["failed_stage"], "output_writing")

    def test_plot_save_failure_leaves_plotting_failure_diagnostic(self):
        import matplotlib.figure

        with patch.object(matplotlib.figure.Figure, "savefig", side_effect=OSError("figure save failure")):
            with self.assertRaisesRegex(OSError, "figure save failure"):
                compare_predictive.run_report(
                    self.manifest_path,
                    Path("results/reports/plot-failure"),
                    with_figures=True,
                )
        failed_output = self.root / "results/reports/plot-failure"
        diagnostic = self._read_json(failed_output / "validation.json")
        self.assertEqual(diagnostic["input_validation_status"], "passed")
        self.assertEqual(diagnostic["report_generation_status"], "failed")
        self.assertEqual(diagnostic["failed_stage"], "plotting")
        self.assertTrue(diagnostic["figures_requested"])
        self.assertFalse((failed_output / "report.md").exists())

    def test_plot_diagnostic_write_failure_does_not_mask_original_error(self):
        import matplotlib.figure

        original = OSError("original figure save failure")
        write_json = compare_predictive._write_json

        def fail_diagnostic(path: Path, value: object) -> None:
            if path.name == "validation.json" and isinstance(value, dict) and value.get("status") == "failed":
                raise OSError("diagnostic write failure")
            write_json(path, value)

        with (
            patch.object(matplotlib.figure.Figure, "savefig", side_effect=original),
            patch.object(compare_predictive, "_write_json", side_effect=fail_diagnostic),
            patch("sys.stderr"),
        ):
            with self.assertRaises(OSError) as caught:
                compare_predictive.run_report(
                    self.manifest_path,
                    Path("results/reports/plot-diagnostic-failure"),
                    with_figures=True,
                )
        self.assertIs(caught.exception, original)

    def test_diagnostic_write_failure_does_not_replace_original_error(self):
        original = RuntimeError("original validation failure")
        with (
            patch.object(compare_predictive, "validate_comparison", side_effect=original),
            patch.object(compare_predictive, "_write_json", side_effect=OSError("diagnostic failure")),
            patch("sys.stderr"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                compare_predictive.run_report(
                    self.manifest_path,
                    Path("results/reports/diagnostic-failure"),
                )
        self.assertIs(caught.exception, original)

    def test_git_provenance_unavailable_is_explicit_and_nonfatal(self):
        with patch.object(compare_predictive.subprocess, "run", side_effect=FileNotFoundError("git unavailable")):
            environment = compare_predictive._reporting_environment()
        self.assertFalse(environment["git"]["available"])
        self.assertIsNone(environment["git"]["revision"])
        self.assertIsNone(environment["git"]["dirty"])
        self.assertIn("git unavailable", environment["git"]["error"])

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_sample(run_dir: Path, rows: list[dict]) -> None:
        _write_gzip_csv(
            run_dir / "training_sample.csv.gz",
            ["domain", "split", "sample_order", "row_index"],
            rows,
        )

    @staticmethod
    def _write_predictions(run_dir: Path, domain: str, rows: list[dict]) -> None:
        _write_gzip_csv(
            run_dir / f"predictions_{domain}_test.csv.gz",
            list(rows[0]),
            rows,
        )

    def _refresh_domain_results(
        self,
        run_dir: Path,
        domain: str,
        rows: list[dict],
        threshold: float = 0.5,
    ) -> None:
        metrics = self._read_json(run_dir / "metrics.json")
        metrics[domain] = evaluate_binary(
            [int(row["y_true"]) for row in rows],
            [float(row["y_probability"]) for row in rows],
            threshold,
            10,
        )
        _write_json(run_dir / "metrics.json", metrics)
        calibration_path = run_dir / "calibration_bins.csv"
        existing = validation._read_csv(calibration_path, "fixture", calibration_path.name)
        retained = [row for row in existing if row["domain"] != domain]
        _write_gzip_or_plain_calibration(
            calibration_path,
            retained + _calibration_rows({domain: metrics[domain]}, [domain]),
        )
        metadata = self._read_json(run_dir / "metadata.json")
        identity_column = "seqn" if domain == "nhanes" else "row_index"
        metadata["splits"][f"{domain}/test"] = _summary(
            domain,
            "test",
            [int(row[identity_column]) for row in rows],
            [int(row["y_true"]) for row in rows],
        )
        _write_json(run_dir / "metadata.json", metadata)

    def _change_matched_pair_population(self, mismatch: str) -> None:
        pair_specs = [
            spec for spec in self.manifest["runs"]
            if spec["train_domain"] == "kaggle"
        ]
        for spec in pair_specs:
            run_dir = self.root / spec["artifact_dir"]
            path = run_dir / "predictions_source_test.csv.gz"
            rows = validation._read_csv(path, spec["run_id"], path.name)
            target = next(row for row in rows if row["row_index"] == "10")
            if mismatch == "identity":
                target["row_index"] = str(int(target["row_index"]) + 10000)
            else:
                target["y_true"] = str(1 - int(target["y_true"]))
            self._write_predictions(run_dir, "source", rows)
            self._refresh_domain_results(run_dir, "source", rows)


class ReportingImportTests(unittest.TestCase):
    def test_help_does_not_initialize_model_or_gpu_runtime(self):
        code = (
            "import runpy,sys;"
            "sys.argv=['compare_predictive','--help'];"
            "\ntry: runpy.run_module('src.reporting.compare_predictive',run_name='__main__')"
            "\nexcept SystemExit as exc: assert exc.code == 0"
            "\nassert not ({'catboost','tabpfn','torch'} & sys.modules.keys())"
            "\nassert 'matplotlib' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)


class TabICLReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, self.manifest_path = tabicl_fixture(self.root)
        self.patches = [
            patch.object(validation, "PROJECT_ROOT", self.root),
            patch.object(validation, "RESULTS_DIR", self.root / "results"),
            patch.object(compare_predictive, "PROJECT_ROOT", self.root),
            patch.object(compare_predictive, "RESULTS_DIR", self.root / "results"),
        ]
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)

    def validate(self):
        return validation.validate_comparison(self.manifest, self.manifest_path)

    def run_dir(self, role: str) -> Path:
        spec = next(run for run in self.manifest["runs"] if run["role"] == role)
        return self.root / spec["artifact_dir"]

    def read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def update_evaluation(self, role: str, domain: str, rows: list[dict]) -> None:
        run_dir = self.run_dir(role)
        PredictiveReportingTests._write_predictions(run_dir, domain, rows)
        metrics_path = run_dir / "metrics.json"
        metrics = self.read_json(metrics_path)
        config = self.read_json(run_dir / "config.json")
        metrics[domain] = evaluate_binary(
            [int(row["y_true"]) for row in rows],
            [float(row["y_probability"]) for row in rows],
            config["threshold"],
            config["calibration_bins"],
        )
        _write_json(metrics_path, metrics)
        calibration_path = run_dir / "calibration_bins.csv"
        existing = validation._read_csv(calibration_path, role, calibration_path.name)
        retained = [row for row in existing if row["domain"] != domain]
        _write_gzip_or_plain_calibration(
            calibration_path,
            retained + _calibration_rows({domain: metrics[domain]}, [domain]),
        )
        metadata_path = run_dir / "metadata.json"
        metadata = self.read_json(metadata_path)
        identity_column = "seqn" if domain == "nhanes" else "row_index"
        metadata["splits"][f"{domain}/test"] = _summary(
            domain,
            "test",
            [int(row[identity_column]) for row in rows],
            [int(row["y_true"]) for row in rows],
        )
        _write_json(metadata_path, metadata)

    def test_six_run_tables_have_declared_order_counts_profiles_and_signs(self):
        runs, result = self.validate()
        self.assertEqual(len(runs), 6)
        self.assertEqual(len(result["cross_run_population_checks"]), 15)
        self.assertTrue(all(
            check["identities_match"] and check["labels_match"]
            for check in result["cross_run_population_checks"]
        ))
        for domain in ("source", "kaggle"):
            selected = [run for run in runs if run.spec["train_domain"] == domain]
            self.assertEqual(len({tuple(run.training_identities) for run in selected}), 1)
        metric_rows = tables.build_metric_rows(runs, self.manifest)
        difference_rows = tables.build_difference_rows(runs, self.manifest)
        runtime_rows = tables.build_runtime_rows(runs, self.manifest)
        calibration_rows = tables.build_calibration_rows(runs, self.manifest)
        self.assertEqual(len(metric_rows), 18)
        self.assertEqual(len(difference_rows), 12)
        self.assertEqual(len(runtime_rows), 6)
        self.assertEqual(len(calibration_rows), 180)
        figure_points = figures.performance_points(runs, self.manifest, "f1")
        self.assertEqual({point["model_label"] for point in figure_points}, {
            "CatBoost", "TabPFN", "TabICLv2",
        })
        self.assertEqual({point["model_marker"] for point in figure_points}, {"o", "s", "^"})
        self.assertEqual(
            [(row["train_domain_id"], row["model"]) for row in metric_rows[::3]],
            [
                ("source", "catboost"), ("source", "tabpfn"), ("source", "tabicl"),
                ("kaggle", "catboost"), ("kaggle", "tabpfn"), ("kaggle", "tabicl"),
            ],
        )
        self.assertEqual(
            [row["comparison_id"] for row in difference_rows[::3]],
            [
                "source_tabpfn_vs_catboost",
                "source_tabicl_vs_catboost",
                "kaggle_tabpfn_vs_catboost",
                "kaggle_tabicl_vs_catboost",
            ],
        )
        for row in difference_rows:
            self.assertGreater(row["candidate_minus_baseline_roc_auc"], 0)
            self.assertGreater(row["candidate_minus_baseline_average_precision"], 0)
            self.assertGreater(row["candidate_minus_baseline_f1"], 0)
            self.assertLess(row["candidate_minus_baseline_brier_score"], 0)
        self.assertEqual(
            [row["external_query_batch_size"] for row in runtime_rows],
            [256, 256, 128, 256, 1024, 128],
        )
        self.assertEqual(
            [row["native_ensemble_batch_size"] for row in runtime_rows],
            [None, None, 1, None, None, 1],
        )
        self.assertEqual(
            [row["model_profile_id"] for row in metric_rows[::3]],
            ["catboost_cpu", "tabpfn_v35_float32", "tabicl_v2_none_fp32_uncached"] * 2,
        )
        self.assertTrue(result["expected_methodological_differences"])
        self.assertEqual(
            [item["recovery_detection"] for item in result["native_diagnostic_evidence"]],
            ["no_recovery_detected", "no_recovery_detected"],
        )

        rendered = tables.generate_tables(runs, self.manifest)
        self.assertEqual(
            len(list(csv.DictReader(rendered["metrics.csv"].splitlines()))),
            18,
        )
        self.assertEqual(
            len(list(csv.DictReader(rendered["metric_differences.csv"].splitlines()))),
            12,
        )
        report = tables.build_report(runs, self.manifest, result, "timestamp")
        self.assertIn("TabICL", report)
        self.assertIn("candidate minus CatBoost", report)
        self.assertIn("not the vendor-default normalization ensemble", report)
        self.assertIn("do not prove unchanged scheduling", report)
        self.assertIn("not national-population estimates", report)

    def test_candidate_baseline_declarations_are_validated(self):
        cases = [
            (
                "duplicate comparison",
                lambda m: m["comparisons"][1].update(
                    id=m["comparisons"][0]["id"],
                    baseline_role="source_catboost",
                    candidate_role="source_tabpfn",
                    training_domain="source",
                ),
                "duplicate comparison id",
            ),
            ("unknown role", lambda m: m["comparisons"][0].update(candidate_role="missing_role"), "unknown run role"),
            ("self comparison", lambda m: m["comparisons"][0].update(candidate_role="source_catboost"), "cannot compare a role with itself"),
            ("cross domain", lambda m: m["comparisons"][0].update(candidate_role="kaggle_tabpfn"), "cross-training-domain"),
            ("missing candidate", lambda m: m["comparisons"].pop(), "four explicit comparisons"),
            (
                "duplicate run id",
                lambda m: m["runs"][1].update(run_id=m["runs"][0]["run_id"]),
                "duplicate run_id",
            ),
            (
                "duplicate artifact path",
                lambda m: m["runs"][1].update(artifact_dir=m["runs"][0]["artifact_dir"]),
                "duplicate artifact_dir",
            ),
        ]
        for _, mutate, expected in cases:
            broken = json.loads(json.dumps(self.manifest))
            mutate(broken)
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ReportValidationError, expected,
            ):
                validation._check_manifest(broken)

    def test_shared_calibration_boundaries_are_validated_before_plotting(self):
        runs, _ = self.validate()
        runs[1].metrics["source"]["calibration_bins"][0]["upper_bound"] = 0.25
        with self.assertRaisesRegex(ReportValidationError, "calibration bin boundaries differ"):
            validation._check_calibration_boundary_compatibility(runs, DOMAINS)

    def test_third_model_training_identity_sequence_mismatch_is_rejected(self):
        run_dir = self.run_dir("source_tabicl")
        sample_path = run_dir / "training_sample.csv.gz"
        rows = validation._read_csv(sample_path, "tabicl", sample_path.name)
        rows[-1]["row_index"] = str(int(rows[-1]["row_index"]) + 100000)
        identities = [int(row["row_index"]) for row in rows]
        PredictiveReportingTests._write_sample(run_dir, rows)
        metadata_path = run_dir / "metadata.json"
        metadata = self.read_json(metadata_path)
        fingerprints = identity_fingerprints("source", "row_index", identities)
        metadata["training_context"].update(fingerprints)
        metadata["sampling"].update(fingerprints)
        _write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ReportValidationError, "training identity mismatch for source"):
            self.validate()

    def test_duplicate_resolved_run_directories_are_rejected(self):
        self.manifest["runs"][1]["artifact_dir"] = (
            "results/predictive/source_catboost/../source_catboost"
        )
        with self.assertRaisesRegex(ReportValidationError, "same resolved run directory"):
            self.validate()

    def test_third_model_profile_mismatch_is_rejected(self):
        run_dir = self.run_dir("source_tabicl")
        config_path = run_dir / "config.json"
        config = self.read_json(config_path)
        config["model_params"]["norm_methods"] = ["power"]
        _write_json(config_path, config)
        metadata_path = run_dir / "metadata.json"
        metadata = self.read_json(metadata_path)
        metadata["configuration"] = config
        _write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ReportValidationError, "TabICL norm_methods"):
            self.validate()

    def test_third_model_evaluation_identity_mismatch_is_rejected(self):
        run_dir = self.run_dir("source_tabicl")
        path = run_dir / "predictions_source_test.csv.gz"
        rows = validation._read_csv(path, "tabicl", path.name)
        rows[0]["row_index"] = "999999"
        self.update_evaluation("source_tabicl", "source", rows)
        with self.assertRaisesRegex(ReportValidationError, "test identity mismatch for source-trained pair"):
            self.validate()

    def test_failed_tabicl_artifact_and_prediction_metric_inconsistency_are_rejected(self):
        run_dir = self.run_dir("source_tabicl")
        (run_dir / "calibration_bins.csv").unlink()
        with self.assertRaisesRegex(ReportValidationError, "required artifact is missing"):
            self.validate()
        with self.assertRaisesRegex(ReportValidationError, "required artifact is missing"):
            compare_predictive.run_report(
                self.manifest_path,
                Path("results/reports/tabicl-incomplete"),
            )
        diagnostic = self.read_json(
            self.root / "results/reports/tabicl-incomplete/validation.json",
        )
        self.assertEqual(diagnostic["input_validation_status"], "failed")
        self.assertFalse(
            (self.root / "results/reports/tabicl-incomplete/report.md").exists()
        )

        metrics_path = run_dir / "metrics.json"
        metrics = self.read_json(metrics_path)
        _write_gzip_or_plain_calibration(
            run_dir / "calibration_bins.csv",
            _calibration_rows(metrics, DOMAINS),
        )
        metrics["source"]["roc_auc"] = 0.123
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ReportValidationError, "prediction-derived value"):
            self.validate()

    def test_schema_two_figures_include_three_models_and_resolvable_links(self):
        run_dir = self.run_dir("source_tabicl")
        metadata_path = run_dir / "metadata.json"
        metadata = self.read_json(metadata_path)
        metadata["resources"] = {"peak_gpu_memory": None, "total_gpu_usage_bytes": None}
        _write_json(metadata_path, metadata)
        runs, _ = self.validate()
        runtime = tables.build_runtime_rows(runs, self.manifest)
        row = next(item for item in runtime if item["run_id"] == "tabicl_v2_source_strict_common_16_n2000_none_fp32_uncached_seed42")
        self.assertIsNone(row["pytorch_peak_allocated_bytes"])
        self.assertIsNone(row["pytorch_peak_reserved_bytes"])
        self.assertIsNone(row["total_gpu_usage_bytes"])
        output = compare_predictive.run_report(
            self.manifest_path,
            Path("results/reports/tabicl-figures"),
            with_figures=True,
        )
        report_manifest = self.read_json(output / "manifest.json")
        self.assertEqual(report_manifest["report_generation_status"], "completed")
        self.assertEqual(len(report_manifest["figure_files"]), 12)
        self.assertEqual(set(report_manifest["figure_files"]), {
            f"figures/{name}" for name in figures.FIGURE_FILES
        })
        self.assertTrue(all(
            (output / name).is_file() and (output / name).stat().st_size > 0
            for name in report_manifest["figure_files"]
        ))
        with (output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 18)
        with (output / "metric_differences.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn(
            "Ranking performance (ROC-AUC and average precision), classification performance "
            "at the fixed threshold (F1), and probability quality (Brier score) measure different aspects",
            report,
        )
        self.assertIn("### Timing (seconds)", report)
        self.assertIn("### Execution profile and resources", report)
        self.assertIn("Not recorded", report)
        self.assertIn("verbose native console echo", report)
        link_paths = re.findall(r"\]\(([^)]+)\)", report)
        self.assertEqual(
            set(link_paths),
            {f"figures/{name}" for name in figures.FIGURE_FILES}
            | {"calibration_bins.csv"},
        )
        self.assertTrue(all((output / link).is_file() for link in link_paths))

    def test_tabicl_runtime_points_preserve_missing_and_zero_durations(self):
        for spec in self.manifest["runs"]:
            metadata_path = self.run_dir(spec["role"]) / "metadata.json"
            metadata = self.read_json(metadata_path)
            metadata["prediction_duration_seconds"]["source"] = 0
            metadata["prediction_duration_seconds"]["kaggle"] = None
            _write_json(metadata_path, metadata)
        runs, _ = self.validate()
        points = figures.runtime_points(runs, self.manifest)
        self.assertTrue(any(point["duration_seconds"] == 0 for point in points))
        self.assertTrue(any(point["duration_seconds"] is None for point in points))

    def test_legacy_csv_output_contract_and_values_remain_unchanged(self):
        legacy_root = self.root / "legacy"
        legacy_manifest, legacy_path = fixture(legacy_root)
        with (
            patch.object(validation, "PROJECT_ROOT", legacy_root),
            patch.object(validation, "RESULTS_DIR", legacy_root / "results"),
        ):
            runs, _ = validation.validate_comparison(legacy_manifest, legacy_path)
            rendered = tables.generate_tables(runs, legacy_manifest)
        metric_rows = list(csv.DictReader(rendered["metrics.csv"].splitlines()))
        self.assertEqual(
            list(metric_rows[0]),
            [
                "run_id", "model", "train_domain_id", "train_domain_label",
                "evaluation_domain_id", "evaluation_domain_label", "feature_set",
                "training_budget", "model_seed", "sampling_seed", "threshold",
                "prediction_batch_size", "test_sample_size", "positive_count",
                "positive_prevalence", "roc_auc", "average_precision", "f1", "brier_score",
            ],
        )
        first_metric = next(csv.DictReader(rendered["metrics.csv"].splitlines()))
        self.assertEqual(first_metric["model"], "catboost")
        self.assertEqual(first_metric["train_domain_id"], "source")
        self.assertEqual(first_metric["evaluation_domain_id"], "source")
        self.assertEqual(first_metric["training_budget"], "2000")
        self.assertEqual(first_metric["roc_auc"], "0.375")
        differences = next(csv.DictReader(rendered["metric_differences.csv"].splitlines()))
        self.assertEqual(differences["tabpfn_minus_catboost_roc_auc"], "0.625")
        self.assertNotIn("candidate_minus_baseline_roc_auc", differences)


if __name__ == "__main__":
    unittest.main()
