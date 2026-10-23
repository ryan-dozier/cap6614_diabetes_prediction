from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.data import DatasetSchema
from src.experiments import config as config_module
from src.experiments import runner
from src.experiments.config import ExperimentConfig, load_config
from src.experiments.metrics import evaluate_binary
from src.experiments.results import create_run_directory, write_json
from src.models.catboost import CatBoostAdapter
from src.paths import CONFIGS_DIR, PROJECT_ROOT


def sample_values() -> dict:
    return json.loads((CONFIGS_DIR / "catboost_source.json").read_text())


class ConfigTests(unittest.TestCase):
    def test_load_and_resolved_serialization(self):
        config = load_config(CONFIGS_DIR / "catboost_source.json")
        self.assertEqual(config.test_domains, ("source", "kaggle", "nhanes"))
        self.assertEqual(config.feature_set, "strict_common_16")
        self.assertEqual(config.to_dict()["output_dir"], str(config.resolved_output_dir))
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())

    def test_invalid_values_and_types(self):
        invalid = [
            {"unknown": 1}, {"model": "tabpfn"}, {"train_domain": []},
            {"validation_domain": "kaggle"}, {"test_domains": []},
            {"test_domains": ["source", "source"]}, {"test_domains": "source"},
            {"test_domains": ["unknown"]}, {"feature_set": "full_24"},
            {"feature_set": None}, {"seed": True}, {"seed": "42"}, {"seed": -1},
            {"threshold": True}, {"threshold": 0}, {"threshold": 1},
            {"threshold": float("nan")}, {"threshold": "0.5"},
            {"calibration_bins": False}, {"calibration_bins": 1.5},
            {"early_stopping_rounds": 0}, {"early_stopping_rounds": "100"},
            {"output_dir": "../escape"}, {"output_dir": "/tmp/escape"},
            {"output_dir": "results/../../escape"}, {"output_dir": "results"},
            {"experiment_name": ""}, {"model_params": []},
        ]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**sample_values(), **update})
        values = sample_values()
        del values["feature_set"]
        with self.assertRaisesRegex(ValueError, "missing"):
            ExperimentConfig.from_dict(values)

    def test_parameter_allowlist_and_types(self):
        for key in (
            "random_seed", "random_state", "allow_writing_files", "use_best_model",
            "early_stopping_rounds", "od_wait", "od_type", "od_pval", "n_estimators",
            "num_boost_round", "cat_features", "class_weights", "train_dir",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**sample_values(), "model_params": {key: 42}})
        for params in (
            {"iterations": True}, {"iterations": "10"}, {"depth": 0},
            {"learning_rate": float("inf")}, {"learning_rate": False},
            {"loss_function": "MultiClass"}, {"eval_metric": "Accuracy"},
            {"thread_count": 0}, {"thread_count": True}, {"task_type": "TPU"},
            {"verbose": "100"}, {"l2_leaf_reg": -1},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**sample_values(), "model_params": params})

    def test_symlink_escape_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            (root / "outside").mkdir()
            (root / "results" / "escape").symlink_to(root / "outside", target_is_directory=True)
            with patch.multiple(config_module, PROJECT_ROOT=root, RESULTS_DIR=root / "results"):
                with self.assertRaises(ValueError):
                    ExperimentConfig.from_dict({**sample_values(), "output_dir": "results/escape/run"})
                config = ExperimentConfig.from_dict({**sample_values(), "output_dir": "results/run"})
                output = create_run_directory(config)
                self.assertTrue(output.is_dir())
                with self.assertRaises(FileExistsError):
                    create_run_directory(config)


class MetricsTests(unittest.TestCase):
    def test_boundaries_empty_bins_and_threshold(self):
        result = evaluate_binary([0, 1, 1], [0, 1, 0.5], 0.5, 4)
        self.assertEqual(result["confusion_matrix"], {"tn": 1, "fp": 0, "fn": 0, "tp": 2})
        self.assertEqual([entry["count"] for entry in result["calibration_bins"]], [1, 0, 1, 1])
        empty = result["calibration_bins"][1]
        for key in ("mean_predicted_probability", "observed_positive_rate", "absolute_gap"):
            self.assertIsNone(empty[key])
        self.assertAlmostEqual(result["ece"], 1 / 6)
        self.assertEqual(result["average_precision"], 1)
        self.assertTrue(np.isfinite(result["log_loss"]))
        self.assertEqual(result["threshold"], 0.5)
        json.dumps(result, allow_nan=False)

    def test_one_class(self):
        for labels in ([0, 0], [1, 1]):
            result = evaluate_binary(labels, [0, 1], calibration_bins=2)
            self.assertIsNone(result["roc_auc"])
            self.assertIsNone(result["average_precision"])
            self.assertIn("one class", result["warnings"][0])
            self.assertTrue(np.isfinite(result["log_loss"]))
            self.assertEqual(sum(result["confusion_matrix"].values()), 2)
            json.dumps(result, allow_nan=False)

    def test_invalid_inputs(self):
        for labels, probability in (
            ([], []), ([0, 1], [0]), ([[0, 1]], [0, 1]),
            ([0, 1], [[0, 1]]), ([0, 2], [0, 1]), ([0, np.nan], [0, 1]),
            ([0, 1], [np.nan, 1]), ([0, 1], [np.inf, 1]),
            ([0, 1], [-0.01, 1]), ([0, 1], [0, 1.01]),
            ([0, 1], ["0", "1"]),
        ):
            with self.subTest(labels=labels, probability=probability), self.assertRaises(ValueError):
                evaluate_binary(labels, probability)
        with self.assertRaises(ValueError):
            evaluate_binary([0, 1], [0, 1], calibration_bins=True)
        for threshold in (0, 1, float("nan"), True):
            with self.assertRaises(ValueError):
                evaluate_binary([0, 1], [0, 1], threshold=threshold)

    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            write_json(path, {"number": np.int64(1), "undefined": None})
            self.assertEqual(json.loads(path.read_text()), {"number": 1, "undefined": None})
            for value in (float("nan"), np.float64("inf")):
                with self.assertRaises(ValueError):
                    write_json(path, {"invalid": value})


class AdapterTests(unittest.TestCase):
    def adapter(self) -> CatBoostAdapter:
        return CatBoostAdapter(
            ("category", "value"), ("category",), "diagnosed_diabetes",
            seed=42,
            model_params={"iterations": 5, "depth": 2, "verbose": False, "thread_count": 1},
            early_stopping_rounds=2,
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"category": [1, 2] * 6, "value": np.arange(12, dtype=float)})

    def test_tiny_fit_native_save_and_no_mutation(self):
        adapter = self.adapter()
        X = self.frame()
        original = X.copy(deep=True)
        y = pd.Series([0, 1] * 6)
        with self.assertRaises(RuntimeError):
            adapter.predict_proba(X)
        adapter.fit(X, y, X.copy(), y.copy())
        probability = adapter.predict_proba(X)
        self.assertEqual(probability.shape, (12,))
        np.testing.assert_array_equal(adapter.predict(X), probability >= 0.5)
        pd.testing.assert_frame_equal(X, original)
        metadata = adapter.metadata()
        self.assertEqual(metadata["classes"], [0, 1])
        self.assertFalse(metadata["requested_parameters"]["allow_writing_files"])
        self.assertTrue(metadata["requested_parameters"]["use_best_model"])
        self.assertEqual(metadata["requested_parameters"]["random_seed"], 42)
        json.dumps(metadata, allow_nan=False)
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.cbm"
            adapter.save(model_path)
            self.assertGreater(model_path.stat().st_size, 0)
        with self.assertRaises(RuntimeError):
            adapter.fit(X, y, X, y)

    def test_bad_schema_missing_nonfinite_and_labels(self):
        X = self.frame()
        y = pd.Series([0, 1] * 6)
        for bad in (
            X[["value", "category"]], X.assign(id=1), X.assign(value=np.nan),
            X.assign(value=np.inf), X.assign(category=None), X.assign(category=np.inf),
        ):
            with self.subTest(columns=list(bad)), self.assertRaises(ValueError):
                self.adapter().fit(bad, y, X, y)
        for labels in (y.iloc[:-1], pd.Series([0] * 12), pd.Series([2] * 12)):
            with self.assertRaises(ValueError):
                self.adapter().fit(X, labels, X, y)
            with self.assertRaises(ValueError):
                self.adapter().fit(X, y, X, labels)
        with self.assertRaises(ValueError):
            CatBoostAdapter(("id",), (), "target", seed=42, model_params={}, early_stopping_rounds=1)

    def test_positive_class_lookup_and_probability_validation(self):
        adapter = self.adapter()
        model = Mock()
        model.classes_ = np.array([1, 0])
        adapter._model = model
        model.predict_proba.return_value = np.tile([0.5, 0.5], (12, 1))
        np.testing.assert_array_equal(adapter.predict(self.frame()), np.ones(12, dtype=int))
        model.predict_proba.return_value = np.tile([0.8, 0.2], (12, 1))
        np.testing.assert_allclose(adapter.predict_proba(self.frame()), 0.8)
        for bad in (np.ones(12), np.ones((12, 2)), np.full((12, 2), np.nan)):
            model.predict_proba.return_value = bad
            with self.assertRaises(ValueError):
                adapter.predict_proba(self.frame())


class RunnerTests(unittest.TestCase):
    schema = DatasetSchema(
        target="diagnosed_diabetes", features=("value",), categorical_features=(),
        numeric_features=("value",), source_only_excluded=(),
    )

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.multiple(
            config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results",
        ))
        self.config = ExperimentConfig.from_dict({**sample_values(), "output_dir": "results/run"})
        self.events = []
        self.stack.enter_context(patch.object(runner.data, "load_feature_schema", return_value=self.schema))
        self.stack.enter_context(patch.object(runner, "prepared_fingerprints", return_value={"prepared.csv": "abc"}))
        self.stack.enter_context(patch.object(runner.data, "load_split", side_effect=self.load_split))
        self.factory = self.stack.enter_context(patch("src.models.catboost.CatBoostAdapter"))
        self.adapter = self.factory.return_value
        self.adapter.fit.side_effect = lambda *args: self.events.append("fit")
        self.adapter.predict_proba.return_value = np.array([0.5, 0.8])
        self.adapter.save.side_effect = lambda path: path.write_bytes(b"mock model")
        self.adapter.metadata.return_value = {"preprocessing": {"policy": "test"}, "compute_settings": {"task_type": "CPU"}}
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def load_split(self, domain, split, **kwargs):
        self.events.append((domain, split))
        self.assertTrue(kwargs["include_row_index"])
        self.assertEqual(kwargs["include_seqn"], domain == "nhanes")
        first = {"train": 0, "val": 2, "test": 4}[split]
        frame = pd.DataFrame({
            "row_index": [first, first + 1],
            "value": [1.0, 2.0],
            "diagnosed_diabetes": [0, 1],
        })
        if domain == "nhanes":
            frame.insert(1, "seqn", [100 + first, 101 + first])
        return frame

    def run_metadata(self):
        return json.loads((self.config.resolved_output_dir / "metadata.json").read_text())

    def test_fit_once_then_three_tests_and_artifacts(self):
        output = runner.run(self.config)
        self.assertEqual(self.events, [
            ("source", "train"), ("source", "val"), "fit",
            ("source", "test"), ("kaggle", "test"), ("nhanes", "test"),
        ])
        self.adapter.fit.assert_called_once()
        self.assertEqual(tuple(self.adapter.fit.call_args.args[0].columns), ("value",))
        self.assertEqual(self.run_metadata()["status"], "completed")
        self.assertEqual(len(self.run_metadata()["splits"]), 5)
        metrics = pd.read_csv(output / "metrics.csv")
        self.assertEqual(metrics["domain"].tolist(), ["source", "kaggle", "nhanes"])
        bins = pd.read_csv(output / "calibration_bins.csv")
        self.assertEqual(len(bins), 30)
        self.assertEqual(set(bins["split"]), {"test"})
        for domain in self.config.test_domains:
            predictions = pd.read_csv(output / f"predictions_{domain}_test.csv.gz")
            self.assertEqual(predictions["y_pred"].tolist(), [1, 1])
            self.assertIn("row_index", predictions)
            self.assertEqual("seqn" in predictions, domain == "nhanes")
        with self.assertRaises(FileExistsError):
            runner.run(self.config)
        self.adapter.fit.assert_called_once()
        self.assertEqual(self.run_metadata()["status"], "completed")

    def test_fit_failure_marks_run_failed(self):
        self.adapter.fit.side_effect = RuntimeError("fit broke")
        with self.assertRaisesRegex(RuntimeError, "fit broke"):
            runner.run(self.config)
        metadata = self.run_metadata()
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["message"], "fit broke")
        self.assertTrue((self.config.resolved_output_dir / "config.json").exists())
        self.assertFalse(any(split == "test" for item in self.events if isinstance(item, tuple) for split in [item[1]]))

    def test_prediction_failure_preserves_model_and_prior_results(self):
        self.adapter.predict_proba.side_effect = [np.array([0.5, 0.8]), RuntimeError("predict broke")]
        with self.assertRaisesRegex(RuntimeError, "predict broke"):
            runner.run(self.config)
        self.assertEqual(self.run_metadata()["status"], "failed")
        output = self.config.resolved_output_dir
        self.assertTrue((output / "model.cbm").exists())
        self.assertTrue((output / "predictions_source_test.csv.gz").exists())

    def test_failure_reporting_does_not_mask_original(self):
        with patch.object(runner, "write_json", side_effect=[OSError("original error"), RuntimeError("reporting error")]):
            with self.assertLogs(runner.LOGGER, level="ERROR"):
                with self.assertRaisesRegex(OSError, "original error"):
                    runner.run(self.config)

    def test_namespaced_and_nhanes_identities(self):
        source = runner.load_partition("source", "train", "strict_common_16", self.schema)
        kaggle = runner.load_partition("kaggle", "train", "strict_common_16", self.schema)
        runner.require_disjoint(source, kaggle)
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.require_disjoint(source, source)
        nhanes = runner.load_partition("nhanes", "train", "strict_common_16", self.schema)
        self.assertEqual(nhanes.identities, frozenset({("nhanes", 100), ("nhanes", 101)}))
        self.assertEqual(nhanes.summary["identity_column"], "seqn")

    def test_overlap_rejected_before_fit_or_test_prediction(self):
        original = self.load_split

        def overlapping(domain, split, **kwargs):
            frame = original(domain, split, **kwargs)
            if split == "val":
                frame["row_index"] = [0, 1]
            return frame

        with patch.object(runner.data, "load_split", side_effect=overlapping):
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.run(self.config)
        self.adapter.fit.assert_not_called()
        self.assertEqual(self.run_metadata()["status"], "failed")

    def test_test_overlap_rejected_before_prediction(self):
        original = self.load_split

        def overlapping(domain, split, **kwargs):
            frame = original(domain, split, **kwargs)
            if domain == "source" and split == "test":
                frame["row_index"] = [0, 1]
            return frame

        with patch.object(runner.data, "load_split", side_effect=overlapping):
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.run(self.config)
        self.adapter.fit.assert_called_once()
        self.adapter.predict_proba.assert_not_called()
        self.assertEqual(self.run_metadata()["status"], "failed")

    def test_nhanes_overlap_uses_participants_not_row_indices(self):
        original = self.load_split

        def repeated_participant(domain, split, **kwargs):
            frame = original(domain, split, **kwargs)
            if split == "val":
                frame["seqn"] = [100, 101]
            return frame

        with patch.object(runner.data, "load_split", side_effect=repeated_participant):
            train = runner.load_partition("nhanes", "train", "strict_common_16", self.schema)
            val = runner.load_partition("nhanes", "val", "strict_common_16", self.schema)
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.require_disjoint(train, val)


class ProvenanceAndImportTests(unittest.TestCase):
    def test_fingerprints_hash_distinct_file_once(self):
        config = load_config(CONFIGS_DIR / "catboost_source.json")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prepared"
            path.write_bytes(b"prepared fixture")
            domains = {domain: path for domain in config.test_domains}
            with patch.multiple(
                runner.data, METADATA_PATH=path, NHANES_METADATA_PATH=path,
                COMMON_PATHS=domains, NHANES_COMMON_PATHS={"strict_common_16": path},
                SPLIT_ASSIGNMENT_PATHS=domains,
            ):
                original_open = Path.open
                reads = []

                def tracked_open(file, *args, **kwargs):
                    reads.append(file)
                    return original_open(file, *args, **kwargs)

                with patch.object(Path, "open", tracked_open):
                    result = runner.prepared_fingerprints(config)
                self.assertEqual(reads, [path])
                self.assertEqual(list(result.values()), [hashlib.sha256(b"prepared fixture").hexdigest()])

    def test_general_imports_are_inert_and_help_works(self):
        code = """
import sys
from unittest.mock import patch
from src import data
with patch.object(data, 'load_feature_schema', side_effect=AssertionError('schema loaded')), \\
     patch.object(data, 'load_split', side_effect=AssertionError('data loaded')), \\
     patch.object(data, '_load_metadata_cached', side_effect=AssertionError('metadata loaded')):
    import src.experiments.config
    import src.experiments.metrics
    import src.experiments.results
    import src.experiments.runner
    import src.models.base
    import src.models.catboost
assert 'catboost' not in sys.modules
assert 'torch' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        help_result = subprocess.run(
            [sys.executable, "-m", "src.experiments.runner", "--help"],
            cwd=PROJECT_ROOT, text=True, capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--config", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
