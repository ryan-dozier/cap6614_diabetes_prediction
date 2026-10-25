from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.data import DatasetSchema
from src.experiments import config as config_module
from src.experiments import runner
from src.experiments import smoke_tabpfn
from src.experiments.config import ExperimentConfig, load_config
from src.experiments.sampling import Partition, partition_summary, select_training
from src.models import tabpfn as tabpfn_module
from src.models.tabpfn import CHECKPOINT_FILENAME, TabPFNAdapter
from src.paths import CONFIGS_DIR, PROJECT_ROOT


def fixture(domain="source", split="train", count=20) -> Partition:
    row = np.arange(count, dtype=np.int64)
    provenance = pd.DataFrame({"row_index": row})
    if domain == "nhanes":
        provenance["seqn"] = 100 + row[::-1]
    y = pd.Series((row % 4 == 0).astype(np.int64), name="diagnosed_diabetes")
    identity = "seqn" if domain == "nhanes" else "row_index"
    return Partition(
        pd.DataFrame({"value": row / 100, "category": np.where(y, "a", "b")}),
        y, provenance,
        frozenset((domain, int(value)) for value in provenance[identity]),
        partition_summary(domain, split, provenance, y),
    )


def tabpfn_values() -> dict:
    return json.loads((CONFIGS_DIR / "tabpfn_source_n2000.json").read_text())


class SamplingTests(unittest.TestCase):
    def test_exact_budget_class_counts_alignment_and_no_mutation(self):
        full = fixture()
        selected = select_training(full, 8, 42)
        self.assertEqual(len(selected.X), 8)
        self.assertEqual(selected.summary["class_counts"], {"0": 6, "1": 2})
        self.assertTrue(selected.identities <= full.identities)
        self.assertEqual(len(selected.identities), 8)
        np.testing.assert_array_equal(selected.X["value"], selected.provenance["row_index"] / 100)
        np.testing.assert_array_equal(selected.y, (selected.provenance["row_index"] % 4 == 0).astype(int))
        self.assertEqual(len(full.X), 20)
        self.assertEqual(selected.provenance["row_index"].tolist(), sorted(selected.provenance["row_index"]))

    def test_determinism_reordering_and_nhanes_identity(self):
        for domain in ("source", "kaggle", "nhanes"):
            full = fixture(domain)
            first = select_training(full, 9, 42)
            perm = np.random.default_rng(9).permutation(len(full.X))
            reordered = Partition(
                full.X.iloc[perm], full.y.iloc[perm], full.provenance.iloc[perm],
                full.identities, full.summary,
            )
            second = select_training(reordered, 9, 42)
            pd.testing.assert_frame_equal(first.X, second.X)
            pd.testing.assert_series_equal(first.y, second.y)
            pd.testing.assert_frame_equal(first.provenance, second.provenance)
            self.assertEqual(first.summary, second.summary)
            self.assertEqual(first.summary["identity_column"], "seqn" if domain == "nhanes" else "row_index")

    def test_invalid_budgets_and_split_roles(self):
        for budget in (0, 1, 21, -1, True, 2.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select_training(fixture(), budget, 42)
        with self.assertRaises(ValueError):
            select_training(fixture(), 5, None)
        for split in ("val", "test"):
            with self.assertRaisesRegex(ValueError, "train partition"):
                select_training(fixture(split=split), 5, 42)
        full = fixture()
        self.assertIs(select_training(full, None, None), full)
        self.assertEqual(len(select_training(full, 20, 42).X), 20)

    def test_rare_class_preserved(self):
        full = fixture()
        full.y[:] = 0
        full.y.iloc[0] = 1
        self.assertEqual(select_training(full, 2, 42).y.tolist().count(1), 1)
        full.y[:] = 1
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            select_training(full, 4, 42)


class ExtendedConfigTests(unittest.TestCase):
    def test_float32_research_configs_only_change_precision_and_run_identity(self):
        existing_outputs = {
            load_config(path).resolved_output_dir
            for path in CONFIGS_DIR.glob("*.json")
            if not path.stem.endswith("_batch1024")
        }
        batch_outputs = set()
        for domain in ("source", "kaggle"):
            original = load_config(CONFIGS_DIR / f"tabpfn_{domain}_n2000.json")
            new = load_config(CONFIGS_DIR / f"tabpfn_{domain}_n2000_float32.json")
            self.assertEqual(original.model_params["inference_precision"], "autocast")
            self.assertEqual(new.model_params["inference_precision"], "float32")
            self.assertNotEqual(original.experiment_name, new.experiment_name)
            self.assertNotEqual(original.resolved_output_dir, new.resolved_output_dir)
            old_values, new_values = original.to_dict(), new.to_dict()
            for key in ("experiment_name", "output_dir"):
                old_values.pop(key)
                new_values.pop(key)
            new_values["model_params"]["inference_precision"] = "autocast"
            self.assertEqual(old_values, new_values)

            batched = load_config(CONFIGS_DIR / f"tabpfn_{domain}_n2000_float32_batch1024.json")
            self.assertEqual(new.prediction_batch_size, 256)
            self.assertEqual(batched.prediction_batch_size, 1024)
            self.assertEqual(batched.experiment_name, new.experiment_name + "_batch1024")
            self.assertEqual(str(batched.resolved_output_dir), str(new.resolved_output_dir) + "_batch1024")
            self.assertNotIn(batched.resolved_output_dir, existing_outputs | batch_outputs)
            batch_outputs.add(batched.resolved_output_dir)
            expected = new.to_dict()
            expected.update(
                prediction_batch_size=1024,
                experiment_name=new.experiment_name + "_batch1024",
                output_dir=str(new.resolved_output_dir) + "_batch1024",
            )
            self.assertEqual(batched.to_dict(), expected)

    def test_original_configs_and_four_new_configs(self):
        for name in ("catboost_source.json", "catboost_kaggle.json"):
            config = load_config(CONFIGS_DIR / name)
            self.assertIsNone(config.train_sample_size)
            self.assertIsNone(config.sampling_seed)
            self.assertIsNone(config.prediction_batch_size)
            self.assertEqual(config.early_stopping_rounds, 100)
        outputs = set()
        for domain in ("source", "kaggle"):
            for model in ("catboost", "tabpfn"):
                config = load_config(CONFIGS_DIR / f"{model}_{domain}_n2000.json")
                self.assertEqual(config.train_sample_size, 2000)
                self.assertEqual(config.sampling_seed, 42)
                self.assertEqual(config.prediction_batch_size, 256)
                self.assertEqual(config.train_domain, domain)
                self.assertEqual(config.validation_domain, domain)
                self.assertEqual(config.test_domains, ("source", "kaggle", "nhanes"))
                self.assertEqual(config.threshold, 0.5)
                outputs.add(config.resolved_output_dir)
        self.assertEqual(len(outputs), 4)

    def test_model_specific_validation(self):
        for updates in (
            {"early_stopping_rounds": 100}, {"train_sample_size": True},
            {"sampling_seed": None}, {"sampling_seed": True},
            {"prediction_batch_size": 0}, {"prediction_batch_size": False},
            {"feature_set": "full_24"}, {"output_dir": "../escape"}, {"unknown": 1},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**tabpfn_values(), **updates})
        for params in (
            {"iterations": 10}, {"random_state": 42}, {"ignore_pretraining_limits": True},
            {"categorical_features_indices": [0]}, {"n_estimators": "4"},
            {"n_estimators": True}, {"device": "auto"}, {"device": "cuda"},
            {"device": "cuda:-1"}, {"fit_mode": "batched"}, {"fit_mode": []},
            {"inference_precision": "float16"}, {"memory_saving_mode": 1},
            {"model_path": "auto"}, {"model_path": None},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**tabpfn_values(), "model_params": params})
        values = tabpfn_values()
        self.assertIsNone(ExperimentConfig.from_dict(values).early_stopping_rounds)
        values["early_stopping_rounds"] = None
        self.assertIsNone(ExperimentConfig.from_dict(values).early_stopping_rounds)


@dataclass
class FakeInferenceConfig:
    FINGERPRINT_FEATURE: bool = True


class FakeEstimator:
    instances = []

    def __init__(self, **params):
        self.params = params
        self.classes_ = np.array([1, 0])
        self.n_estimators_ = params["n_estimators"]
        self.devices_ = ["cpu"]
        self.categorical_features_indices_ = params["categorical_features_indices"]
        self.inference_config_ = FakeInferenceConfig()
        self.use_autocast_ = False
        self.forced_inference_dtype_ = None
        self.calls = []
        self.fit_count = 0
        self.instances.append(self)

    def fit(self, X, y):
        self.fit_count += 1
        self.X = X.copy()
        self.y = y.copy()
        self.categories = sorted(X["category"].unique()) if "category" in X else []
        return self

    def get_params(self, deep=False):
        return dict(self.params)

    def predict_proba(self, X):
        self.calls.append(X.copy())
        positive = X["value"].to_numpy(dtype=float)
        return np.column_stack([positive, 1 - positive])

    def save_fit_state(self, path):
        Path(path).write_text(json.dumps({
            "params": self.params, "X": self.X.to_dict(orient="list"), "y": self.y.tolist(),
        }))

    @classmethod
    def load_from_fit_state(cls, path, *, device):
        saved = json.loads(Path(path).read_text())
        model = cls(**{**saved["params"], "device": device})
        model.fit(pd.DataFrame(saved["X"]), np.array(saved["y"]))
        return model


def fake_torch(cuda_available=False):
    def device(value):
        return SimpleNamespace(
            type=value.split(":")[0],
            index=int(value.split(":")[1]) if ":" in value else None,
            __str__=lambda: value,
        )

    return SimpleNamespace(
        device=device, float32="fake_float32",
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available, device_count=lambda: 1,
            synchronize=Mock(), reset_peak_memory_stats=Mock(),
            max_memory_allocated=lambda _: 123, max_memory_reserved=lambda _: 456,
            get_device_name=lambda _: "Fake CUDA",
        ),
    )


class TabPFNTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.checkpoint = self.root / CHECKPOINT_FILENAME
        self.checkpoint.write_bytes(b"fake checkpoint; no model weights")
        self.torch = fake_torch()
        self.stack.enter_context(patch.object(tabpfn_module, "_runtime", return_value=(self.torch, FakeEstimator)))
        FakeEstimator.instances = []

    def adapter(self, **params):
        return TabPFNAdapter(
            ("value", "category"), ("category",), "diagnosed_diabetes", seed=42,
            model_params={"model_path": str(self.checkpoint), "device": "cpu", **params},
        )

    def test_fit_context_only_chunk_order_partial_chunk_and_positive_class(self):
        adapter = self.adapter()
        train = fixture()
        original = train.X.copy()
        # Deliberately unusable validation objects prove they are never inspected.
        adapter.fit(train.X, train.y, None, None)
        queries = train.X.iloc[:7].copy()
        queries["category"] = "unseen"
        expected = adapter.predict_proba(queries)
        model = adapter._model
        model.calls.clear()
        actual = adapter.predict_proba_batched(queries, 3)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual([len(call) for call in model.calls], [3, 3, 1])
        self.assertEqual(model.fit_count, 1)
        pd.testing.assert_frame_equal(pd.concat(model.calls), queries)
        pd.testing.assert_frame_equal(train.X, original)
        self.assertEqual(model.categories, ["a", "b"])
        self.assertFalse(model.params["ignore_pretraining_limits"])
        self.assertEqual(model.params["categorical_features_indices"], [1])
        self.assertEqual(model.params["random_state"], 42)
        metadata = adapter.metadata()
        self.assertFalse(metadata["validation_used"])
        self.assertEqual(metadata["effective_ensemble_count"], 4)
        self.assertIn("unknown_value=-1", metadata["preprocessing"]["unseen_categories"])
        json.dumps(metadata, allow_nan=False)

    def test_schema_input_and_probability_validation(self):
        adapter = self.adapter()
        train = fixture()
        with self.assertRaises(RuntimeError):
            adapter.predict_proba(train.X)
        for bad in (
            train.X[["category", "value"]], train.X.assign(value=np.nan),
            train.X.assign(category=np.inf), train.X.assign(id=1),
        ):
            with self.assertRaises(ValueError):
                self.adapter().fit(bad, train.y, None, None)
        for labels in (train.y.iloc[:-1], pd.Series([0] * 20)):
            with self.assertRaises(ValueError):
                self.adapter().fit(train.X, labels, None, None)
        adapter.fit(train.X, train.y, None, None)
        for raw in (
            np.ones(20), np.ones((20, 2)), np.full((20, 2), np.nan),
            np.tile([-0.1, 1.1], (20, 1)),
        ):
            with patch.object(adapter._model, "predict_proba", return_value=raw):
                with self.assertRaises(ValueError):
                    adapter.predict_proba(train.X)

    def test_native_persistence_sidecar_and_dependency_check(self):
        adapter = self.adapter()
        train = fixture()
        adapter.fit(train.X, train.y, None, None)
        artifact = self.root / adapter.artifact_name
        adapter.save(artifact)
        self.assertTrue(artifact.is_file())
        self.assertTrue(artifact.with_suffix(".adapter.json").is_file())
        restored = TabPFNAdapter.load(artifact, device="cpu")
        np.testing.assert_array_equal(adapter.predict_proba(train.X), restored.predict_proba(train.X))
        self.checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint identity"):
            TabPFNAdapter.load(artifact, device="cpu")

    def test_missing_checkpoint_device_version_and_ensemble_fail_explicitly(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "never downloads"):
            self.adapter()
        self.checkpoint.write_bytes(b"fixture")
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            self.adapter(device="cuda:0")
        adapter = self.adapter()
        train = fixture()
        original_fit = FakeEstimator.fit

        def change_ensemble(model, X, y):
            original_fit(model, X, y)
            model.n_estimators_ = 1

        with patch.object(FakeEstimator, "fit", change_ensemble):
            with self.assertRaisesRegex(RuntimeError, "ensemble count"):
                adapter.fit(train.X, train.y, None, None)

    def test_cuda_measurement_hooks(self):
        self.torch.cuda.is_available = lambda: True
        adapter = self.adapter(device="cuda:0")
        adapter.reset_peak_memory()
        adapter.synchronize()
        self.torch.cuda.reset_peak_memory_stats.assert_called_once()
        self.assertEqual(adapter.resource_metadata()["peak_gpu_memory"]["allocated_bytes"], 123)

    def run_fake_smoke(self, *, unchunked_oom=False, chunked_oom=False):
        def predict(model, X):
            if (unchunked_oom and len(X) == 7) or chunked_oom:
                raise RuntimeError("CUDA out of memory")
            return np.column_stack([X["value"], 1 - X["value"]])

        values = tabpfn_values()
        values["model_params"] = {
            **values["model_params"], "model_path": str(self.checkpoint), "device": "cpu",
        }
        config_path = self.root / "smoke_config.json"
        config_path.write_text(json.dumps(values))
        full, val = fixture(), fixture(split="val")
        val.identities = frozenset((domain, identity + 1000) for domain, identity in val.identities)
        with (
            patch.multiple(config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results"),
            patch.object(smoke_tabpfn.data, "load_feature_schema", return_value=ExtendedRunnerTests.schema),
            patch.object(smoke_tabpfn, "load_partition", side_effect=[full, val]) as load,
            patch.object(sys, "argv", [
                "smoke_tabpfn", "--config", str(config_path),
                "--output-dir", "results/smoke/check", "--sample-size", "8",
                "--query-size", "7", "--batch-size", "3",
                "--inference-precision", "float32",
            ]),
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(
                FakeEstimator, "predict_proba",
                autospec=True,
                side_effect=predict,
            ),
        ):
            if chunked_oom:
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    smoke_tabpfn.main()
            else:
                smoke_tabpfn.main()
        self.assertEqual([call.args[1] for call in load.call_args_list], ["train", "val"])
        return json.loads((self.root / "results/smoke/check/smoke.json").read_text())

    def test_optional_smoke_uses_only_training_and_validation(self):
        report = self.run_fake_smoke()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["chunking_max_absolute_difference"], 0)
        self.assertEqual(report["reload_max_absolute_difference"], 0)
        self.assertEqual(report["config"]["model_params"]["inference_precision"], "float32")
        self.assertEqual(report["config"]["prediction_batch_size"], 3)
        self.assertEqual(report["chunking_threshold_disagreements"], 0)
        self.assertEqual(report["chunking_mean_absolute_difference"], 0)
        self.assertEqual(tabpfn_values()["model_params"]["inference_precision"], "autocast")
        self.assertEqual(report["intended_chunked_path_status"], "completed")
        self.assertTrue(report["probability_validation"]["finite"])
        self.assertEqual(list(report["operations"]), [
            "context_preparation", "chunked_prediction", "save_fitted_state",
            "reload_fitted_state", "reloaded_chunked_prediction", "unchunked_comparison",
        ])
        self.assertTrue(all(op["duration_seconds"] >= 0 for op in report["operations"].values()))

    def test_smoke_separates_unchunked_oom_from_intended_path(self):
        report = self.run_fake_smoke(unchunked_oom=True)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["intended_chunked_path_status"], "completed")
        self.assertEqual(report["reload_agreement"], "passed")
        self.assertEqual(report["unchunked_comparison_status"], "out_of_memory")
        self.assertEqual(report["operations"]["unchunked_comparison"]["status"], "failed")

    def test_smoke_stops_on_intended_chunked_oom(self):
        report = self.run_fake_smoke(chunked_oom=True)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["operations"]["chunked_prediction"]["status"], "failed")
        self.assertNotIn("unchunked_comparison", report["operations"])
        self.assertNotIn("reload_fitted_state", report["operations"])


class ExtendedRunnerTests(unittest.TestCase):
    schema = DatasetSchema("diagnosed_diabetes", ("value", "category"), ("category",), ("value",), ())

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.multiple(config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results"))
        self.stack.enter_context(patch.object(runner.data, "load_feature_schema", return_value=self.schema))
        self.stack.enter_context(patch.object(runner, "prepared_fingerprints", return_value={}))
        self.stack.enter_context(patch.object(runner, "load_partition", side_effect=self.load))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.adapter = Mock()
        self.adapter.artifact_name = "model.tabpfn_fit"
        self.adapter.save.side_effect = lambda path: path.write_bytes(b"fake fit state")
        self.adapter.metadata.return_value = {"preprocessing": {}, "compute_settings": {}, "validation_used": False}
        self.adapter.resource_metadata.return_value = {}
        self.adapter.predict_proba.return_value = np.array([0.2, 0.8] * 10)
        self.adapter.predict_proba_batched.return_value = np.array([0.2, 0.8] * 10)
        self.factory = self.stack.enter_context(patch.object(runner, "create_adapter", return_value=self.adapter))

    def load(self, domain, split, *args):
        partition = fixture(domain, split)
        if split != "train":
            offset = 1000 if split == "val" else 2000
            partition.provenance["row_index"] += offset
            if domain == "nhanes":
                partition.provenance["seqn"] += offset
            column = partition.summary["identity_column"]
            partition.identities = frozenset((domain, int(value)) for value in partition.provenance[column])
            partition.summary = partition_summary(domain, split, partition.provenance, partition.y)
        return partition

    def config(self, model, seed=42):
        values = tabpfn_values() if model == "tabpfn" else json.loads((CONFIGS_DIR / "catboost_source_n2000.json").read_text())
        return ExperimentConfig.from_dict({
            **values, "train_sample_size": 8, "seed": seed,
            "output_dir": f"results/{model}_{seed}",
        })

    def test_matched_rows_full_test_counts_artifacts_and_policy(self):
        contexts = []
        for model, seed in (("tabpfn", 42), ("catboost", 7)):
            config = self.config(model, seed)
            self.adapter.artifact_name = "model.tabpfn_fit" if model == "tabpfn" else "model.cbm"
            output = runner.run(config)
            metadata = json.loads((output / "metadata.json").read_text())
            contexts.append(self.adapter.fit.call_args.args[:2])
            sample = pd.read_csv(output / "training_sample.csv.gz")
            self.assertEqual(len(sample), 8)
            self.assertEqual(sample["sample_order"].tolist(), list(range(8)))
            self.assertEqual(set(sample["split"]), {"train"})
            self.assertEqual(metadata["training_partition_full"]["rows"], 20)
            self.assertEqual(metadata["training_context"]["rows"], 8)
            self.assertEqual(metadata["sampling"]["requested_budget"], 8)
            for domain in config.test_domains:
                self.assertEqual(metadata["splits"][f"{domain}/test"]["rows"], 20)
            self.assertTrue((output / self.adapter.artifact_name).exists())
            if model == "tabpfn":
                self.assertIn("validation unused", metadata["selection_policy"])
                self.assertFalse((output / "model.cbm").exists())
            else:
                self.assertIn("val exclusively for early stopping", metadata["selection_policy"])
        pd.testing.assert_frame_equal(contexts[0][0], contexts[1][0])
        pd.testing.assert_series_equal(contexts[0][1], contexts[1][1])
        self.assertEqual(self.adapter.fit.call_count, 2)
        self.assertEqual(self.adapter.predict_proba_batched.call_count, 6)
        self.assertEqual(self.adapter.synchronize.call_count, 16)

    def test_full_training_overlap_not_just_selected_rows(self):
        config = self.config("tabpfn")
        selected = select_training(fixture(), 8, 42)
        excluded = list(fixture().identities - selected.identities)[0][1]
        original = self.load

        def overlap(domain, split, *args):
            part = original(domain, split, *args)
            if domain == "source" and split == "test":
                part.identities = frozenset({("source", excluded)})
            return part

        with patch.object(runner, "load_partition", side_effect=overlap):
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.run(config)
        self.adapter.predict_proba_batched.assert_not_called()
        metadata = json.loads((config.resolved_output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")

    def test_lazy_prediction_oom_failure_context_and_no_retry(self):
        config = self.config("tabpfn")
        self.adapter.predict_proba_batched.side_effect = RuntimeError("CUDA out of memory")
        with self.assertLogs(runner.LOGGER, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                runner.run(config)
        metadata = json.loads((config.resolved_output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        context = metadata["error"]["out_of_memory_context"]
        self.assertEqual(context["train_sample_size"], 8)
        self.assertEqual(context["prediction_batch_size"], 256)
        self.assertEqual(context["n_estimators"], 4)
        self.assertEqual(context["device"], "cuda:0")
        self.adapter.fit.assert_called_once()
        self.adapter.predict_proba_batched.assert_called_once()
        with self.assertRaises(FileExistsError):
            runner.run(config)


class OptionalImportTests(unittest.TestCase):
    def test_optional_imports_blocked_and_config_loading_still_works(self):
        code = """
import sys
import importlib.abc
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tabpfn', 'catboost'}:
            raise AssertionError('Optional import attempted: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from src.experiments.config import load_config
from src.paths import CONFIGS_DIR
import src.experiments.runner
import src.experiments.metrics
import src.experiments.sampling
import src.models.tabpfn
import src.experiments.smoke_tabpfn
for path in CONFIGS_DIR.glob('*.json'):
    load_config(path)
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_version_mismatch_rejected_before_optional_import(self):
        with patch.object(tabpfn_module, "version", return_value="0.0"):
            with self.assertRaisesRegex(RuntimeError, "supports TabPFN 9.2.0"):
                tabpfn_module._runtime()


if __name__ == "__main__":
    unittest.main()
