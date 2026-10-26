from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.data import DatasetSchema
from src.experiments import config as config_module, runner
from src.experiments.config import ExperimentConfig, load_config, validate_model_params
from src.experiments.results import create_run_directory
from src.experiments.sampling import Partition, partition_summary
from src.models import tabicl
from src.models.tabicl import CHECKPOINT_FILENAME, TabICLAdapter
from src.paths import CONFIGS_DIR, RESULTS_DIR


SCHEMA = DatasetSchema(
    target="diagnosed_diabetes", features=("value", "category"),
    categorical_features=("category",), numeric_features=("value",), source_only_excluded=(),
)


def partition(domain="source", split="train", count=20, offset=0):
    row = np.arange(count, dtype=np.int64)
    provenance = pd.DataFrame({"row_index": row + offset})
    if domain == "nhanes":
        provenance["seqn"] = row + offset + 1000
    labels = pd.Series(row % 2, name=SCHEMA.target)
    return Partition(
        pd.DataFrame({"value": (row % 81) / 100, "category": row % 3}), labels, provenance,
        frozenset((domain, int(i)) for i in provenance.row_index),
        partition_summary(domain, split, provenance, labels),
    )


def values(checkpoint, domain="source"):
    return {
        "experiment_name": f"tabicl_v2_{domain}_none_fp32_smoke",
        "model": "tabicl", "train_domain": domain, "validation_domain": domain,
        "test_domains": ["source", "kaggle", "nhanes"], "feature_set": "strict_common_16",
        "seed": 42, "sampling_seed": 42, "train_sample_size": 8,
        "threshold": 0.5, "calibration_bins": 10, "early_stopping_rounds": None,
        "prediction_batch_size": 3, "output_dir": f"results/smoke/{domain}_none_fp32",
        "model_params": {"model_path": str(checkpoint), "norm_methods": ["none"], "device": "cpu"},
    }


class FakeDevice:
    def __init__(self, name):
        self.name = name
        self.type = name.split(":")[0]
        self.index = int(name.split(":")[1]) if ":" in name else None

    def __str__(self):
        return self.name


def fake_torch():
    state = {"threads": 6}
    return SimpleNamespace(
        device=FakeDevice, get_num_threads=lambda: state["threads"],
        set_num_threads=lambda value: state.update(threads=value),
        get_autocast_dtype=lambda _: "torch.float16",
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _: "Fake GPU",
            synchronize=Mock(), reset_peak_memory_stats=Mock(), empty_cache=Mock(),
            max_memory_allocated=lambda _: 123, max_memory_reserved=lambda _: 456,
            mem_get_info=lambda _: (2**30, 2**33),
        ),
    )


class FakeEstimator:
    instances = []
    load_calls = 0

    def __init__(self, **params):
        self.params = params
        self.calls = []
        self.fit_count = 0
        self.instances.append(self)

    def fit(self, X, y):
        self.fit_count += 1
        self.X, self.y = X.copy(deep=True), y.copy()
        self.classes_ = np.array([1, 0])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns)
        self.model_kv_cache_ = None
        self.model_ = SimpleNamespace(max_classes=10)
        self.model_config_ = {"max_classes": 10}
        self.ensemble_generator_ = SimpleNamespace(
            ensemble_configs_={"none": [([0, 1], [1, 0])] * self.params["n_estimators"]},
            preprocessors_={"none": object()},
            unique_filter_=SimpleNamespace(features_to_keep_=np.array([True, True])),
        )
        self.X_encoder_ = SimpleNamespace(tfm_=SimpleNamespace(
            transformers_=[("categorical", object(), [1]), ("continuous", object(), [0])],
        ))
        self.inference_config_ = SimpleNamespace(**{
            key: {"use_amp": self.params["use_amp"], "device": self.params["device"]}
            for key in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
        })
        print("Native fit diagnostics")
        return self

    def get_params(self, deep=False):
        return self.params.copy()

    def predict_proba(self, X):
        self.calls.append(X.copy(deep=True))
        p = X["value"].to_numpy(dtype=float)
        return np.column_stack([p, 1 - p])

    def save(self, path, **flags):
        if flags != tabicl.SAVE_FLAGS:
            raise AssertionError(flags)
        Path(path).write_text(json.dumps({
            "params": self.params, "X": self.X.to_dict(orient="list"), "y": self.y.tolist(),
        }), encoding="utf-8")

    @classmethod
    def load(cls, path, device=None):
        cls.load_calls += 1
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(**saved["params"])
        result.fit(pd.DataFrame(saved["X"]), np.array(saved["y"]))
        result.fit_count = 0
        return result


class TabICLTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.checkpoint = self.root / CHECKPOINT_FILENAME
        payload = b"mock checkpoint: never real model weights"
        self.checkpoint.write_bytes(payload)
        self.stack.enter_context(patch.object(tabicl, "CHECKPOINT_SIZE", len(payload)))
        self.stack.enter_context(patch.object(tabicl, "CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest()))
        self.torch = fake_torch()
        self.stack.enter_context(patch.object(tabicl, "_runtime", return_value=(self.torch, FakeEstimator)))
        FakeEstimator.instances = []
        FakeEstimator.load_calls = 0

    def adapter(self, **params):
        return TabICLAdapter(
            SCHEMA.features, SCHEMA.categorical_features, SCHEMA.target, seed=42,
            model_params={**values(self.checkpoint)["model_params"], **params},
        )

    def config(self, domain="source"):
        return ExperimentConfig.from_dict(values(self.checkpoint, domain))


class ConfigTests(TabICLTestCase):
    def test_research_configs_preserve_smoke_protocol_and_isolate_outputs(self):
        configs = []
        for domain in ("source", "kaggle"):
            research = load_config(CONFIGS_DIR / f"tabicl_v2_{domain}_n2000_none_fp32.json")
            smoke = load_config(CONFIGS_DIR / "smoke" / f"tabicl_v2_{domain}_none_fp32.json")
            config = replace(
                research, experiment_name=smoke.experiment_name, output_dir=smoke.output_dir,
            )
            self.assertEqual(config.to_dict(), smoke.to_dict())
            self.assertTrue(research.resolved_output_dir.is_relative_to(RESULTS_DIR / "predictive"))
            self.assertNotEqual(research.resolved_output_dir, smoke.resolved_output_dir)
            configs.append(research)
        changed = {
            key for key, value in configs[0].to_dict().items()
            if value != configs[1].to_dict()[key]
        }
        self.assertEqual(changed, {"train_domain", "validation_domain", "experiment_name", "output_dir"})
        with patch.multiple(config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results"):
            paths = [create_run_directory(config) for config in configs]
            self.assertNotEqual(*paths)
            for config, path in zip(configs, paths):
                marker = path / "preserved.txt"
                marker.write_text("existing run")
                with self.assertRaises(FileExistsError):
                    create_run_directory(config)
                self.assertEqual(marker.read_text(), "existing run")

    def test_explicit_experimental_profile_and_validation(self):
        config = self.config()
        self.assertEqual(config.model_params["norm_methods"], ["none"])
        for updates in (
            {"early_stopping_rounds": 10}, {"prediction_batch_size": True}, {"seed": True},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**values(self.checkpoint), **updates})
        params = config.model_params
        for bad in (
            {}, {"model_path": str(self.checkpoint)}, {"norm_methods": ["none"]},
            {**params, "norm_methods": ["none", "power"]},
            {**params, "norm_methods": "none"}, {**params, "model_path": "auto"},
            {**params, "allow_auto_download": False}, {**params, "random_state": 42},
            {**params, "inference_config": {}}, {**params, "use_amp": "auto"},
            {**params, "use_fa3": 0}, {**params, "kv_cache": 0},
            {**params, "n_estimators": True}, {**params, "batch_size": None},
            {**params, "n_jobs": 0}, {**params, "device": "cuda"},
            {**params, "offload_mode": "auto"},
        ):
            with self.subTest(params=bad), self.assertRaises(ValueError):
                validate_model_params(bad, "tabicl")
        validate_model_params({**params, "use_amp": True}, "tabicl")
        self.assertEqual(self.adapter(use_amp=True).profile, "tabicl_v2_none_amp_uncached")


class AdapterTests(TabICLTestCase):
    def test_train_only_categorical_copies_chunks_and_metadata(self):
        adapter = self.adapter()
        self.assertEqual(FakeEstimator.instances, [])
        full = partition()
        original, labels = full.X.copy(deep=True), full.y.copy(deep=True)
        with self.assertRaises(RuntimeError):
            adapter.predict_proba(full.X)
        adapter.set_diagnostic_path(self.root / "native.log")
        adapter.fit(full.X, full.y, None, None)
        model = adapter._model
        self.assertEqual(model.fit_count, 1)
        self.assertEqual(model.X.category.tolist(), original.category.astype(str).tolist())
        self.assertFalse(model.params["allow_auto_download"])
        self.assertFalse(model.params["support_many_classes"])
        self.assertEqual(model.params["random_state"], 42)
        queries = full.X.iloc[:7].copy()
        queries["category"] = "unseen"
        before = queries.copy(deep=True)
        result = adapter.predict_proba_batched(queries, 3)
        np.testing.assert_array_equal(result, queries.value)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual([len(X) for X in model.calls], [3, 3, 1])
        pd.testing.assert_frame_equal(pd.concat(model.calls), adapter._inputs(queries))
        pd.testing.assert_frame_equal(queries, before)
        pd.testing.assert_frame_equal(full.X, original)
        pd.testing.assert_series_equal(full.y, labels)
        self.assertEqual(model.fit_count, 1)
        metadata = adapter.metadata()
        self.assertFalse(metadata["validation_used"])
        self.assertEqual(metadata["experimental_profile"], "tabicl_v2_none_fp32_uncached")
        self.assertEqual(metadata["preprocessing"]["normalization_views"], ["none"])
        self.assertEqual(metadata["effective_ensemble_count"], 8)
        self.assertEqual(metadata["native_diagnostics"]["recovery_detection"], "no_recovery_detected")
        self.assertIn("not proof", metadata["native_diagnostics"]["limitations"])
        self.assertIn("docs/tabicl_stage3_results.md", metadata["query_policy"])
        self.assertIn("not certification of this run", metadata["query_policy"])
        json.dumps(metadata, allow_nan=False)
        self.assertIn("Native fit diagnostics", (self.root / "native.log").read_text())
        with self.assertRaises(RuntimeError):
            adapter.fit(full.X, full.y, None, None)

    def test_inputs_labels_probability_and_fixed_sum_criterion(self):
        full = partition()
        for bad in (
            full.X[["category", "value"]], full.X.assign(id=1), full.X.assign(value=np.inf),
            full.X.assign(value=np.nan), full.X.assign(category=np.inf),
            full.X.assign(category=None),
        ):
            with self.subTest(columns=list(bad)), self.assertRaises(ValueError):
                self.adapter().fit(bad, full.y, None, None)
        for labels in (full.y.iloc[:-1], pd.Series([0] * 20), pd.Series([2] * 20)):
            with self.assertRaises(ValueError):
                self.adapter().fit(full.X, labels, None, None)
        adapter = self.adapter()
        adapter.fit(full.X, full.y, None, None)
        for bad in (
            np.ones(20), np.ones((20, 2)), np.full((20, 2), np.nan),
            np.tile([-0.1, 1.1], (20, 1)),
            np.tile([0.5, 0.5 + 2 * tabicl.PROBABILITY_SUM_ATOL], (20, 1)),
        ):
            with patch.object(adapter._model, "predict_proba", return_value=bad), self.assertRaises(ValueError):
                adapter.predict_proba(full.X)
        with patch.object(adapter._model, "predict_proba", return_value=np.tile([0.5, 0.5], (20, 1))):
            np.testing.assert_array_equal(adapter.predict(full.X), np.ones(20))

    def test_checkpoint_and_device_guards_before_estimator_construction(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "never downloads"):
            self.adapter()
        self.checkpoint.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.adapter()
        with self.assertRaisesRegex(ValueError, "named"):
            self.adapter(model_path=str(self.root / "another.ckpt"))
        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            self.adapter(device="cuda:1")
        self.assertEqual(FakeEstimator.instances, [])

    def test_checkpoint_rechecked_and_effective_profile_guarded(self):
        full = partition()
        adapter = self.adapter()
        self.checkpoint.write_bytes(b"changed after construction")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            adapter.fit(full.X, full.y, None, None)
        self.assertEqual(FakeEstimator.instances, [])

    def test_effective_ensemble_and_constructor_changes_fail(self):
        full = partition()
        original = FakeEstimator.fit
        for mutation in (
            lambda model: model.ensemble_generator_.ensemble_configs_["none"].pop(),
            lambda model: model.params.update(allow_auto_download=True),
            lambda model: setattr(model, "model_kv_cache_", object()),
        ):
            def fit(model, X, y):
                original(model, X, y)
                mutation(model)
                return model
            with patch.object(FakeEstimator, "fit", fit), self.assertRaises(RuntimeError):
                self.adapter().fit(full.X, full.y, None, None)

    def test_native_diagnostics_on_failure_and_thread_restoration(self):
        full = partition()
        adapter = self.adapter()
        adapter.set_diagnostic_path(self.root / "native.log")
        adapter.fit(full.X, full.y, None, None)

        def fail(X):
            self.torch.set_num_threads(1)
            print("OOM with batch_size=8, reducing to 4")
            print("Offload decision: CPU (swap)", file=sys.stderr)
            warnings.warn("native warning", UserWarning)
            raise RuntimeError("CUDA out of memory")

        with patch.object(adapter._model, "predict_proba", side_effect=fail):
            with self.assertLogs(tabicl.LOGGER, "WARNING"), self.assertRaisesRegex(RuntimeError, "out of memory"):
                adapter.predict_proba(full.X)
        self.assertEqual(self.torch.get_num_threads(), 6)
        diagnostics = adapter.diagnostic_metadata()
        self.assertEqual(diagnostics["recovery_detection"], "recovery_detected")
        self.assertTrue(diagnostics["swap_fallback_detected"])
        text = (self.root / "native.log").read_text()
        self.assertIn("native warning", text)
        self.assertIn("reducing to 4", text)

    def test_native_embedded_persistence_does_not_require_checkpoint(self):
        full = partition()
        adapter = self.adapter()
        adapter.fit(full.X, full.y, None, None)
        artifact = self.root / adapter.artifact_name
        adapter.save(artifact)
        saved = json.loads(artifact.with_suffix(".adapter.json").read_text())
        self.assertEqual(saved["save_flags"], tabicl.SAVE_FLAGS)
        self.checkpoint.unlink()
        restored = TabICLAdapter.load(artifact, device="cpu", diagnostic_path=self.root / "reload.log")
        np.testing.assert_array_equal(adapter.predict_proba(full.X), restored.predict_proba(full.X))
        self.assertEqual(restored._model.fit_count, 0)
        self.assertEqual(restored.metadata()["artifact_sha256"], tabicl._sha256(artifact))
        self.assertEqual(FakeEstimator.load_calls, 1)
        for updates in (
            {"saved_device": "cuda:0"}, {"versions": {}}, {"save_flags": {}},
            {"artifact_sha256": "bad"}, {"profile": "default_tabicl"},
            {"features": ["id", "category"]},
        ):
            artifact.with_suffix(".adapter.json").write_text(json.dumps({**saved, **updates}))
            with self.subTest(updates=updates), self.assertRaises((ValueError, RuntimeError)):
                TabICLAdapter.load(artifact, device="cpu")
        self.assertEqual(FakeEstimator.load_calls, 1)
        artifact.with_suffix(".adapter.json").write_text(json.dumps(saved))
        artifact.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            TabICLAdapter.load(artifact, device="cpu")

    def test_reporting_and_thread_errors_do_not_hide_native_failure(self):
        full = partition()
        adapter = self.adapter()
        adapter.fit(full.X, full.y, None, None)
        with (
            patch.object(adapter._model, "predict_proba", side_effect=RuntimeError("original native failure")),
            patch.object(self.torch, "set_num_threads", side_effect=RuntimeError("thread restoration failed")),
            self.assertLogs(tabicl.LOGGER, "ERROR"),
            self.assertRaisesRegex(RuntimeError, "original native failure") as caught,
        ):
            adapter.predict_proba(full.X)
        self.assertIn("thread restoration failed", caught.exception.__notes__[0])
        with self.assertRaisesRegex(TypeError, "metadata value"):
            tabicl._json_value(object())

    def test_cuda_hooks(self):
        adapter = self.adapter(device="cuda:0")
        self.assertIsNone(adapter.resource_metadata()["peak_gpu_memory"])
        adapter.reset_peak_memory()
        adapter.synchronize()
        self.assertEqual(adapter.resource_metadata()["peak_gpu_memory"]["allocated_bytes"], 123)
        self.torch.cuda.reset_peak_memory_stats.assert_called_once()

    def test_restore_checks_native_schema_against_sidecar(self):
        full = partition()
        adapter = self.adapter()
        adapter.fit(full.X, full.y, None, None)
        artifact = self.root / adapter.artifact_name
        adapter.save(artifact)
        path = artifact.with_suffix(".adapter.json")
        saved = json.loads(path.read_text())
        for updates in (
            {"features": ["category", "value"]}, {"categorical_features": []},
        ):
            path.write_text(json.dumps({**saved, **updates}))
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "schema"):
                TabICLAdapter.load(artifact, device="cpu")


class RunnerTests(TabICLTestCase):
    def test_factory_and_runner_wiring_without_native_runtime(self):
        self.stack.enter_context(patch.multiple(
            config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results",
        ))
        for domain in ("source", "kaggle"):
            config = self.config(domain)
            train, val = partition(domain), partition(domain, "val", offset=100)
            test = partition("nhanes", "test", count=7, offset=200)
            with (
                patch.object(runner.data, "load_feature_schema", return_value=SCHEMA),
                patch.object(runner, "prepared_fingerprints", return_value={"mock": "hash"}),
                patch.object(runner, "load_partition", side_effect=[train, val, test]) as load,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                output = runner.run(replace(config, test_domains=("nhanes",)))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertFalse(metadata["model"]["validation_used"])
            self.assertEqual(metadata["environment"]["packages"]["tabicl"], "2.2.0")
            self.assertEqual(metadata["model"]["compute_settings"], metadata["compute_settings"])
            self.assertIn("validation unused", metadata["selection_policy"])
            self.assertEqual([call.args[1] for call in load.call_args_list], ["train", "val", "test"])
            self.assertTrue((output / "native_diagnostics.log").is_file())
            self.assertEqual(len(metadata["model"]["native_diagnostics"]["operations"]), 5)
            with self.assertRaises(FileExistsError):
                runner.run(replace(config, test_domains=("nhanes",)))

    def test_failed_fit_persists_native_diagnostics_and_oom_context(self):
        self.stack.enter_context(patch.multiple(
            config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results",
        ))

        def fail(model, X, y):
            print("native fit about to fail")
            raise RuntimeError("CUDA out of memory")

        with (
            patch.object(runner.data, "load_feature_schema", return_value=SCHEMA),
            patch.object(runner, "prepared_fingerprints", return_value={}),
            patch.object(runner, "load_partition", side_effect=[partition(), partition(split="val", offset=100)]),
            patch.object(FakeEstimator, "fit", fail),
            self.assertLogs(runner.LOGGER, "ERROR"),
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            runner.run(self.config())
        output = self.config().resolved_output_dir
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["out_of_memory_context"]["n_estimators"], 8)
        self.assertIn("native fit about to fail", (output / "native_diagnostics.log").read_text())


class OptionalImportTests(unittest.TestCase):
    def test_imports_and_cli_help_without_optional_packages(self):
        code = """
import importlib.abc, sys, runpy
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'tabicl', 'tabpfn', 'catboost'}:
            raise AssertionError('Optional package imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import src.models.tabicl
import src.experiments.runner
sys.argv = ['runner', '--help']
try:
    runpy.run_module('src.experiments.runner', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsupported_version_fails_before_import(self):
        with patch.object(tabicl, "version", return_value="0.0"), self.assertRaisesRegex(RuntimeError, "2.2.0"):
            tabicl._runtime()


if __name__ == "__main__":
    unittest.main()
