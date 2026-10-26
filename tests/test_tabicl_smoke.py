from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.experiments import config as config_module, smoke_tabicl as smoke
from src.experiments.config import load_config
from src.experiments.sampling import select_training
from src.paths import CONFIGS_DIR
from test_tabicl_adapter import FakeEstimator, SCHEMA, TabICLTestCase, partition


class SmokeTests(TabICLTestCase):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.multiple(
            config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.root / "results",
        ))
        self.stack.enter_context(patch.object(smoke, "RESULTS_DIR", self.root / "results"))
        self.stack.enter_context(patch.object(smoke, "_runtime", return_value=(self.torch, FakeEstimator)))
        self.stack.enter_context(patch.object(smoke.data, "load_feature_schema", return_value=SCHEMA))
        self.stack.enter_context(patch.dict(os.environ, HF_HUB_OFFLINE="1"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.full, self.val = partition(count=2100), partition(split="val", count=2100, offset=10000)
        self.loader = self.stack.enter_context(patch.object(
            smoke, "load_partition", side_effect=lambda domain, split, *args: self.full if split == "train" else self.val,
        ))

        def fake_process(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, "-m", "src.experiments.smoke_tabicl"])
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(command[3], "--reload-request")
            smoke._reload_worker(Path(command[4]))
            return subprocess.CompletedProcess(command, 0)

        self.process = self.stack.enter_context(patch.object(smoke.subprocess, "run", side_effect=fake_process))

    def run_smoke(self, domain="source", **kwargs):
        self.full.summary["domain"] = domain
        self.val.summary["domain"] = domain
        return smoke.run(
            self.config(domain), output_dir=f"results/smoke/tabicl_{domain}_none_fp32",
            **kwargs,
        )

    def freeze_mock_context(self, domain="source"):
        selected = select_training(self.full, 2000, 42)
        reference = selected.provenance.copy()
        reference.insert(0, "sample_order", np.arange(len(reference)))
        reference.insert(0, "split", "train")
        reference.insert(0, "domain", domain)
        path = self.root / "training_reference.csv.gz"
        reference.to_csv(path, index=False)
        self.stack.enter_context(patch.dict(smoke.FROZEN_CONTEXT_HASHES, {domain: selected.summary["ordered_identity_sha256"]}))
        self.stack.enter_context(patch.dict(smoke.CONTEXT_REFERENCES, {domain: path}))

    def test_equivalence_suite_and_offline_reload_request_without_real_models(self):
        output = self.run_smoke()
        report = json.loads((output / "smoke.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["numerically_eligible"])
        self.assertFalse(report["resource_eligible"])
        self.assertEqual(report["criteria"]["absolute_tolerance"], 7.62939453125e-6)
        self.assertEqual(report["queries"]["rows"], 65)
        self.assertEqual(report["sampling"]["selected_training_rows"], 512)
        self.assertEqual([call.args[1] for call in self.loader.call_args_list], ["train", "val"])
        self.assertTrue(report["checks"]["fit_input_nonmutation"])
        self.assertTrue(report["checks"]["prediction_input_nonmutation"])
        self.assertTrue(report["checks"]["unseen_category_finite_nonmutating"])
        for name in ("reload", "permutation", "sentinels_alone", "chunk_1", "chunk_17", "chunk_32", "chunk_128"):
            self.assertTrue(report["checks"][name]["eligible"], name)
        self.assertTrue(report["isolated_checkpoint_removed_before_reload"])
        self.assertTrue(self.checkpoint.is_file())
        self.assertFalse(Path(report["model"]["checkpoint"]["path"]).exists())
        self.assertEqual(FakeEstimator.instances[0].fit_count, 1)
        self.assertEqual(FakeEstimator.load_calls, 1)
        self.assertTrue((output / "native_diagnostics.log").exists())
        self.assertTrue((output / "native_reload.log").exists())
        self.assertFalse((output / "reload_queries.csv").exists())
        self.assertFalse((output / "reload_request.json").exists())
        self.assertIn("not proof", report["resource_screen"]["limitations"])
        self.process.assert_called_once()
        with self.assertRaises(FileExistsError):
            self.run_smoke()

    def test_feasibility_both_domains_uses_exact_mock_context_and_separate_timings(self):
        for domain in ("source", "kaggle"):
            self.full = partition(domain, count=2100)
            self.val = partition(domain, split="val", count=2100, offset=10000)
            self.freeze_mock_context(domain)
            config = self.config(domain)
            config.model_params["device"] = "cuda:0"
            output = smoke.run(
                config, output_dir=f"results/smoke/{domain}_none_fp32_n2000",
                suite="feasibility", sample_size=2000, query_size=257,
            )
            report = json.loads((output / "smoke.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["resource_eligible"])
            self.assertTrue(report["numerically_eligible"])
            self.assertEqual(report["timing_queries"]["rows"], 2049)
            self.assertEqual(len(report["timing"]["repetitions_seconds"]), 3)
            for operation in ("preparation", "prediction_cold", "timing_warmup", "save"):
                entry = report["operations"][operation]
                self.assertGreater(entry["duration_seconds"], 0)
                self.assertEqual(entry["peak_allocated_bytes"], 123)
                self.assertEqual(entry["peak_reserved_bytes"], 456)
            self.assertIn("reload", report["reload"]["operations"])
            self.assertEqual(report["resource_screen"]["recovery_detection"], "no_recovery_detected")

    def test_n2000_wrong_hash_refused_before_estimator_fit(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.run_smoke(sample_size=2000, query_size=257)
        self.assertEqual(FakeEstimator.instances, [])
        report = json.loads((self.root / "results/smoke/tabicl_source_none_fp32/smoke.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("traceback", report["error"])

    def test_n2000_reference_sequence_not_just_hash(self):
        self.freeze_mock_context()
        reference_path = smoke.CONTEXT_REFERENCES["source"]
        import pandas as pd

        reference = pd.read_csv(reference_path)
        reference.loc[0, "row_index"] += 1
        reference.to_csv(reference_path, index=False)
        with self.assertRaisesRegex(ValueError, "identity sequence"):
            self.run_smoke(sample_size=2000, query_size=257)
        self.assertEqual(FakeEstimator.instances, [])

    def test_oom_failure_persisted_without_retry_and_original_checkpoint_untouched(self):
        def fail(model, X, y):
            print("native preparation failure diagnostics")
            raise RuntimeError("CUDA out of memory")

        with patch.object(FakeEstimator, "fit", fail), self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.run_smoke()
        output = self.root / "results/smoke/tabicl_source_none_fp32"
        report = json.loads((output / "smoke.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["operations"]["preparation"]["status"], "failed")
        self.assertEqual(len(FakeEstimator.instances), 1)
        self.assertIn("native preparation failure diagnostics", (output / "native_diagnostics.log").read_text())
        self.assertTrue(self.checkpoint.exists())
        self.assertFalse(list(output.glob("isolated_checkpoint_*")))
        self.process.assert_not_called()

    def test_native_recovery_makes_feasibility_ineligible_even_if_probabilities_match(self):
        self.freeze_mock_context()
        config = self.config()
        config.model_params["device"] = "cuda:0"
        original = FakeEstimator.predict_proba

        def recover(model, X):
            print("OOM with batch_size=4, reducing to 2")
            return original(model, X)

        with patch.object(FakeEstimator, "predict_proba", recover), self.assertLogs("src.models.tabicl", "WARNING"):
            with self.assertRaisesRegex(RuntimeError, "acceptance criteria"):
                smoke.run(
                    config, output_dir="results/smoke/recovery_none_fp32",
                    sample_size=2000, query_size=257, suite="feasibility",
                )
        report = json.loads((self.root / "results/smoke/recovery_none_fp32/smoke.json").read_text())
        self.assertTrue(report["numerically_eligible"])
        self.assertFalse(report["resource_eligible"])
        self.assertEqual(report["resource_screen"]["recovery_detection"], "recovery_detected")

    def test_beyond_tolerance_and_near_threshold_flips_are_not_hidden(self):
        baseline = np.array([0.25, 0.75])
        good = baseline + smoke.FP32_ATOL
        self.assertTrue(smoke.comparison(good, baseline)["eligible"])
        bad = baseline + 2 * smoke.FP32_ATOL
        self.assertFalse(smoke.comparison(bad, baseline)["eligible"])
        result = smoke.comparison(np.array([0.5]), np.array([0.5 - smoke.FP32_ATOL / 2]))
        self.assertTrue(result["within_fixed_criterion"])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["threshold_disagreements"], 1)
        for bad in (np.array([np.nan]), np.array([[0.5]]), np.array([1.1])):
            with self.assertRaises(ValueError):
                smoke.comparison(bad, np.array([0.5]))

    def test_failed_numerical_suite_retains_probabilities_and_exits_failed(self):
        original = FakeEstimator.predict_proba

        def composed(model, X):
            probability = original(model, X)
            if len(X) == 17:
                probability[:, 0] += 2 * smoke.FP32_ATOL
                probability[:, 1] -= 2 * smoke.FP32_ATOL
            return probability

        with patch.object(FakeEstimator, "predict_proba", composed), self.assertRaisesRegex(RuntimeError, "acceptance criteria"):
            self.run_smoke()
        output = self.root / "results/smoke/tabicl_source_none_fp32"
        report = json.loads((output / "smoke.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["numerically_eligible"])
        self.assertFalse(report["checks"]["chunk_17"]["eligible"])
        self.assertTrue((output / "chunk_17.npy").exists())
        self.assertEqual(report["criteria"]["absolute_tolerance"], smoke.FP32_ATOL)

    def test_reload_worker_failure_diagnostics_remain_and_temporaries_are_cleaned(self):
        with patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(RuntimeError, "Fresh-process"):
                self.run_smoke()
        output = self.root / "results/smoke/tabicl_source_none_fp32"
        report = json.loads((output / "smoke.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertTrue((output / "reload_process.log").exists())
        self.assertFalse((output / "reload_request.json").exists())
        self.assertFalse((output / "reload_queries.csv").exists())

    def test_request_guardrails(self):
        config = self.config()
        for changes in (
            {"sample_size": 64}, {"query_size": 0}, {"batch_size": True},
            {"sample_size": 2000, "query_size": 65}, {"suite": "feasibility"},
            {"output_dir": "results/predictive/escape"},
        ):
            params = dict(output_dir="results/smoke/guard", sample_size=512, query_size=65, batch_size=128, suite="equivalence")
            params.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                smoke._validate_request(config, **params)
        config.model_params["n_estimators"] = 1
        with self.assertRaisesRegex(ValueError, "8 ensemble members"):
            smoke._validate_request(config, "results/smoke/reduced", 512, 65, 128, "equivalence")
        config.model_params["n_estimators"] = 8
        config.model_params["use_amp"] = True
        with self.assertRaisesRegex(ValueError, "do not certify AMP"):
            smoke._validate_request(
                config, "results/smoke/amp", 512, 65, 128, "equivalence",
            )


class CommandTests(TabICLTestCase):
    def test_smoke_configs_are_explicit_none_fp32_and_distinct(self):
        outputs = set()
        for domain in ("source", "kaggle"):
            config = load_config(CONFIGS_DIR / "smoke" / f"tabicl_v2_{domain}_none_fp32.json")
            self.assertIn("none_fp32", config.experiment_name)
            self.assertEqual(config.model_params["norm_methods"], ["none"])
            self.assertIs(config.model_params["use_amp"], False)
            self.assertEqual(config.model_params["batch_size"], 1)
            self.assertEqual(config.prediction_batch_size, 128)
            self.assertEqual(config.train_sample_size, 2000)
            self.assertEqual(config.test_domains, ("source", "kaggle", "nhanes"))
            outputs.add(config.resolved_output_dir)
        self.assertEqual(len(outputs), 2)

    def test_help_imports_without_optional_models_and_exposes_no_tolerance_flag(self):
        code = """
import importlib.abc, sys, runpy
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'tabicl', 'tabpfn', 'catboost'}:
            raise AssertionError('Optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.argv = ['smoke_tabicl', '--help']
try:
    runpy.run_module('src.experiments.smoke_tabicl', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TRAIN/VAL ONLY", result.stdout)
        self.assertIn("ensemble", result.stdout)
        self.assertNotIn("--atol", result.stdout)
        self.assertNotIn("--rtol", result.stdout)
