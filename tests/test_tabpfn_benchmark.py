from __future__ import annotations

import contextlib
import gc
import io
import json
import subprocess
import sys
import tempfile
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import DatasetSchema
from src.experiments import benchmark_tabpfn as benchmark
from src.experiments.config import load_config
from src.experiments.sampling import Partition, partition_summary
from src.models.base import ModelAdapter
from src.paths import CONFIGS_DIR, PROJECT_ROOT


SCHEMA = DatasetSchema("diagnosed_diabetes", ("value",), (), ("value",), ())


def partition(split, count=2100, domain="source"):
    identifiers = np.arange(count, dtype=np.int64) + (10000 if split == "val" else 0)
    provenance = pd.DataFrame({"row_index": identifiers})
    if domain == "nhanes":
        provenance["seqn"] = identifiers[::-1] + 50000
    y = pd.Series(np.arange(count) % 2, name=SCHEMA.target)
    identity = "seqn" if domain == "nhanes" else "row_index"
    return Partition(
        pd.DataFrame({"value": identifiers / 100000}), y, provenance,
        frozenset((domain, int(value)) for value in provenance[identity]),
        partition_summary(domain, split, provenance, y),
    )


class FakeAdapter(ModelAdapter):
    def __init__(self, plan, clock):
        super().__init__(SCHEMA.features, (), SCHEMA.target)
        self.plan = plan
        self.clock = clock
        self.events = []
        self.fit_inputs = []
        self.calls = []
        self.full_calls = 0
        self.batch_size = None
        self.current_rows = []

    def fit(self, X_train, y_train, X_val, y_val):
        self.events.append("fit")
        self.fit_inputs.append((X_train.copy(), y_train.copy()))
        self.clock[0] += 2
        if "fit_error" in self.plan:
            raise self.plan["fit_error"]

    def predict_proba_batched(self, X, batch_size):
        self.batch_size = batch_size
        self.full_calls += 1
        self.current_rows = []
        if self.full_calls == self.plan.get("fail_call"):
            raise self.plan["error"]
        self.events.append("predict")
        result = super().predict_proba_batched(X, batch_size)
        self.calls.append(self.current_rows)
        self.clock[0] += self.plan.get("durations", [1] * 5)[self.full_calls - 1]
        if self.full_calls in self.plan.get("disagree_calls", []):
            result = result + 0.01
        if self.full_calls == self.plan.get("invalid_call"):
            return self.plan["invalid_value"]
        return result

    def predict_proba(self, X):
        self.current_rows.append(X["value"].tolist())
        return X["value"].to_numpy(copy=True)

    def synchronize(self):
        self.events.append("synchronize")

    def reset_peak_memory(self):
        self.events.append("reset")

    def resource_metadata(self):
        self.events.append("memory")
        if self.plan.get("memory_error") and self.full_calls:
            raise RuntimeError("memory diagnostics broken")
        return {"peak_gpu_memory": {"allocated_bytes": 100 + self.full_calls, "reserved_bytes": 200 + self.full_calls}}

    def save(self, path):
        raise AssertionError("The benchmark must not save a fitted research model.")

    def metadata(self):
        return {
            "checkpoint": {"path": "fake-local", "sha256": "fake-hash"},
            "compute_settings": {
                "device": "cuda:0", "forced_dtype": "torch.float32",
                "autocast_enabled": False,
            },
        }


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = replace(load_config(CONFIGS_DIR / "tabpfn_source_n2000_float32.json"), train_sample_size=20)
        self.before_config = self.config.to_dict()
        self.research_bytes = {path: path.read_bytes() for path in CONFIGS_DIR.glob("*.json")}
        self.train = partition("train", 40)
        self.val = partition("val")
        self.adapters = []
        self.created_configs = []
        self.loads = []
        self.plans = []
        self.clock = [0.]
        self.cleanup = []
        self.active = False

        def create(config, schema):
            self.assertFalse(self.active, "Two adapters are alive across cases")
            self.active = True
            self.created_configs.append(config.to_dict())
            plan = self.plans[len(self.adapters)] if len(self.adapters) < len(self.plans) else {}
            adapter = FakeAdapter(plan, self.clock)
            self.adapters.append(adapter)
            return adapter

        def load(domain, split, feature_set, schema):
            self.assertIn(split, ("train", "val"), "Test data must never be loaded")
            self.loads.append((domain, split))
            return self.train if split == "train" else self.val

        def release(device):
            self.cleanup.append(device)
            self.active = False

        for target, value in (
            ("RESULTS_DIR", self.root),
            ("create_adapter", create),
            ("load_partition", load),
            ("release_cuda", release),
        ):
            patcher = patch.object(benchmark, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(benchmark.data, "load_feature_schema", return_value=SCHEMA)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(benchmark.time, "perf_counter", side_effect=lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.console = io.StringIO()
        redirect = contextlib.redirect_stdout(self.console)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def tearDown(self):
        self.assertEqual(self.config.to_dict(), self.before_config)
        self.assertEqual({path: path.read_bytes() for path in CONFIGS_DIR.glob("*.json")}, self.research_bytes)

    def output(self):
        return self.root / "benchmarks" / "unit"

    def run_benchmark(self, **kwargs):
        return benchmark.run(self.config, output_dir=self.output(), **kwargs)

    def report(self):
        return json.loads((self.output() / "benchmark.json").read_text())

    def test_protocol_artifacts_order_and_summaries(self):
        self.plans = [
            {"durations": [5, 4, 2, 3]},
            {"durations": [5, 2, 1, 1.5]},
            {"durations": [5, 1, .5, .75]},
        ]
        output = self.run_benchmark(batch_sizes=[1024, 256, 512])
        report = self.report()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(self.loads, [("source", "train"), ("source", "val")])
        self.assertEqual([adapter.batch_size for adapter in self.adapters], [256, 1024, 512])
        self.assertEqual(len(self.cleanup), 3)
        expected = self.val.X.iloc[:2049]["value"].tolist()
        for adapter in self.adapters:
            self.assertEqual(len(adapter.fit_inputs), 1)
            self.assertEqual(adapter.full_calls, 4)
            for call in adapter.calls:
                self.assertEqual([item for chunk in call for item in chunk], expected)
                self.assertEqual(len(call[-1]), 1)
                self.assertEqual(len(call), (2048 // adapter.batch_size) + 1)
            pd.testing.assert_frame_equal(adapter.fit_inputs[0][0], self.adapters[0].fit_inputs[0][0])
            pd.testing.assert_series_equal(adapter.fit_inputs[0][1], self.adapters[0].fit_inputs[0][1])
            self.assertEqual(adapter.events[:5], ["synchronize", "reset", "fit", "synchronize", "memory"])
            self.assertEqual(
                adapter.events[5:],
                ["synchronize", "reset", "predict", "synchronize", "memory"] * 4,
            )
        self.assertEqual(self.created_configs, [self.before_config] * 3)
        by_size = {case["batch_size"]: case for case in report["cases"]}
        self.assertEqual(by_size[256]["median_duration_seconds"], 3)
        self.assertEqual(by_size[1024]["median_duration_seconds"], 1.5)
        self.assertEqual(by_size[1024]["median_rows_per_second"], 2049 / 1.5)
        self.assertEqual(by_size[1024]["speedup_vs_256"], 2)
        self.assertEqual(by_size[256]["min_duration_seconds"], 2)
        self.assertEqual(by_size[256]["max_duration_seconds"], 4)
        self.assertEqual(by_size[256]["context_preparation"]["duration_seconds"], 2)
        self.assertEqual(by_size[256]["warmup"]["duration_seconds"], 5)
        self.assertEqual(by_size[256]["peak_allocated_bytes"], 104)
        self.assertEqual(report["fastest_eligible_batch_size"], 512)
        self.assertEqual(len(list(output.glob("*.npy"))), 9)
        self.assertEqual(report["reference_probability_artifact"], "probabilities_batch256_repetition1.npy")
        for case in report["cases"]:
            self.assertTrue(case["numerically_eligible"])
            for row in case["repetitions"]:
                np.testing.assert_array_equal(np.load(output / row["probability_artifact"], allow_pickle=False), expected)
                self.assertTrue(row["allclose"])
                self.assertEqual(row["threshold_disagreements"], 0)
        summary = pd.read_csv(output / "benchmark_summary.csv")
        self.assertEqual(summary["batch_size"].tolist(), [1024, 256, 512])
        self.assertEqual(len(pd.read_csv(output / "benchmark_repetitions.csv")), 9)
        provenance = pd.read_csv(output / "validation_queries.csv.gz")
        self.assertEqual(provenance.columns.tolist(), ["domain", "split", "query_order", "row_index"])
        self.assertEqual(provenance["row_index"].tolist(), list(range(10000, 12049)))
        self.assertEqual(len(pd.read_csv(output / "training_sample.csv.gz")), 20)
        self.assertNotIn("class_counts", report["validation_queries"])
        self.assertIn("resolved device=cuda:0, precision=torch.float32", self.console.getvalue())

    def test_argument_and_configuration_validation(self):
        for kwargs in (
            {"query_size": 0}, {"query_size": True}, {"query_size": 1.2},
            {"repetitions": 0}, {"repetitions": False},
            {"batch_sizes": []}, {"batch_sizes": [512]},
            {"batch_sizes": [256, 256]}, {"batch_sizes": [256, 0]},
            {"batch_sizes": [256, True]}, {"batch_sizes": [256, 1.5]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_benchmark(**kwargs)
        invalid_configs = [
            replace(self.config, model_params={**self.config.model_params, "inference_precision": value})
            for value in ("auto", "autocast")
        ]
        invalid_configs += [
            replace(self.config, model_params={key: value for key, value in self.config.model_params.items() if key != "inference_precision"}),
            replace(self.config, model_params={**self.config.model_params, "device": "cpu"}),
            replace(self.config, model_params={key: value for key, value in self.config.model_params.items() if key != "device"}),
            replace(self.config, train_sample_size=None, sampling_seed=None),
            load_config(CONFIGS_DIR / "catboost_source_n2000.json"),
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                benchmark.run(config, output_dir=self.output())
        for path in (self.root, self.root / "benchmarks", self.root / "other", self.root / "benchmarks" / ".." / "escape"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                benchmark.validate_request(self.config, path, 1, [256], 1)
        self.assertFalse(self.output().exists())
        self.assertFalse(self.adapters)

    def test_symlink_escape_and_research_output_refused(self):
        (self.root / "benchmarks").mkdir()
        (self.root / "benchmarks" / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            benchmark.validate_request(self.config, self.root / "benchmarks" / "link" / "escape", 1, [256], 1)
        with patch("src.experiments.config.RESULTS_DIR", self.root):
            config = replace(self.config, output_dir=str(self.output()))
            with self.assertRaisesRegex(ValueError, "separate"):
                benchmark.validate_request(config, self.output(), 1, [256], 1)

    def test_existing_output_refusal_preserves_files(self):
        self.output().mkdir(parents=True)
        sentinel = self.output() / "benchmark.json"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_benchmark()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse(self.loads)

    def test_query_selection_deterministic_and_label_independent(self):
        for domain in ("source", "kaggle", "nhanes"):
            val = partition("val", 30, domain)
            queries, provenance, summary = benchmark.select_queries(val, 9)
            order = np.random.default_rng(99).permutation(30)
            reordered = Partition(
                val.X.iloc[order], 1 - val.y.iloc[order], val.provenance.iloc[order],
                val.identities, val.summary,
            )
            again = benchmark.select_queries(reordered, 9)
            pd.testing.assert_frame_equal(queries, again[0])
            pd.testing.assert_frame_equal(provenance, again[1])
            self.assertEqual(summary, again[2])
            identity = "seqn" if domain == "nhanes" else "row_index"
            self.assertEqual(provenance[identity].tolist(), sorted(val.provenance[identity])[:9])
        with self.assertRaisesRegex(ValueError, "validation"):
            benchmark.select_queries(self.train, 1)

    def test_query_overflow_and_identity_overlap_fail_before_adapter(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.run_benchmark(query_size=2200)
        self.assertEqual(self.report()["status"], "failed")
        self.assertTrue(all(case["status"] == "not_run" for case in self.report()["cases"]))
        self.assertFalse(self.adapters)

    def test_overlap_fails(self):
        self.val = partition("train")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_benchmark()
        self.assertFalse(self.adapters)

    def test_fixed_reference_and_candidate_disagreement(self):
        self.plans = [{}, {"disagree_calls": [2]}, {}]
        self.run_benchmark()
        report = self.report()
        self.assertEqual(report["status"], "completed_with_issues")
        candidate = report["cases"][1]
        self.assertEqual(candidate["status"], "agreement_failed")
        self.assertEqual(candidate["execution_status"], "completed")
        self.assertFalse(candidate["numerically_eligible"])
        self.assertAlmostEqual(candidate["repetitions"][0]["max_absolute_difference"], .01)
        self.assertTrue(candidate["repetitions"][1]["allclose"])
        self.assertTrue(report["cases"][2]["numerically_eligible"])
        self.assertNotEqual(report["fastest_eligible_batch_size"], 512)

    def test_reference_disagreement_stops(self):
        self.plans = [{"disagree_calls": [3]}]
        with self.assertRaisesRegex(RuntimeError, "baseline is unstable"):
            self.run_benchmark()
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cases"][0]["status"], "agreement_failed")
        self.assertFalse(report["cases"][0]["numerically_eligible"])
        self.assertEqual(report["cases"][1]["status"], "not_run")
        self.assertEqual(len(self.adapters), 1)
        self.assertEqual(self.adapters[0].full_calls, 3)
        self.assertEqual(len(list(self.output().glob("*.npy"))), 2)

    def test_reference_oom_stops_and_preserves_diagnostics(self):
        error = RuntimeError("CUDA out of memory. fake")
        self.plans = [{"fail_call": 3, "error": error}]
        with self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, error)
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failing_operation"], "prediction")
        self.assertEqual(report["failing_batch_size"], 256)
        self.assertEqual(report["failing_repetition"], 2)
        self.assertEqual(report["cases"][0]["status"], "out_of_memory")
        self.assertEqual(report["cases"][1]["status"], "not_run")
        rows = pd.read_csv(self.output() / "benchmark_repetitions.csv")
        self.assertEqual(rows["status"].tolist(), ["completed", "out_of_memory"])
        self.assertEqual(len(self.cleanup), 1)

    def test_candidate_oom_continues_after_cleanup(self):
        self.plans = [{}, {"fail_call": 3, "error": RuntimeError("CUDA out of memory")}, {}]
        self.run_benchmark()
        report = self.report()
        self.assertEqual(report["status"], "completed_with_issues")
        self.assertEqual(report["cases"][1]["status"], "out_of_memory")
        self.assertFalse(report["cases"][1]["numerically_eligible"])
        self.assertEqual(report["cases"][1]["completed_repetitions"], 1)
        self.assertEqual(report["cases"][1]["failing_repetition"], 2)
        self.assertTrue(report["cases"][2]["numerically_eligible"])
        self.assertEqual(len(self.cleanup), 3)

    def test_adapter_references_released_between_cases_even_after_oom(self):
        references = []
        plans = [{}, {"fail_call": 2, "error": RuntimeError("CUDA out of memory")}, {}]

        def create(config, schema):
            self.assertTrue(all(reference() is None for reference in references))
            adapter = FakeAdapter(plans[len(references)], self.clock)
            references.append(weakref.ref(adapter))
            return adapter

        def release(device):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))

        with patch.object(benchmark, "create_adapter", side_effect=create), patch.object(
            benchmark, "release_cuda", side_effect=release,
        ):
            self.run_benchmark()
        self.assertEqual(len(references), 3)
        self.assertEqual(self.report()["status"], "completed_with_issues")

    def test_candidate_unexpected_error_stops_remaining_cases(self):
        error = ValueError("not an OOM")
        self.plans = [{}, {"fail_call": 3, "error": error}, {}]
        with self.assertRaises(ValueError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, error)
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cases"][1]["status"], "failed")
        self.assertEqual(report["cases"][2]["status"], "not_run")
        self.assertEqual(report["cases"][1]["completed_repetitions"], 1)
        self.assertEqual(report["cases"][1]["median_duration_seconds"], 1)

    def test_failed_candidate_diagnostics_preserve_original_oom(self):
        error = RuntimeError("CUDA out of memory")
        self.plans = [{}, {"fail_call": 2, "error": error}]
        real_persist = benchmark.persist

        def persist(output, report):
            if report["cases"][1].get("error"):
                raise OSError("cannot persist OOM")
            real_persist(output, report)

        with patch.object(benchmark, "persist", side_effect=persist), self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, error)
        self.assertEqual(len(self.adapters), 2)
        self.assertIn("Could not save benchmark diagnostics", " ".join(error.__notes__))

    def test_unexpected_comparison_failure_is_not_masked(self):
        error = ValueError("comparison failed")
        with patch.object(benchmark, "comparison", side_effect=error), self.assertRaises(ValueError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.report()["failing_operation"], "agreement")
        self.assertEqual(self.report()["cases"][0]["repetitions"][0]["status"], "failed")

    def test_cleanup_failure_after_predictions_stops_and_preserves_measurements(self):
        with patch.object(benchmark, "release_cuda", side_effect=OSError("cleanup failure")):
            with self.assertRaisesRegex(OSError, "cleanup failure"):
                self.run_benchmark()
        case = self.report()["cases"][0]
        self.assertEqual(self.report()["failing_operation"], "cleanup")
        self.assertEqual(case["status"], "failed")
        self.assertEqual(case["completed_repetitions"], 3)
        self.assertFalse(case["numerically_eligible"])
        self.assertEqual(case["median_duration_seconds"], 1)

    def test_context_and_warmup_oom_are_distinguished(self):
        for plan, operation in (
            ({"fit_error": RuntimeError("CUDA out of memory")}, "context_preparation"),
            ({"fail_call": 1, "error": RuntimeError("CUDA out of memory")}, "warmup"),
        ):
            with self.subTest(operation=operation):
                self.plans = [plan]
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    self.run_benchmark()
                report = self.report()
                self.assertEqual(report["failing_operation"], operation)
                self.assertIsNone(report["failing_repetition"])
                self.assertFalse(report["cases"][0]["repetitions"])
                # Each subcase has an independent new output.
                self.root = self.root / "next"
                self.adapters.clear()
                self.active = False
                benchmark.RESULTS_DIR = self.root

    def test_non_oom_and_interrupt_propagate(self):
        for error in (ValueError("unexpected"), RuntimeError("CPU out of memory"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.plans = [{"fail_call": 2, "error": error}]
                with self.assertRaises(type(error)) as caught:
                    self.run_benchmark()
                self.assertIs(caught.exception, error)
                self.assertEqual(self.report()["status"], "failed")
                self.assertEqual(self.report()["cases"][0]["status"], "failed")
                self.root = self.root / "next"
                self.adapters.clear()
                self.active = False
                benchmark.RESULTS_DIR = self.root

    def test_invalid_probabilities_fail_explicitly(self):
        for value in (np.array([np.nan]), np.array([1.1]), np.zeros((1, 1)), np.zeros(2)):
            with self.subTest(value=value):
                self.plans = [{"invalid_call": 2, "invalid_value": value}]
                with self.assertRaises(ValueError):
                    self.run_benchmark(query_size=1)
                self.assertEqual(self.report()["failing_operation"], "probability_validation")
                self.root = self.root / "next"
                self.adapters.clear()
                self.active = False
                benchmark.RESULTS_DIR = self.root

    def test_diagnostics_failure_does_not_mask_original(self):
        original = RuntimeError("CUDA out of memory")
        self.plans = [{"fail_call": 2, "error": original}]
        real_persist = benchmark.persist

        def persist(output, report):
            if report["cases"][0].get("error"):
                raise OSError("diagnostic storage failed")
            real_persist(output, report)

        with patch.object(benchmark, "persist", side_effect=persist), self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, original)
        self.assertIn("Could not save benchmark diagnostics", " ".join(original.__notes__))
        self.assertEqual(len(self.cleanup), 1)

    def test_cleanup_failure_does_not_mask_candidate_oom(self):
        original = RuntimeError("CUDA out of memory")
        self.plans = [{}, {"fail_call": 2, "error": original}]

        def cleanup(device):
            self.active = False
            if len(self.adapters) == 2:
                raise OSError("cleanup failed")

        with patch.object(benchmark, "release_cuda", side_effect=cleanup), self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, original)
        self.assertIn("cleanup also failed", " ".join(original.__notes__))
        self.assertEqual(self.report()["status"], "failed")
        self.assertEqual(self.report()["cases"][2]["status"], "not_run")

    def test_memory_diagnostic_failure_preserves_prediction_error(self):
        original = RuntimeError("CUDA out of memory")
        self.plans = [{"fail_call": 1, "error": original, "memory_error": True}]
        with self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, original)
        self.assertIn("memory diagnostics", " ".join(original.__notes__))

    def test_cli_completed_with_issues_nonzero_and_no_tolerance_flags(self):
        self.plans = [{}, {"disagree_calls": [2]}]
        argv = [
            "benchmark", "--config", str(CONFIGS_DIR / "tabpfn_source_n2000_float32.json"),
            "--output-dir", str(self.output()), "--batch-sizes", "256", "512",
            "--query-size", "1", "--repetitions", "1",
        ]
        with patch.object(benchmark, "load_config", return_value=self.config), patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as caught:
                benchmark.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(self.report()["benchmark_settings"]["rtol"], 1e-5)
        self.assertEqual(self.report()["benchmark_settings"]["atol"], 1e-6)
        self.assertEqual(self.report()["command_arguments"]["query_size"], 1)

    def test_import_and_help_do_not_load_gpu_runtime(self):
        code = (
            "import runpy,sys;"
            "sys.argv=['benchmark','--help'];"
            "\ntry: runpy.run_module('src.experiments.benchmark_tabpfn',run_name='__main__')"
            "\nexcept SystemExit as exc: assert exc.code == 0"
            "\nassert 'torch' not in sys.modules"
            "\nassert 'tabpfn' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--rtol", result.stdout)
        self.assertNotIn("--atol", result.stdout)


if __name__ == "__main__":
    unittest.main()
