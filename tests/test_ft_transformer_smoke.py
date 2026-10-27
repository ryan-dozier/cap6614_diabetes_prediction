from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data
from src.data import DatasetSchema
from src.experiments import config as config_module
from src.experiments import smoke_ft_transformer as smoke
from src.experiments.config import ExperimentConfig, load_config
from src.experiments.sampling import Partition, partition_summary, select_training
from src.models.ft_transformer import FTTransformerAdapter
from src.models.ft_transformer_config import FT_DEFAULT_PARAMS
from src.paths import CONFIGS_DIR, PROJECT_ROOT


SCHEMA = DatasetSchema(
    target="diagnosed_diabetes",
    features=("numeric", "category"),
    categorical_features=("category",),
    numeric_features=("numeric",),
    source_only_excluded=(),
)


def partition(
    domain: str,
    split: str,
    start: int,
    rows: int,
    *,
    one_class: bool = False,
) -> Partition:
    labels = np.zeros(rows, dtype=np.int64) if one_class else np.arange(rows, dtype=np.int64) % 2
    identities = np.arange(start, start + rows, dtype=np.int64)
    provenance = pd.DataFrame({"row_index": identities})
    X = pd.DataFrame({
        "numeric": labels.astype(np.float64) + np.linspace(-0.15, 0.15, rows),
        "category": ["alpha", "beta", "gamma", "alpha"] * (rows // 4)
        + ["alpha"] * (rows % 4),
    })
    y = pd.Series(labels, name="diagnosed_diabetes")
    return Partition(
        X=X,
        y=y,
        provenance=provenance,
        identities=frozenset((domain, int(value)) for value in identities),
        summary=partition_summary(domain, split, provenance, y),
    )


def config_dict(domain: str = "source", output_dir: str = "results/smoke/base") -> dict:
    return {
        "experiment_name": f"ft_transformer_{domain}_smoke",
        "model": "ft_transformer",
        "train_domain": domain,
        "validation_domain": domain,
        "test_domains": ["source", "kaggle", "nhanes"],
        "feature_set": "strict_common_16",
        "seed": 42,
        "sampling_seed": 42,
        "threshold": 0.5,
        "calibration_bins": 10,
        "train_sample_size": 2000,
        "prediction_batch_size": 256,
        "early_stopping_rounds": None,
        "model_params": dict(FT_DEFAULT_PARAMS),
        "output_dir": output_dir,
    }


class FTTransformerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        (self.results / "smoke").mkdir(parents=True)
        self.path_patches = patch.multiple(
            config_module, PROJECT_ROOT=self.root, RESULTS_DIR=self.results,
        )
        self.path_patches.start()
        self.addCleanup(self.path_patches.stop)
        self.smoke_results_patch = patch.object(smoke, "RESULTS_DIR", self.results)
        self.smoke_results_patch.start()
        self.addCleanup(self.smoke_results_patch.stop)
        self.config = ExperimentConfig.from_dict(config_dict())

    def test_smoke_config_files_are_complete_approved_base_profiles(self):
        for domain in ("source", "kaggle"):
            path = CONFIGS_DIR / "smoke" / f"ft_transformer_{domain}.json"
            config = load_config(path)
            self.assertEqual(config.model, "ft_transformer")
            self.assertEqual(config.train_domain, domain)
            self.assertEqual(config.validation_domain, domain)
            self.assertEqual(config.feature_set, "strict_common_16")
            self.assertEqual(config.train_sample_size, 2000)
            self.assertEqual(config.sampling_seed, 42)
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.model_params, FT_DEFAULT_PARAMS)

    def test_mode_arguments_resolve_and_reject_before_data_access(self):
        functional, details = smoke.resolve_smoke_config(
            self.config, output_dir="results/smoke/functional", mode="functional",
        )
        self.assertEqual(functional.model_params["device"], "cpu")
        self.assertEqual(functional.model_params["max_epochs"], 3)
        self.assertEqual(functional.train_sample_size, 64)
        self.assertEqual(details["effective_device"], "cpu")
        resource, details = smoke.resolve_smoke_config(
            self.config, output_dir="results/smoke/resource", mode="resource",
        )
        self.assertEqual(resource.model_params["max_epochs"], 2)
        self.assertTrue(details["training_budget_truncated"])
        extended, _ = smoke.resolve_smoke_config(
            self.config, output_dir="results/smoke/resource-10",
            mode="resource", resource_epochs=10,
        )
        self.assertEqual(extended.model_params["max_epochs"], 10)
        feasibility, details = smoke.resolve_smoke_config(
            self.config, output_dir="results/smoke/feasibility", mode="feasibility",
        )
        self.assertEqual(feasibility.model_params, FT_DEFAULT_PARAMS)
        self.assertEqual(feasibility.model_params["max_epochs"], 200)
        self.assertFalse(details["training_budget_truncated"])

        invalid = (
            {"mode": "bad"},
            {"mode": "functional", "resource_epochs": 2},
            {"mode": "resource", "resource_epochs": 11},
            {"mode": "resource", "resource_epochs": True},
            {"mode": "functional", "device_override": "cuda"},
            {"mode": "feasibility", "device_override": "cuda:any"},
        )
        for index, kwargs in enumerate(invalid):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                smoke.resolve_smoke_config(
                    self.config,
                    output_dir=f"results/smoke/invalid-{index}",
                    **kwargs,
                )

    def test_device_override_is_explicit_and_never_implicit_fallback(self):
        effective, details = smoke.resolve_smoke_config(
            self.config,
            output_dir="results/smoke/cpu-feasibility",
            mode="feasibility",
            device_override="cpu",
        )
        self.assertEqual(effective.model_params["device"], "cpu")
        self.assertEqual(details["device_override"], "cpu")
        self.assertFalse(details["effective_device"] == details["base_device"])

    def test_nhanes_cross_domain_and_wrong_base_profiles_are_rejected(self):
        nhanes = ExperimentConfig.from_dict(config_dict("nhanes"))
        with self.assertRaisesRegex(ValueError, "source or kaggle"):
            smoke.resolve_smoke_config(
                nhanes, output_dir="results/smoke/nhanes", mode="functional",
            )
        with self.assertRaisesRegex(ValueError, "validation_domain must equal train_domain"):
            ExperimentConfig.from_dict({
                **config_dict(),
                "validation_domain": "kaggle",
            })
        bad_profile = replace(self.config, model_params={**FT_DEFAULT_PARAMS, "max_epochs": 100})
        with self.assertRaisesRegex(ValueError, "exact approved"):
            smoke.resolve_smoke_config(
                bad_profile, output_dir="results/smoke/changed-profile", mode="feasibility",
            )

    def test_resource_and_feasibility_populations_reuse_exact_shared_sampler(self):
        full = partition("source", "train", 0, 2050)
        validation = partition("source", "val", 3000, 40)
        expected = select_training(full, 2000, 42)
        with patch.object(smoke, "select_training", wraps=select_training) as sampler:
            selected, selected_validation, metadata = smoke.select_smoke_populations(
                full, validation, mode="resource", seed=42,
            )
        sampler.assert_called_once_with(full, 2000, 42)
        pd.testing.assert_frame_equal(selected.X, expected.X)
        pd.testing.assert_frame_equal(selected.provenance, expected.provenance)
        self.assertEqual(selected.summary["identity_sha256"], expected.summary["identity_sha256"])
        self.assertEqual(
            selected.summary["ordered_identity_sha256"],
            expected.summary["ordered_identity_sha256"],
        )
        self.assertIs(selected_validation, validation)
        self.assertEqual(metadata["selected_training_rows"], 2000)

    def test_functional_subsets_and_validation_queries_are_deterministic(self):
        full = partition("source", "train", 0, 100)
        validation = partition("source", "val", 1000, 50)
        train_a, val_a, _ = smoke.select_smoke_populations(
            full, validation, mode="functional", seed=42,
        )
        train_b, val_b, _ = smoke.select_smoke_populations(
            full, validation, mode="functional", seed=42,
        )
        self.assertEqual(len(train_a.X), 64)
        self.assertEqual(len(val_a.X), 32)
        self.assertEqual(set(train_a.y), {0, 1})
        self.assertEqual(set(val_a.y), {0, 1})
        self.assertEqual(train_a.provenance["row_index"].tolist(), train_b.provenance["row_index"].tolist())
        self.assertEqual(val_a.provenance["row_index"].tolist(), val_b.provenance["row_index"].tolist())
        self.assertFalse(train_a.identities & val_a.identities)
        queries_a, identities_a, summary_a = smoke._select_queries(val_a, 44)
        queries_b, identities_b, summary_b = smoke._select_queries(val_a, 44)
        pd.testing.assert_frame_equal(queries_a, queries_b)
        pd.testing.assert_frame_equal(identities_a, identities_b)
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(len(queries_a), 32)
        self.assertEqual(summary_a["partition_summary"]["ordered_identity_sha256"],
                         summary_b["partition_summary"]["ordered_identity_sha256"])

    def test_both_class_and_disjoint_identity_requirements(self):
        full = partition("source", "train", 0, 80)
        one_class_validation = partition("source", "val", 100, 40, one_class=True)
        with self.assertRaisesRegex(ValueError, "both classes"):
            smoke.select_smoke_populations(
                full, one_class_validation, mode="functional", seed=42,
            )
        overlapping_validation = partition("source", "val", 20, 40)
        with self.assertRaisesRegex(ValueError, "Identity overlap"):
            smoke.select_smoke_populations(
                full, overlapping_validation, mode="functional", seed=42,
            )

    def _install_partitions(self, train: Partition, validation: Partition):
        calls: list[tuple[str, str]] = []

        def fake_loader(domain, split, feature_set, schema):
            calls.append((domain, split))
            self.assertEqual(feature_set, "strict_common_16")
            self.assertIs(schema, SCHEMA)
            if split == "train":
                return train
            if split == "val":
                return validation
            raise AssertionError(f"Forbidden smoke partition request: {domain}/{split}")

        self.enterContext(patch.object(data, "load_feature_schema", return_value=SCHEMA))
        self.enterContext(patch.object(smoke, "load_partition", side_effect=fake_loader))
        self.enterContext(patch.object(smoke, "environment_metadata", return_value={
            "packages": {"torch": "test", "rtdl-revisiting-models": "test"},
        }))
        return calls

    def test_real_functional_fit_checks_persistence_and_information_boundary(self):
        train = partition("source", "train", 0, 80)
        validation = partition("source", "val", 1000, 40)
        calls = self._install_partitions(train, validation)
        output_dir = "results/smoke/functional-success"
        with contextlib.redirect_stdout(io.StringIO()):
            output = smoke.run(
                self.config, output_dir=output_dir, mode="functional",
            )
        self.assertEqual(calls, [("source", "train"), ("source", "val")])
        report = json.loads((output / "smoke.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["mode"], "functional")
        self.assertEqual(report["training"]["population_size"], 64)
        self.assertEqual(report["validation"]["population_size"], 32)
        self.assertEqual(report["training"]["epoch_cap"], 3)
        self.assertLessEqual(report["training"]["executed_epochs"], 3)
        self.assertIn(report["training"]["stopping_reason"], ("patience", "max_epochs"))
        self.assertFalse(report["training"]["truncated"])
        self.assertFalse(report["validation"]["complete"])
        self.assertEqual(report["full_partition_sizes"], {"training": 80, "validation": 40})
        self.assertEqual(
            report["information_boundary"],
            "Only source/kaggle train and same-domain val partitions are permitted. No test partition or NHANES data is loaded.",
        )
        self.assertEqual(report["device"]["effective"], "cpu")
        self.assertEqual(report["device"]["explicit_override"], None)
        self.assertFalse(report["device"]["matches_intended_feasibility_device"])
        self.assertEqual(report["requested_configuration"]["model_params"], FT_DEFAULT_PARAMS)
        self.assertEqual(report["effective_configuration"]["model_params"]["n_blocks"], 1)
        self.assertTrue((output / "requested_config.json").is_file())
        self.assertTrue((output / "effective_config.json").is_file())
        self.assertTrue((output / "training_identities.csv.gz").is_file())
        self.assertTrue((output / "validation_identities.csv.gz").is_file())
        self.assertTrue((output / "validation_query_identities.csv.gz").is_file())
        self.assertTrue((output / "reference_probabilities.npy").is_file())
        self.assertTrue((output / "model.ft_transformer.pt").is_file())
        check_names = (
            "reference_prediction",
            "repeated_prediction",
            "permuted_prediction_restored_order",
            *(f"prediction_batch_{size}" for size in smoke.QUERY_BATCH_SIZES),
            "save_reload",
            "query_input_unchanged",
        )
        for name in check_names:
            self.assertEqual(report["checks"][name]["status"], "passed", name)
        for size in (3, 7):
            self.assertNotEqual(len(report["validation_queries"]["gateway_positions"]) % size, 0)
        self.assertEqual(report["criteria"], {"rtol": 1e-5, "atol": 1e-6, "threshold": 0.5})
        self.assertGreater(report["total_duration_seconds"], 0)
        self.assertGreater(report["operations"]["fit"]["duration_seconds"], 0)
        self.assertGreater(report["operations"]["save"]["duration_seconds"], 0)
        self.assertGreater(report["operations"]["reload"]["duration_seconds"], 0)
        self.assertIsNone(report["operations"]["fit"]["resources"]["peak_gpu_memory"])
        self.assertIn("unseen_category_counts", report["validation"])
        queries = pd.read_csv(output / "validation_query_identities.csv.gz")
        self.assertEqual(len(queries), 32)
        with self.assertRaises(FileExistsError):
            smoke.run(self.config, output_dir=output_dir, mode="functional")

    def test_training_failure_writes_report_and_preserves_exception(self):
        self._install_partitions(
            partition("source", "train", 0, 80),
            partition("source", "val", 1000, 40),
        )
        with (
            patch.object(
                FTTransformerAdapter, "fit",
                side_effect=RuntimeError("controlled smoke training failure"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "controlled smoke training failure"),
        ):
            smoke.run(
                self.config, output_dir="results/smoke/fit-failure", mode="functional",
            )
        report = json.loads(
            (self.results / "smoke/fit-failure/smoke.json").read_text(),
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["message"], "controlled smoke training failure")
        self.assertEqual(report["operations"]["fit"]["status"], "failed")
        self.assertTrue((self.results / "smoke/fit-failure/training_identities.csv.gz").exists())
        self.assertFalse((self.results / "smoke/fit-failure/model.ft_transformer.pt").exists())

    def test_comparison_failure_is_persisted_before_exception(self):
        self._install_partitions(
            partition("source", "train", 0, 80),
            partition("source", "val", 1000, 40),
        )

        def failed_comparison(*args, **kwargs):
            return {
                "status": "failed",
                "query_rows": 32,
                "batch_size": 7,
                "max_absolute_probability_difference": 0.25,
                "threshold_disagreement_count": 1,
                "rtol": 1e-5,
                "atol": 1e-6,
            }

        with (
            patch.object(smoke, "_compare", side_effect=failed_comparison),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(AssertionError, "repeated_prediction"),
        ):
            smoke.run(
                self.config, output_dir="results/smoke/comparison-failure", mode="functional",
            )
        report = json.loads(
            (self.results / "smoke/comparison-failure/smoke.json").read_text(),
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["checks"]["repeated_prediction"]["status"], "failed")
        self.assertEqual(
            report["checks"]["repeated_prediction"]["max_absolute_probability_difference"],
            0.25,
        )
        self.assertTrue((self.results / "smoke/comparison-failure/reference_probabilities.npy").is_file())

    def test_invalid_cli_resource_cap_exits_nonzero_before_data_access(self):
        config_path = CONFIGS_DIR / "smoke/ft_transformer_source.json"
        with patch.object(smoke, "load_partition", side_effect=AssertionError("data must not load")):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    smoke.main([
                        "--config", str(config_path),
                        "--output-dir", "results/smoke/invalid-cli",
                        "--mode", "resource",
                        "--resource-epochs", "11",
                    ])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse((self.results / "smoke/invalid-cli").exists())


if __name__ == "__main__":
    unittest.main()
