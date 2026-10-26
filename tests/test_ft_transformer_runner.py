from __future__ import annotations

import builtins
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import DatasetSchema
from src.experiments import config as config_module
from src.experiments import runner
from src.experiments.config import ExperimentConfig
from src.experiments.metrics import evaluate_binary
from src.experiments.sampling import Partition, partition_summary, select_training
from src.models.ft_transformer import FTTransformerAdapter
from src.models.ft_transformer_config import FT_DEFAULT_PARAMS
from src.paths import PROJECT_ROOT


TARGET = "diagnosed_diabetes"
SCHEMA = DatasetSchema(
    target=TARGET,
    features=("numeric", "category"),
    categorical_features=("category",),
    numeric_features=("numeric",),
    source_only_excluded=(),
)


def minimal_params() -> dict:
    return {
        "device": "cpu",
        "n_blocks": 1,
        "d_block": 16,
        "attention_n_heads": 4,
        "ffn_d_hidden": 32,
        "ffn_d_hidden_multiplier": None,
        "max_epochs": 2,
        "patience": 1,
        "train_batch_size": 8,
        "validation_batch_size": 4,
    }


def config_values(output_dir: str = "results/ft-transformer-run") -> dict:
    return {
        "experiment_name": "ft_transformer_synthetic_integration",
        "model": "ft_transformer",
        "train_domain": "source",
        "validation_domain": "source",
        "test_domains": ["kaggle", "nhanes"],
        "feature_set": "strict_common_16",
        "seed": 42,
        "threshold": 0.5,
        "calibration_bins": 5,
        "model_params": minimal_params(),
        "output_dir": output_dir,
        "early_stopping_rounds": None,
        "train_sample_size": 16,
        "sampling_seed": 42,
        "prediction_batch_size": 3,
    }


def make_partition(
    domain: str,
    split: str,
    *,
    start: int,
    rows: int,
    categories: list[str],
) -> Partition:
    labels = np.arange(rows, dtype=np.int64) % 2
    row_index = np.arange(start, start + rows, dtype=np.int64)
    provenance = pd.DataFrame({"row_index": row_index})
    identity_column = "row_index"
    identities = row_index
    if domain == "nhanes":
        seqn = row_index + 50_000
        provenance["seqn"] = seqn
        identity_column = "seqn"
        identities = seqn
    X = pd.DataFrame({
        "numeric": labels.astype(np.float64) + np.linspace(-0.1, 0.1, rows),
        "category": categories,
    })
    y = pd.Series(labels, name=TARGET)
    return Partition(
        X=X,
        y=y,
        provenance=provenance,
        identities=frozenset((domain, int(value)) for value in identities),
        summary=partition_summary(domain, split, provenance, y),
    )


def synthetic_partitions() -> tuple[Partition, Partition, Partition, Partition]:
    return (
        make_partition(
            "source", "train", start=0, rows=30,
            categories=["alpha", "beta", "gamma"] * 10,
        ),
        make_partition(
            "source", "val", start=100, rows=10,
            categories=["alpha", "validation-only"] * 5,
        ),
        make_partition(
            "kaggle", "test", start=200, rows=11,
            categories=["alpha", "kaggle-only"] * 5 + ["alpha"],
        ),
        make_partition(
            "nhanes", "test", start=300, rows=7,
            categories=["beta", "nhanes-only"] * 3 + ["beta"],
        ),
    )


class FTTransformerConfigTests(unittest.TestCase):
    def make_config(self, **updates) -> ExperimentConfig:
        return ExperimentConfig.from_dict({**config_values(), **updates})

    def test_minimal_cpu_and_frozen_research_profile_are_accepted(self):
        minimal = self.make_config()
        self.assertEqual(minimal.model, "ft_transformer")
        self.assertIsNone(minimal.early_stopping_rounds)
        self.assertEqual(minimal.model_params["device"], "cpu")

        profile = dict(FT_DEFAULT_PARAMS)
        profile["device"] = "cuda:0"
        research = self.make_config(
            model_params=profile,
            seed=42,
            train_sample_size=2000,
            sampling_seed=42,
            prediction_batch_size=256,
        )
        self.assertEqual(research.model_params, profile)
        self.assertEqual(research.to_dict()["model"], "ft_transformer")
        self.assertEqual(
            ExperimentConfig.from_dict(research.to_dict()).to_dict(),
            research.to_dict(),
        )

    def test_ft_parameter_validation_rejects_invalid_keys_types_and_ranges(self):
        invalid_updates = (
            {"unexpected": 1},
            {"n_blocks": True},
            {"max_epochs": 1.5},
            {"train_batch_size": False},
            {"learning_rate": False},
            {"learning_rate": float("nan")},
            {"learning_rate": 10**1000},
            {"weight_decay": -0.1},
            {"min_delta": float("inf")},
            {"attention_dropout": 1.0},
            {"d_block": 15},
            {"ffn_d_hidden": None},
            {"ffn_d_hidden_multiplier": 1.25},
            {"validation_frequency": 2},
            {"preprocessing": "quantile"},
            {"unknown_category_policy": "reserved_embedding"},
            {"precision": "float16"},
            {"class_weighting": "balanced"},
            {"ffn_activation": "GELU"},
            {"device": "cuda"},
            {"device": "cuda:abc"},
        )
        for update in invalid_updates:
            params = {**minimal_params(), **update}
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.make_config(model_params=params)

    def test_ft_rejects_nonnull_catboost_early_stopping_and_cross_domain_validation(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_rounds must be null"):
            self.make_config(early_stopping_rounds=2)
        with self.assertRaisesRegex(ValueError, "validation_domain must equal train_domain"):
            self.make_config(validation_domain="kaggle")

    def test_ft_rejects_invalid_shared_sampling_and_prediction_settings(self):
        with self.assertRaises(ValueError):
            self.make_config(prediction_batch_size=True)
        with self.assertRaises(ValueError):
            self.make_config(train_sample_size=True)

    def test_config_and_runner_import_without_torch_or_ft_runtime(self):
        code = r"""
import sys

class BlockOptionalRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'rtdl_revisiting_models'}:
            raise AssertionError(f'optional FT runtime imported: {fullname}')
        return None

sys.meta_path.insert(0, BlockOptionalRuntime())
from src.experiments.config import ExperimentConfig
from src.experiments import runner
common = {
    'experiment_name': 'lazy_ft_config', 'train_domain': 'source',
    'validation_domain': 'source', 'test_domains': ['kaggle'],
    'feature_set': 'strict_common_16', 'seed': 42, 'threshold': 0.5,
    'calibration_bins': 5, 'output_dir': 'results/lazy-ft',
    'early_stopping_rounds': None, 'train_sample_size': None,
    'sampling_seed': None, 'prediction_batch_size': 3,
}
ft = ExperimentConfig.from_dict({
    **common, 'model': 'ft_transformer',
    'model_params': {
        'device': 'cpu', 'n_blocks': 1, 'd_block': 16,
        'attention_n_heads': 4, 'ffn_d_hidden': 32,
        'ffn_d_hidden_multiplier': None,
    },
})
catboost = ExperimentConfig.from_dict({
    **common, 'model': 'catboost', 'early_stopping_rounds': 2,
    'model_params': {},
})
assert ft.model == 'ft_transformer' and catboost.model == 'catboost'
assert callable(runner.run)
assert 'torch' not in sys.modules
assert 'rtdl_revisiting_models' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_ft_runtime_fails_only_when_constructing_ft_adapter(self):
        config = self.make_config()

        original_import = builtins.__import__

        def block_ft_package(name, *args, **kwargs):
            if name == "rtdl_revisiting_models":
                raise ModuleNotFoundError(
                    "No module named 'rtdl_revisiting_models'",
                    name="rtdl_revisiting_models",
                )
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_ft_package):
            with self.assertRaisesRegex(RuntimeError, "rtdl-revisiting-models==0.0.2"):
                runner.create_adapter(config, SCHEMA)


class FTTransformerRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.config_patch = patch.multiple(
            config_module,
            PROJECT_ROOT=self.root,
            RESULTS_DIR=self.results,
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.config = ExperimentConfig.from_dict(config_values())
        self.partitions = synthetic_partitions()

    def test_real_adapter_runner_lifecycle_and_artifacts(self):
        train, validation, kaggle, nhanes = self.partitions
        partitions = iter((train, validation, kaggle, nhanes))
        load_patch = patch.object(
            runner,
            "load_partition",
            side_effect=lambda *_args, **_kwargs: next(partitions),
        )
        original_create_adapter = runner.create_adapter
        adapters = []

        def capture_adapter(config, schema):
            adapter = original_create_adapter(config, schema)
            adapters.append(adapter)
            return adapter

        original_fit = FTTransformerAdapter.fit
        fit_calls = []

        def capture_fit(adapter, *args):
            fit_calls.append(args)
            return original_fit(adapter, *args)

        with (
            patch.object(runner.data, "load_feature_schema", return_value=SCHEMA),
            patch.object(runner, "prepared_fingerprints", return_value={"synthetic": "fixture"}),
            load_patch,
            patch.object(runner, "create_adapter", side_effect=capture_adapter),
            patch.object(FTTransformerAdapter, "fit", autospec=True, side_effect=capture_fit),
            redirect_stdout(io.StringIO()),
        ):
            output = runner.run(self.config)

        self.assertEqual(len(adapters), 1)
        adapter = adapters[0]
        self.assertIsInstance(adapter, FTTransformerAdapter)
        self.assertEqual(len(fit_calls), 1)
        self.assertEqual(len(fit_calls[0]), 4)
        expected_selected = select_training(train, 16, 42)
        pd.testing.assert_frame_equal(fit_calls[0][0], expected_selected.X)
        pd.testing.assert_series_equal(fit_calls[0][1], expected_selected.y)
        pd.testing.assert_frame_equal(fit_calls[0][2], validation.X)
        pd.testing.assert_series_equal(fit_calls[0][3], validation.y)

        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue((output / adapter.artifact_name).is_file())
        sample = pd.read_csv(output / "training_sample.csv.gz")
        self.assertEqual(
            sample["row_index"].tolist(),
            expected_selected.provenance["row_index"].tolist(),
        )
        self.assertEqual(
            metadata["sampling"]["identity_sha256"],
            expected_selected.summary["identity_sha256"],
        )
        self.assertEqual(
            metadata["sampling"]["ordered_identity_sha256"],
            expected_selected.summary["ordered_identity_sha256"],
        )

        model_metadata = metadata["model"]
        self.assertEqual(model_metadata["model_type"], "task-trained")
        self.assertTrue(model_metadata["validation_used"])
        self.assertIn("same-domain validation labels", metadata["selection_policy"])
        self.assertNotIn("pretrained", metadata["selection_policy"])
        self.assertEqual(model_metadata["training"]["training_population_size"], 16)
        self.assertEqual(
            model_metadata["training"]["validation_population_size"],
            len(validation.X),
        )
        self.assertEqual(
            model_metadata["training"]["validation_unseen_category_counts"],
            {"category": 5},
        )
        self.assertEqual(model_metadata["training"]["epochs_executed"], 2)
        self.assertIn("best_epoch", model_metadata["training"])
        self.assertIn("stopping_reason", model_metadata["training"])
        self.assertEqual(model_metadata["artifact_path"], str(output / adapter.artifact_name))
        self.assertEqual(metadata["preprocessing"], model_metadata["preprocessing"])
        self.assertEqual(metadata["compute_settings"], model_metadata["compute_settings"])
        self.assertEqual(
            metadata["environment"]["packages"]["rtdl-revisiting-models"],
            "0.0.2",
        )
        self.assertEqual(
            metadata["environment"]["packages"]["torch"],
            adapter._torch.__version__,
        )
        self.assertIsNone(metadata["resources"]["peak_gpu_memory"])
        self.assertGreater(metadata["fit_duration_seconds"], 0)

        expected_unseen = {
            "kaggle": {"category": 5},
            "nhanes": {"category": 3},
        }
        self.assertEqual(metadata["unseen_category_counts"], expected_unseen)
        self.assertEqual(set(metadata["prediction_duration_seconds"]), {"kaggle", "nhanes"})
        self.assertTrue(all(value > 0 for value in metadata["prediction_duration_seconds"].values()))

        metrics = json.loads((output / "metrics.json").read_text())
        for domain, partition in (("kaggle", kaggle), ("nhanes", nhanes)):
            predictions = pd.read_csv(output / f"predictions_{domain}_test.csv.gz")
            identity_column = "seqn" if domain == "nhanes" else "row_index"
            self.assertEqual(
                predictions[identity_column].tolist(),
                partition.provenance[identity_column].tolist(),
            )
            self.assertEqual(predictions["y_probability"].shape, (len(partition.X),))
            self.assertTrue(predictions["y_probability"].between(0, 1).all())
            recomputed = evaluate_binary(
                predictions["y_true"].to_numpy(),
                predictions["y_probability"].to_numpy(),
                self.config.threshold,
                self.config.calibration_bins,
            )
            self.assertAlmostEqual(metrics[domain]["roc_auc"], recomputed["roc_auc"])
            self.assertAlmostEqual(metrics[domain]["log_loss"], recomputed["log_loss"])

        restored = FTTransformerAdapter.load(output / adapter.artifact_name, device="cpu")
        for domain, partition in (("kaggle", kaggle), ("nhanes", nhanes)):
            predictions = pd.read_csv(output / f"predictions_{domain}_test.csv.gz")
            reloaded_probabilities = restored.predict_proba_batched(partition.X, 3)
            np.testing.assert_allclose(
                reloaded_probabilities,
                predictions["y_probability"].to_numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
        with self.assertRaises(FileExistsError):
            runner.run(self.config)

    def test_failure_after_run_creation_is_persisted_without_completed_status(self):
        load_patch = patch.object(
            runner,
            "load_partition",
            side_effect=[self.partitions[0], self.partitions[1]],
        )
        from src.models import ft_transformer

        with (
            patch.object(runner.data, "load_feature_schema", return_value=SCHEMA),
            patch.object(runner, "prepared_fingerprints", return_value={"synthetic": "fixture"}),
            load_patch,
            patch.object(
                ft_transformer.FTTransformerAdapter,
                "fit",
                side_effect=RuntimeError("controlled FT fit failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "controlled FT fit failure"),
        ):
            runner.run(self.config)
        metadata = json.loads((self.config.resolved_output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], "RuntimeError")
        self.assertEqual(metadata["error"]["message"], "controlled FT fit failure")
        self.assertFalse((self.config.resolved_output_dir / "metrics.json").exists())
        self.assertFalse(any(
            item.name.startswith("predictions_")
            for item in self.config.resolved_output_dir.iterdir()
        ))
        with self.assertRaises(FileExistsError):
            runner.run(self.config)


if __name__ == "__main__":
    unittest.main()
