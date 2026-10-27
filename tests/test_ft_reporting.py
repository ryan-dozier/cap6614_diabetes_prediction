from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_predictive_reporting import tabicl_fixture, _write_json, PROJECT
from src.models.ft_transformer_config import FT_DEFAULT_PARAMS
from src.reporting import validation, tables, figures, compare_predictive

MANIFEST = json.loads((PROJECT / "configs/reports/predictive_n2000_seed42_with_ft_transformer.json").read_text())


class FTReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        baseline, _ = tabicl_fixture(self.root)
        self.manifest = copy.deepcopy(MANIFEST)
        self.manifest["runs"] = baseline["runs"]
        for domain in ("source", "kaggle"):
            spec = copy.deepcopy(next(s for s in MANIFEST["runs"] if s["role"] == domain + "_ft_transformer"))
            folder = self.root / "results/predictive" / spec["role"]
            shutil.copytree(self.root / "results/predictive" / (domain + "_tabpfn"), folder)
            spec["artifact_dir"] = "results/predictive/" + spec["role"]
            self.manifest["runs"].append(spec)
            config = json.loads((folder / "config.json").read_text())
            config.update(model="ft_transformer", experiment_name=spec["run_id"],
                          output_dir=str(folder), prediction_batch_size=256,
                          model_params=copy.deepcopy(FT_DEFAULT_PARAMS))
            metadata = json.loads((folder / "metadata.json").read_text())
            metadata["configuration"] = config
            metadata["prediction_batch_size"] = 256
            architecture = {k: FT_DEFAULT_PARAMS[k] for k in (
                "n_blocks", "d_block", "attention_n_heads", "attention_dropout",
                "ffn_d_hidden", "ffn_d_hidden_multiplier", "ffn_activation",
                "ffn_dropout", "residual_dropout")}
            compute = {"device": "cuda:0", "precision": "float32",
                       "architecture": architecture, "determinism": {
                           "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                           "cudnn_benchmark": False, "cudnn_deterministic": True}}
            val = metadata["splits"][domain + "/val"]
            metadata["compute_settings"] = compute
            metadata["model"] = {
                "adapter": "FTTransformerAdapter", "ready": True,
                "model_type": "task-trained", "validation_used": True,
                "model_params": copy.deepcopy(FT_DEFAULT_PARAMS),
                "precision": "float32", "device": "cuda:0",
                "categorical_features": ["ethnicity", "gender", "smoking_status"],
                "versions": {"rtdl-revisiting-models": "0.0.2"},
                "compute_settings": compute,
                "training": {
                    "training_history": [
                        {"epoch": i, "validation_roc_auc": .7, "training_loss": .6,
                         "validation_loss": .6, "training_duration_seconds": .1,
                         "validation_duration_seconds": .1} for i in range(1, 22)],
                    "best_epoch": 1, "best_validation_roc_auc": .7,
                    "epochs_executed": 21, "patience_counter": 20,
                    "patience_reference": .7, "stopping_reason": "patience",
                    "training_population_size": 2000,
                    "validation_population_size": val["rows"],
                    "validation_class_counts": val["class_counts"],
                }}
            _write_json(folder / "config.json", config)
            _write_json(folder / "metadata.json", metadata)
        self.path = self.root / "configs/ft.json"
        _write_json(self.path, self.manifest)
        for module in (validation, compare_predictive):
            for name, value in (("PROJECT_ROOT", self.root), ("RESULTS_DIR", self.root / "results")):
                context = patch.object(module, name, value)
                context.start()
                self.addCleanup(context.stop)

    def validate(self):
        return validation.validate_comparison(self.manifest, self.path)

    def test_four_model_counts_and_training_label(self):
        runs, result = self.validate()
        self.assertEqual(len(runs), 8)
        self.assertEqual(len(tables.build_metric_rows(runs, self.manifest)), 24)
        self.assertEqual(len(tables.build_difference_rows(runs, self.manifest)), 18)
        self.assertEqual(len(tables.build_calibration_rows(runs, self.manifest)), 240)
        runtime = tables.build_runtime_rows(runs, self.manifest)
        self.assertEqual(len(runtime), 8)
        self.assertEqual([r["fit_time_kind"] for r in runtime if r["model"] == "ft_transformer"], ["fit", "fit"])
        self.assertEqual(len(result["cross_run_population_checks"]), 21)
        points = figures.performance_points(runs, self.manifest, "roc_auc")
        self.assertEqual({p["model_label"] for p in points},
                         {"CatBoost", "TabPFN", "TabICLv2", "FT-Transformer"})

    def test_rejects_ft_metadata_corruption(self):
        path = self.root / "results/predictive/source_ft_transformer/metadata.json"
        original = json.loads(path.read_text())
        for field, value in (("validation_used", False), ("precision", "float16"),
                             ("device", "cpu"), ("model_type", "pretrained")):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed["model"][field] = value
                _write_json(path, changed)
                with self.assertRaises(validation.ReportValidationError):
                    self.validate()
        for field, value in (("best_epoch", 21), ("patience_counter", 19),
                             ("stopping_reason", "max_epochs"),
                             ("validation_population_size", 999)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed["model"]["training"][field] = value
                _write_json(path, changed)
                with self.assertRaises(validation.ReportValidationError):
                    self.validate()

    def test_rejects_changed_frozen_profile(self):
        self.manifest["model_profiles"]["ft_transformer"]["package_version"] = "other"
        with self.assertRaises(validation.ReportValidationError):
            self.validate()

    def test_rejects_missing_ft_run_or_comparison(self):
        self.manifest["runs"].pop()
        with self.assertRaises(validation.ReportValidationError):
            self.validate()

    def test_report_generation_and_overwrite_protection(self):
        out = self.root / "results/reports/ft"
        compare_predictive.run_report(self.path, out, with_figures=True)
        self.assertEqual(len(list((out / "figures").glob("*.png"))), 6)
        self.assertEqual(len(list((out / "figures").glob("*.pdf"))), 6)
        report = (out / "report.md").read_text()
        self.assertIn("FT-Transformer", report)
        self.assertNotIn("six-run", report)
        before = (out / "report.md").read_bytes()
        with self.assertRaises(Exception):
            compare_predictive.run_report(self.path, out)
        self.assertEqual((out / "report.md").read_bytes(), before)
