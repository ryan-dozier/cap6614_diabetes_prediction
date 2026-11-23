from __future__ import annotations

import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from src.reporting import adaptation as report
import test_adaptation_runner


class BootstrapTests(unittest.TestCase):
    def test_stratified_draws_and_independent_effects(self):
        labels = np.array([0, 0, 1, 1])
        first = report.stratified_indices(labels, 20, report.BOOTSTRAP_SEED)
        np.testing.assert_array_equal(first, report.stratified_indices(labels, 20, report.BOOTSTRAP_SEED))
        self.assertTrue(np.all(labels[first].sum(axis=1) == 2))
        self.assertTrue(np.all(labels[first[:, :2]] == 0))
        mixed = np.array([[0.1, 0.2, 0.8, 0.9]])
        baseline = np.full((1, 4), 0.5)
        fixed = np.array([[0, 1, 2, 3], [0, 0, 3, 3]])
        actual = report.bootstrap_effects(labels, mixed, baseline, fixed)
        expected = np.array([
            [0.5, 0.5, 0.225, np.log(0.5) * -1 + np.log(0.9 * 0.8 * 0.8 * 0.9) / 4],
            [0.5, 0.5, 0.24, np.log(0.5) * -1 + np.log(0.9)],
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(np.quantile(actual[:, 2], [0.025, 0.975], method="linear"),
                                   [0.225375, 0.239625])
        np.testing.assert_array_equal(report.bootstrap_effects(labels, mixed, mixed, fixed), np.zeros((2, 4)))

    def test_prediction_alignment_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            frame = pd.DataFrame({"domain": ["nhanes"] * 4, "partition": ["test"] * 4,
                                  "identity": [4, 2, 1, 3], "y_true": [1, 0, 0, 1],
                                  "y_probability": [0.8, 0.3, 0.2, 0.7], "y_pred": [1, 0, 0, 1]})
            frame.to_csv(path, index=False)
            self.assertEqual(report._aligned_predictions(path, [1, 2, 3, 4])["identity"].tolist(), [1, 2, 3, 4])
            frame.loc[0, "identity"] = 2
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                report._aligned_predictions(path, [1, 2, 3, 4])
            frame = frame.iloc[1:]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                report._aligned_predictions(path, [1, 2, 3, 4])
            frame = pd.DataFrame({"domain": ["nhanes"] * 4, "partition": ["test"] * 4,
                                  "identity": [1, 2, 3, 4], "y_true": [0, 0, 1, 1],
                                  "y_probability": [0.2, 0.3, 0.7, 0.8], "y_pred": [1, 0, 1, 1]})
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "threshold decisions"):
                report._aligned_predictions(path, [1, 2, 3, 4])
            frame.loc[0, "y_probability"] = 1.5
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                report._aligned_predictions(path, [1, 2, 3, 4])


class ReportFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_adaptation_runner.RunnerTests.setUpClass()
        cls.fixture = test_adaptation_runner.RunnerTests()
        cls.doc, cls.runs = cls.fixture.document, cls.fixture.runs

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_root = self.root / "adaptation/x01_v1/research/artificial"
        self.run_root.mkdir(parents=True)
        self.manifest_dir = self.root / "manifest"
        self.manifest_dir.mkdir()
        (self.manifest_dir / "metadata.json").write_text(json.dumps({"source_file_sha256": {"fixture": "hash"}}))
        self.ids = list(range(100000, 100646))
        self.labels = np.array([0] * 545 + [1] * 101)
        approved = [s for s in self.runs if s["budget"] in (0, 250)]
        self.freeze = {
            "schema_version": "x02_freeze_v1", "status": "approved",
            "protocol_version": matrix.PROTOCOL,
            "protocol_sha256": manifest._sha256_file(runner.PROJECT_ROOT / "docs/plans/x01_adaptation_protocol.md"),
            "evaluation_protocol_sha256": manifest._sha256_file(runner.PROJECT_ROOT / "docs/plans/e05_evaluation_protocol.md"),
            "matrix_sha256": matrix.matrix_sha256(self.runs), "approved_budgets": [250],
            "approved_run_ids": [s["run_id"] for s in approved], "approved_by": "ARTIFICIAL FIXTURE",
            "approved_at": "fixture", "approval_reference": "fixture",
            "feasibility_decision": "fixture", "feasibility_evidence": "fixture",
        }
        self.freeze_path = self.root / "ARTIFICIAL_FREEZE.json"
        self.freeze_path.write_text(json.dumps(self.freeze))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(matrix, "load", return_value=(self.doc, self.runs)))
        self.stack.enter_context(patch.object(runner, "RESULTS_DIR", self.root))
        self.stack.enter_context(patch.object(report.data, "load_feature_schema", return_value=test_adaptation_runner.SCHEMA))
        self.stack.enter_context(patch.object(report, "BOOTSTRAP_REPLICATES", 3))
        self.stack.enter_context(patch.object(runner, "_assignments", side_effect=self.assignments))
        self.stack.enter_context(patch.object(runner, "_read_rows", side_effect=AssertionError("feature loader forbidden")))
        self.stack.enter_context(patch.object(runner, "_whole_partition", side_effect=AssertionError("feature loader forbidden")))

    def assignments(self, domain):
        if domain == "nhanes":
            return pd.DataFrame({"seqn": self.ids, "split": ["test"] * len(self.ids)})
        return pd.DataFrame({"row_index": [9000, 9001], "split": ["val", "val"]})

    def write_run(self, spec, *, reverse=False, conflict=False, constant=False, fixture_probabilities=None):
        path = self.run_root / spec["run_id"]
        path.mkdir()
        fit, selection = report._expected_provenance(self.doc, spec)
        query = pd.DataFrame({"domain": "nhanes", "partition": "test", "identity": self.ids})
        for role, frame in (("fit", fit), ("selection", selection), ("query", query)):
            frame.to_csv(path / f"{role}_identities.csv", index=False)
        y = self.labels.copy()
        if conflict:
            y[0], y[-1] = y[-1], y[0]
        # Change seed-specific probabilities; metrics are computed before averaging.
        p = np.where(y == 1, 0.7, 0.3).astype(float)
        if spec["condition"] == "real_only" and spec["root_seed"] % 2:
            p = 1 - p
        if constant:
            p[:] = 0.5
        if fixture_probabilities is not None:
            p = np.asarray(fixture_probabilities, dtype=float)
        predictions = query.copy()
        predictions["y_true"] = y
        predictions["y_probability"] = p
        predictions["y_pred"] = (p >= 0.5).astype(int)
        if reverse:
            predictions = predictions.iloc[::-1]
        predictions.to_csv(path / "predictions.csv.gz", index=False)
        (path / "fake.model").write_text("artificial")
        (path / "metrics.json").write_text("{}")
        (path / "run_specification.json").write_text(json.dumps(spec))
        real_fit = int((fit["domain"] == "nhanes").sum())
        real_selection = int((selection["domain"] == "nhanes").sum())
        accounting = {"nhanes_available": spec["budget"], "nhanes_fit_or_context_used": real_fit,
                      "nhanes_selection_used": real_selection,
                      "nhanes_reserved_unused": spec["budget"] - real_fit - real_selection,
                      "synthetic_fit_used": len(fit) - real_fit, "synthetic_selection_used": len(selection) - real_selection,
                      "query_labels_for_metrics": len(y), "query_label_category": "research_evaluation"}
        meta = {"status": "completed", "mode": "research", "research_comparison_eligible": True,
                "run_id": spec["run_id"], "specification": spec, "matrix_sha256": matrix.matrix_sha256(self.runs),
                "freeze": self.freeze, "prepared_file_sha256": {"fixture": "hash"},
                "effective_configuration": runner._configuration(spec, None, None, None),
                "feature_order": ["value"], "label_accounting": accounting, "model_artifact": "fake.model",
                "model": {}, "output_sha256": {p.name: manifest._sha256_file(p) for p in path.iterdir()}}
        (path / "metadata.json").write_text(json.dumps(meta))
        return path

    def generate(self, name):
        return report.generate(self.manifest_dir, self.freeze_path, self.run_root, self.root / name)

    def test_empty_incomplete_report_and_no_overwrite(self):
        output = self.generate("ARTIFICIAL_empty_report")
        self.assertIn("Completed runs: 0", (output / "report.md").read_text())
        self.assertEqual(len(pd.read_csv(output / "summary.csv")), 0)
        self.assertTrue((pd.read_csv(output / "comparisons.csv")["status"] == "incomplete").all())
        with self.assertRaises(FileExistsError):
            self.generate("ARTIFICIAL_empty_report")

    def test_completed_pairs_reuse_summary_and_missing(self):
        specs = [s for s in self.runs if s["model"] == "catboost" and s["budget"] == 250 and
                 (s["condition"] == "real_only" or (s["condition"] == "mixed" and s["source"] == "source"))]
        for spec in specs:
            self.write_run(spec, reverse=spec["root_seed"] % 2 == 0)
        output = self.generate("ARTIFICIAL_complete_report")
        rows = pd.read_csv(output / "run_metrics.csv")
        self.assertEqual(len(rows), 10)
        pairs = pd.read_csv(output / "paired_differences.csv")
        summary = pd.read_csv(output / "summary.csv")
        self.assertEqual(len(summary), 4)
        auc = summary.loc[summary["metric"] == "roc_auc"].iloc[0]
        self.assertAlmostEqual(auc["mean_improvement"], 0.4)
        self.assertAlmostEqual(auc["comparator_mean"], 0.6)
        self.assertAlmostEqual(auc["ci_lower"], 0.4)
        self.assertAlmostEqual(auc["ci_upper"], 0.4)
        brier = summary.loc[summary["metric"] == "brier_score"].iloc[0]
        self.assertAlmostEqual(brier["mean_improvement"], 0.16)
        self.assertAlmostEqual(brier["ci_lower"], 0.16)
        self.assertAlmostEqual(brier["ci_upper"], 0.16)
        # The average probability across these seeds has AUC 1, not the mean AUC 0.6.
        ensemble = np.where(self.labels == 1, 0.54, 0.46)
        self.assertEqual(report._four_metrics(self.labels, ensemble)[0], 1.0)
        self.assertAlmostEqual(auc["improvement_sd"], np.std([0, 1, 0, 1, 0], ddof=1))
        self.assertEqual(len(pairs.loc[pairs["status"] == "complete"]), 20)
        self.assertEqual(len(pd.read_csv(output / "bootstrap_effects.csv")), 12)
        self.assertTrue((output / "bootstrap_indices.npy").is_file())
        incomplete = pd.read_csv(output / "comparisons.csv")
        self.assertEqual((incomplete["status"] == "complete").sum(), 1)
        self.assertIn("conditional", (output / "report.md").read_text())
        # Four-seed contrasts have no summary or confidence interval.
        (self.run_root / specs[-1]["run_id"] / "metadata.json").unlink()
        partial = self.generate("ARTIFICIAL_partial_report")
        self.assertEqual(len(pd.read_csv(partial / "summary.csv")), 0)
        self.assertEqual(len(pd.read_csv(partial / "paired_differences.csv")), 16)

    def test_tampering_development_and_conflicting_labels_rejected(self):
        specs = [s for s in self.runs if s["model"] == "catboost" and s["condition"] == "real_only" and s["budget"] == 250]
        p = self.write_run(specs[0])
        (p / "fake.model").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_tamper")
        self.assertIn("fingerprint", (self.root / "ARTIFICIAL_tamper/inventory.csv").read_text())
        (p / "fake.model").write_text("artificial")
        meta = json.loads((p / "metadata.json").read_text())
        meta["mode"] = "development"
        (p / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_development")
        meta["mode"] = "research"
        meta["effective_configuration"]["model_params"]["depth"] = 3
        (p / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_config")
        meta["effective_configuration"] = runner._configuration(specs[0], None, None, None)
        (p / "metadata.json").write_text(json.dumps(meta))
        self.write_run(specs[1], conflict=True)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_conflict")
        self.assertIn("labels conflict", (self.root / "ARTIFICIAL_conflict/inventory.csv").read_text())

    def test_artifact_provenance_and_freeze_are_checked(self):
        spec = next(s for s in self.runs if s["model"] == "catboost" and s["condition"] == "mixed"
                    and s["source"] == "source" and s["budget"] == 250 and s["root_seed"] == 42)
        path = self.write_run(spec)
        meta = json.loads((path / "metadata.json").read_text())
        meta["prepared_file_sha256"] = {"fixture": "stale"}
        (path / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_stale")
        self.assertIn("fingerprints differ", (self.root / "ARTIFICIAL_stale/inventory.csv").read_text())
        meta["prepared_file_sha256"] = {"fixture": "hash"}
        meta["freeze"] = dict(self.freeze, approved_by="different")
        (path / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_freeze_mismatch")
        meta["freeze"] = self.freeze
        fit = pd.read_csv(path / "fit_identities.csv")
        fit.loc[0, "identity"] = 777777
        fit.to_csv(path / "fit_identities.csv", index=False)
        meta["output_sha256"]["fit_identities.csv"] = manifest._sha256_file(path / "fit_identities.csv")
        (path / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.generate("ARTIFICIAL_bad_assignments")
        self.assertIn("identities differ", (self.root / "ARTIFICIAL_bad_assignments/inventory.csv").read_text())

    def test_anchor_reused_across_budgets_without_duplicate_run_metrics(self):
        self.freeze["approved_budgets"] = [250, 500]
        self.freeze["approved_run_ids"] = [s["run_id"] for s in self.runs if s["budget"] in (0, 250, 500)]
        self.freeze_path.write_text(json.dumps(self.freeze))
        chosen = [s for s in self.runs if s["model"] == "catboost" and (
            (s["condition"] == "real_only" and s["budget"] in (250, 500)) or
            (s["condition"] == "mixed" and s["source"] == "source" and s["budget"] in (250, 500)) or
            (s["condition"] == "synthetic_only" and s["source"] == "source"))]
        for spec in chosen:
            self.write_run(spec)
        output = self.generate("ARTIFICIAL_reuse")
        self.assertEqual(len(pd.read_csv(output / "run_metrics.csv")), 25)
        comparisons = pd.read_csv(output / "comparisons.csv")
        self.assertEqual((comparisons["status"] == "complete").sum(), 4)
        pairs = pd.read_csv(output / "paired_differences.csv")
        anchors = pairs.loc[pairs["contrast"] == "synthetic_only"]
        self.assertEqual(anchors["comparator_run_id"].nunique(), 5)
        self.assertEqual(set(anchors["budget"]), {250, 500})
        mixed = pairs.loc[pairs["contrast"] == "real_only", "mixed_run_id"]
        self.assertEqual(mixed.nunique(), 10)

    def test_allowed_device_override_is_not_treated_as_scientific_change(self):
        spec = next(s for s in self.runs if s["model"] == "ft_transformer" and s["condition"] == "real_only"
                    and s["budget"] == 250 and s["root_seed"] == 42)
        path = self.write_run(spec)
        meta = json.loads((path / "metadata.json").read_text())
        meta["effective_configuration"] = runner._configuration(spec, "cuda:7", None, None)
        (path / "metadata.json").write_text(json.dumps(meta))
        output = self.generate("ARTIFICIAL_device_override")
        self.assertEqual(len(pd.read_csv(output / "run_metrics.csv")), 1)

    def test_full_prescribed_bootstrap_report_on_small_artificial_cohort(self):
        self.ids = [100000, 100001, 100002, 100003]
        self.labels = np.array([0, 0, 1, 1])
        specs = [s for s in self.runs if s["model"] == "catboost" and s["budget"] == 250 and
                 (s["condition"] == "real_only" or (s["condition"] == "mixed" and s["source"] == "source"))]
        for spec in specs:
            self.write_run(spec, constant=True)
        with patch.object(report, "BOOTSTRAP_REPLICATES", 2000), patch.object(report, "COHORT_COUNTS", (2, 2)):
            output = self.generate("ARTIFICIAL_full_2000")
        indices = np.load(output / "bootstrap_indices.npy")
        self.assertEqual(indices.shape, (2000, 4))
        self.assertTrue(np.all(self.labels[indices].sum(axis=1) == 2))
        effects = pd.read_csv(output / "bootstrap_effects.csv")
        self.assertEqual(len(effects), 8000)
        self.assertTrue((effects["mean_improvement"] == 0).all())
        summary = pd.read_csv(output / "summary.csv")
        self.assertTrue((summary[["mean_improvement", "ci_lower", "ci_upper"]] == 0).all().all())

    def test_varying_paired_bootstrap_report_against_independent_calculation(self):
        self.ids = list(range(100000, 100006))
        self.labels = np.array([0, 0, 0, 1, 1, 1])
        mixed = [
            [0.12, 0.65, 0.28, 0.82, 0.47, 0.54],
            [0.21, 0.55, 0.35, 0.72, 0.44, 0.67],
            [0.33, 0.61, 0.18, 0.52, 0.74, 0.43],
            [0.45, 0.24, 0.62, 0.68, 0.38, 0.79],
            [0.17, 0.72, 0.31, 0.64, 0.51, 0.41],
        ]
        comparator = [
            [0.35, 0.44, 0.70, 0.53, 0.76, 0.22],
            [0.41, 0.18, 0.58, 0.32, 0.69, 0.75],
            [0.26, 0.49, 0.65, 0.77, 0.36, 0.57],
            [0.61, 0.32, 0.16, 0.43, 0.73, 0.56],
            [0.39, 0.28, 0.63, 0.71, 0.24, 0.59],
        ]
        draws = np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 0, 1, 3, 3, 4],
            [2, 2, 1, 5, 5, 4],
            [1, 1, 0, 4, 4, 3],
            [0, 2, 2, 3, 5, 5],
            [2, 1, 1, 5, 3, 3],
        ], dtype=np.int64)

        def independent_metrics(scores, draw):
            observations = [(int(self.labels[i]), scores[i]) for i in draw]
            positives = [p for y, p in observations if y == 1]
            negatives = [p for y, p in observations if y == 0]
            auc = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
            auc /= len(positives) * len(negatives)
            # AP integrates precision at each distinct positive score threshold.
            ap = 0.0
            for threshold in set(positives):
                relevant = sum(p == threshold for p in positives)
                tp = sum(p >= threshold for p in positives)
                fp = sum(p >= threshold for p in negatives)
                ap += relevant / len(positives) * tp / (tp + fp)
            brier = sum((p - y) ** 2 for y, p in observations) / len(observations)
            loss = -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                        for y, p in observations) / len(observations)
            return np.array([auc, ap, brier, loss])

        def improvement(draw_a, draw_b):
            values = [independent_metrics(a, draw_a) - independent_metrics(b, draw_b)
                      for a, b in zip(mixed, comparator)]
            return np.mean(values, axis=0) * [1, 1, -1, -1]

        def linear_percentile(values, q):
            ordered = sorted(values)
            position = (len(ordered) - 1) * q
            lower = math.floor(position)
            upper = math.ceil(position)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

        specs = [s for s in self.runs if s["model"] == "catboost" and s["budget"] == 250 and
                 (s["condition"] == "real_only" or (s["condition"] == "mixed" and s["source"] == "source"))]
        for spec in specs:
            index = spec["root_seed"] - 42
            scores = mixed[index] if spec["condition"] == "mixed" else comparator[index]
            self.write_run(spec, reverse=index % 2 == 0, fixture_probabilities=scores)

        expected = np.stack([improvement(draw, draw) for draw in draws])
        point = improvement(draws[0], draws[0])
        ensemble_mixed = np.mean(mixed, axis=0)
        ensemble_comparator = np.mean(comparator, axis=0)
        ensemble_effect = (independent_metrics(ensemble_mixed, draws[0]) -
                           independent_metrics(ensemble_comparator, draws[0])) * [1, 1, -1, -1]
        self.assertTrue(np.all(np.abs(point - ensemble_effect) > 1e-6))
        self.assertTrue(np.all(np.abs(point) > 1e-6))
        unpaired = np.stack([improvement(draw, draws[(i + 1) % len(draws)])
                             for i, draw in enumerate(draws)])
        self.assertFalse(np.allclose(expected, unpaired))
        shifted_seed_effects = np.stack([
            np.mean([(independent_metrics(a, draw) -
                      independent_metrics(b, draws[(i + seed + 1) % len(draws)])) * [1, 1, -1, -1]
                     for seed, (a, b) in enumerate(zip(mixed, comparator))], axis=0)
            for i, draw in enumerate(draws)
        ])
        self.assertFalse(np.allclose(expected, shifted_seed_effects))

        with patch.object(report, "BOOTSTRAP_REPLICATES", len(draws)), \
             patch.object(report, "COHORT_COUNTS", (3, 3)), \
             patch.object(report, "stratified_indices", return_value=draws) as sample:
            output = self.generate("ARTIFICIAL_varying_bootstrap")
        sample.assert_called_once()
        np.testing.assert_array_equal(np.load(output / "bootstrap_indices.npy"), draws)
        effects = pd.read_csv(output / "bootstrap_effects.csv")
        summary = pd.read_csv(output / "summary.csv").set_index("metric")
        self.assertEqual(len(effects), len(draws) * 4)
        for column, metric in enumerate(("roc_auc", "average_precision", "brier_score", "log_loss")):
            actual = effects.loc[effects["metric"] == metric].sort_values("replicate")
            self.assertEqual(actual["replicate"].tolist(), list(range(len(draws))))
            np.testing.assert_allclose(actual["mean_improvement"], expected[:, column], rtol=0, atol=1e-12)
            self.assertAlmostEqual(summary.loc[metric, "mean_improvement"], point[column], places=12)
            self.assertAlmostEqual(summary.loc[metric, "ci_lower"], linear_percentile(expected[:, column], 0.025), places=12)
            self.assertAlmostEqual(summary.loc[metric, "ci_upper"], linear_percentile(expected[:, column], 0.975), places=12)
        self.assertTrue((summary["ci_upper"] > summary["ci_lower"]).all())


if __name__ == "__main__":
    unittest.main()
