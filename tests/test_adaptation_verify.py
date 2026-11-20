from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import adaptation_verify as verify
import test_adaptation_runner


class VerificationTests(unittest.TestCase):
    def test_rejects_research_and_output_inside_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(verify, "RESULTS_DIR", Path(tmp)):
            dev = Path(tmp) / "adaptation/x01_v1/development"
            with self.assertRaisesRegex(ValueError, "development run"):
                verify.verify(Path(tmp), Path(tmp) / "adaptation/x01_v1/research/run", dev / "check")
            with self.assertRaisesRegex(ValueError, "separate"):
                verify.verify(Path(tmp), dev / "run", dev / "run" / "check")

    def test_provenance_requires_exact_order(self):
        expected = pd.DataFrame({"domain": ["nhanes", "nhanes"], "partition": ["val", "val"], "identity": [1, 2]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "query.csv"
            expected.to_csv(path, index=False)
            verify._provenance(path, expected)
            expected.iloc[::-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "ordered provenance"):
                verify._provenance(path, expected)

    def test_fake_reload_success_and_decision_failure(self):
        test_adaptation_runner.RunnerTests.setUpClass()
        fixture = test_adaptation_runner.RunnerTests()
        doc, runs = fixture.document, fixture.runs
        spec = fixture.spec("catboost", "real_only")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "adaptation/x01_v1/development/runs" / spec["run_id"]
            run.mkdir(parents=True)
            config = verify.runner._configuration(spec, None, None, 5)
            with fixture.patches()[0], fixture.patches()[1], fixture.patches()[2], fixture.patches()[3]:
                assembled = verify.runner.assemble(doc, spec, "development", 20)
            for role in ("fit", "selection", "query"):
                assembled[role][2].to_csv(run / f"{role}_identities.csv", index=False)
            predictions = assembled["query"][2].copy()
            predictions["y_true"] = assembled["query"][1]
            predictions["y_probability"] = 0.5
            predictions["y_pred"] = 1
            predictions.to_csv(run / "predictions.csv.gz", index=False)
            meta = {
                "mode": "development", "research_comparison_eligible": False,
                "matrix_sha256": verify.matrix.matrix_sha256(runs), "prepared_file_sha256": {"fixture": "hash"},
                "effective_configuration": config, "query_size_requested": 20,
                "label_accounting": assembled["label_accounting"], "feature_order": ["value"],
                "model_artifact": "fake.model", "operation_seconds": {}, "resources": {},
            }
            (run / "metadata.json").write_text(json.dumps(meta))
            output = root / "adaptation/x01_v1/development/checks/pass"
            with patch.object(verify, "RESULTS_DIR", root), \
                 patch.object(verify.matrix, "load", return_value=(doc, runs)), \
                 patch.object(verify.runner, "_inventory_status", return_value=("completed", None)), \
                 patch.object(verify.manifest, "_file_fingerprints", return_value={"fixture": "hash"}), \
                 fixture.patches()[0], fixture.patches()[1], fixture.patches()[2], fixture.patches()[3]:
                with patch.object(verify, "_reload", return_value=lambda X, batch: np.full(len(X), 0.5)):
                    report = verify.verify(Path(tmp), run, output)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["threshold_disagreements"], 0)
                with patch.object(verify, "_reload", return_value=lambda X, batch: np.full(len(X), 0.49)):
                    with self.assertRaisesRegex(ValueError, "tolerance"):
                        verify.verify(Path(tmp), run, output.parent / "fail")
                failed = json.loads((output.parent / "fail/verification.json").read_text())
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(failed["threshold_disagreements"], 20)
                with self.assertRaises(FileExistsError):
                    verify.verify(Path(tmp), run, output)


if __name__ == "__main__":
    unittest.main()
