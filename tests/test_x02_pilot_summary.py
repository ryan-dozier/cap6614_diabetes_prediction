import json
import tempfile
import unittest
from pathlib import Path

from scripts.x02_pilot_summary import summarize


class PilotSummaryTests(unittest.TestCase):
    def test_completed_and_unstarted_runs_with_partial_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.txt").write_text("catboost_example\nft_example\n")
            (root / "logs").mkdir()
            run = root / "runs" / "catboost_example"
            run.mkdir(parents=True)
            metadata = {
                "run_id": "catboost_example", "mode": "development", "status": "completed",
                "specification": {
                    "model": "catboost", "condition": "synthetic_only",
                    "source": "source", "budget": 0, "root_seed": 42,
                },
                "effective_configuration": {"development_training_cap": 5},
                "model": {"best_iteration": 1, "fitted_tree_count": 2},
                "label_accounting": {"synthetic_selection_used": 15000, "query_labels_for_metrics": 64},
                "operation_seconds": {"fit": 3.5},
            }
            (run / "metadata.json").write_text(json.dumps(metadata))
            (root / "logs" / "catboost_example.log").write_text("0:\ttest: 0.1\n4:\ttest: 0.2\n")
            (root / "logs" / "catboost_example.time").write_text(
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:05.67\n"
                "\tMaximum resident set size (kbytes): 12345\n"
            )
            result = summarize(root)
            completed, pending = result["runs"]
            self.assertEqual(completed["catboost_iterations_observed_in_log"], 5)
            self.assertEqual(completed["catboost_saved_trees"], 2)
            self.assertEqual(completed["time_max_rss_kib"], 12345)
            self.assertEqual(completed["time_elapsed"], "0:05.67")
            self.assertEqual(completed["label_accounting"]["synthetic_selection_used"], 15000)
            self.assertNotIn("metrics", completed)
            self.assertEqual(pending["status"], "not_started")
            self.assertIsNone(pending["time_max_rss_kib"])

    def test_failed_partial_run_and_ft_selection_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.txt").write_text("ft_example\nfailed_example\n")
            (root / "logs").mkdir()
            ft = root / "runs" / "ft_example"
            ft.mkdir(parents=True)
            (ft / "metadata.json").write_text(json.dumps({
                "run_id": "ft_example", "mode": "development", "status": "completed",
                "specification": {
                    "model": "ft_transformer", "condition": "mixed", "source": "kaggle",
                    "budget": 500, "root_seed": 42,
                },
                "model": {"training": {
                    "epochs_executed": 2, "stopping_reason": "max_epochs",
                    "training_history": [
                        {"validation_duration_seconds": 1.0}, {"validation_duration_seconds": 2.5},
                    ],
                }},
            }))
            failed = root / "runs" / "failed_example"
            failed.mkdir()
            (failed / "metadata.json").write_text(json.dumps({
                "run_id": "failed_example", "mode": "development", "status": "failed",
                "specification": {
                    "model": "tabpfn", "condition": "real_only", "source": "none",
                    "budget": 250, "root_seed": 42,
                },
                "error": {"type": "RuntimeError", "message": "fixture failure"},
            }))
            ft_result, failure = summarize(root)["runs"]
            self.assertEqual(ft_result["ft_epochs_executed"], 2)
            self.assertEqual(ft_result["ft_validation_seconds"], 3.5)
            self.assertIsNone(ft_result["resources"])
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error"]["message"], "fixture failure")

    def test_rejects_invalid_plan_and_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.txt").write_text("../invalid\n")
            with self.assertRaises(ValueError):
                summarize(root)
            (root / "plan.txt").write_text("run\n")
            run = root / "runs" / "run"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text(json.dumps({"run_id": "other", "mode": "development"}))
            with self.assertRaises(ValueError):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
