from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data import DatasetSchema
from src.experiments import adaptation_launch as launch
from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from test_adaptation_manifest import inputs

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = DatasetSchema("diagnosed_diabetes", ("value",), (), ("value",), ())
TEST_IDS = [9001, 9002, 9003, 9004]


def fake_assignments(domain):
    if domain == "nhanes":
        return pd.DataFrame({"seqn": TEST_IDS, "split": ["test"] * len(TEST_IDS)})
    return pd.DataFrame({"row_index": [7001, 7002, 7003], "split": ["val"] * 3})


class LaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = manifest.build_assignments(inputs(), {})
        cls.runs = matrix.build(cls.document, "a" * 64)
        cls.by_id = {run["run_id"]: run for run in cls.runs}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        for name in ("docs/plans/x01_adaptation_protocol.md", "docs/plans/e05_evaluation_protocol.md",
                     "scripts/x03_newton_cpu.sbatch", "scripts/x03_newton_gpu.sbatch",
                     "scripts/x03_newton_common.sh", "src/data.py", "src/paths.py",
                     "src/domain_labels.py", "src/experiments/adaptation_launch.py"):
            target = self.project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, target)
        self.evidence = self.project / "docs/results/x02_stage2e_feasibility.md"
        self.evidence.parent.mkdir(parents=True)
        self.evidence.write_text("fixture feasibility evidence")
        self.canonical = self.project / "canonical.csv"
        self.canonical.write_text("opaque bytes; no examples may be loaded")
        self.manifest_dir = self.project / "manifest"
        self.manifest_dir.mkdir()
        (self.manifest_dir / "assignments.json").write_text("fixture")
        (self.manifest_dir / "metadata.json").write_text(json.dumps({"source_file_sha256": {"fixture": "hash"}}))
        (self.project / "configs").mkdir()
        for name, _ in matrix.PROFILES.values():
            shutil.copy(ROOT / "configs" / name, self.project / "configs" / name)
        self.freeze_path = self.project / "freeze.json"
        self.freeze = {
            "schema_version": "x02_freeze_v1", "status": "approved",
            "protocol_version": matrix.PROTOCOL,
            "protocol_sha256": manifest._sha256_file(self.project / "docs/plans/x01_adaptation_protocol.md"),
            "evaluation_protocol_sha256": manifest._sha256_file(self.project / "docs/plans/e05_evaluation_protocol.md"),
            "matrix_sha256": matrix.matrix_sha256(self.runs),
            "approved_by": "Fixture operator", "approved_at": "fixture timestamp",
            "feasibility_decision": "fixture", "feasibility_evidence": "docs/results/x02_stage2e_feasibility.md",
            "approval_reference": "fixture", "approved_budgets": [250, 500, 1000],
            "approved_run_ids": [run["run_id"] for run in self.runs],
            "manifest_sha256": "a" * 64,
            "manifest_metadata_sha256": manifest._sha256_file(self.manifest_dir / "metadata.json"),
            "reference_profile_sha256": {name: digest for name, digest in matrix.PROFILES.values()},
            "checkpoint_sha256": matrix.CHECKPOINTS,
            "feasibility_evidence_sha256": manifest._sha256_file(self.evidence),
        }
        self.save_freeze()
        self.run_root = self.project / "results/adaptation/x01_v1/research/runs"
        self.job_dir = self.project / "execution/jobs/fixture_99"
        self.plan_path = self.project / "plan.json"
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(launch, "PROJECT_ROOT", self.project))
        stack.enter_context(patch.object(runner, "PROJECT_ROOT", self.project))
        stack.enter_context(patch.object(runner, "RESULTS_DIR", self.project / "results"))
        stack.enter_context(patch.object(matrix, "load", return_value=(self.document, self.runs)))
        stack.enter_context(patch.object(manifest, "_input_paths", return_value=[self.canonical]))
        stack.enter_context(patch.object(runner.data, "load_feature_schema", return_value=SCHEMA))
        stack.enter_context(patch.object(runner, "_assignments", side_effect=fake_assignments))
        stack.enter_context(patch.dict(os.environ, {
            "SLURM_JOB_ID": "99", "SLURM_CPUS_PER_TASK": "4",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "", "PYTHONHOME": "",
        }))

    def save_freeze(self):
        self.freeze_path.write_text(json.dumps(self.freeze))

    def plan(self, model="catboost", **kwargs):
        return launch.make_plan(
            project=self.project, python=Path(sys.executable), manifest_dir=self.manifest_dir,
            freeze_path=self.freeze_path, run_root=self.run_root, model=model, **kwargs,
        )

    def export(self, plan):
        self.plan_path.write_bytes(manifest._json_bytes(plan))

    def write_completed_run(self, spec, mutate=None):
        """Build a research run directory satisfying the full artifact contract."""
        output = self.run_root / spec["run_id"]
        output.mkdir(parents=True)
        fit, selection = launch._expected_provenance(self.document, spec)
        query = pd.DataFrame({"domain": "nhanes", "partition": "test", "identity": TEST_IDS})
        fit.to_csv(output / "fit_identities.csv", index=False)
        selection.to_csv(output / "selection_identities.csv", index=False)
        query.to_csv(output / "query_identities.csv", index=False)
        (output / "predictions.csv.gz").write_bytes(b"opaque fixture prediction bytes")
        (output / "metrics.json").write_text(json.dumps({"mode": "research"}))
        artifact = "fake.model"
        (output / artifact).write_text(f"fixture artifact for {spec['run_id']}")
        (output / "run_specification.json").write_bytes(manifest._json_bytes(spec))
        real_fit = int((fit["domain"] == "nhanes").sum())
        real_selection = int((selection["domain"] == "nhanes").sum())
        meta = {
            "run_id": spec["run_id"], "mode": "research", "status": "completed",
            "research_comparison_eligible": True, "specification": spec,
            "matrix_sha256": matrix.matrix_sha256(self.runs),
            "freeze": json.loads(self.freeze_path.read_text()),
            "prepared_file_sha256": {"fixture": "hash"},
            "effective_configuration": runner._configuration(spec, None, None, None),
            "feature_order": ["value"],
            "label_accounting": {
                "nhanes_available": spec["budget"], "nhanes_fit_or_context_used": real_fit,
                "nhanes_selection_used": real_selection,
                "nhanes_reserved_unused": spec["budget"] - real_fit - real_selection,
                "synthetic_fit_used": len(fit) - real_fit,
                "synthetic_selection_used": len(selection) - real_selection,
                "query_labels_for_metrics": len(TEST_IDS),
                "query_label_category": "research_evaluation",
            },
            "model_artifact": artifact, "model": {"adapter": "fixture"},
            "code_revision": {"commit": "fixture-revision", "worktree_dirty": False},
        }
        if mutate:
            mutate(output, meta)
        required = ["run_specification.json", "fit_identities.csv", "selection_identities.csv",
                    "query_identities.csv", "predictions.csv.gz", "metrics.json", artifact]
        meta["output_sha256"] = {name: manifest._sha256_file(output / name) for name in required}
        (output / "metadata.json").write_text(json.dumps(meta))
        return output

    def rehash(self, output):
        meta = json.loads((output / "metadata.json").read_text())
        meta["output_sha256"] = {name: manifest._sha256_file(output / name) for name in meta["output_sha256"]}
        (output / "metadata.json").write_text(json.dumps(meta))

    def catboost_specs(self):
        return [run for run in self.runs if run["model"] == "catboost"]

    def classify(self, spec):
        freeze = json.loads(self.freeze_path.read_text())
        return launch._classify(self.run_root / spec["run_id"], spec, self.document, self.runs,
                                freeze, {"fixture": "hash"}, launch._lazy_test_ids())

    def test_full_grid_counts_unique_anchors_commands_and_determinism(self):
        with patch.object(runner, "_verify_checkpoint"):
            plan = self.plan("all", tabpfn_checkpoint=self.canonical, tabicl_checkpoint=self.canonical)
            again = self.plan("all", tabpfn_checkpoint=self.canonical, tabicl_checkpoint=self.canonical)
        self.assertEqual(plan, again)
        self.assertEqual(len(plan["runs"]), 220)
        self.assertEqual(len({run["run_id"] for run in plan["runs"]}), 220)
        self.assertFalse(self.run_root.exists())
        for model in manifest.MODELS:
            selected = [run for run in plan["runs"] if run["specification"]["model"] == model]
            self.assertEqual(len(selected), 55)
            self.assertEqual(Counter(run["specification"]["condition"] for run in selected),
                             {"real_only": 15, "mixed": 30, "synthetic_only": 10})
            anchors = [run["specification"] for run in selected
                       if run["specification"]["condition"] == "synthetic_only"]
            self.assertEqual(len({(run["root_seed"], run["source"]) for run in anchors}), 10)
            self.assertTrue(all(run["budget"] == 0 for run in anchors))
        for run in plan["runs"]:
            command = run["command"]
            self.assertEqual(command[command.index("--mode") + 1], "research")
            self.assertEqual(command[command.index("--freeze") + 1], str(self.freeze_path))
            self.assertNotIn("--training-cap", command)
            self.assertIsNone(run["effective_configuration"]["development_training_cap"])

    def test_plan_only_never_loads_examples_or_models(self):
        with patch.object(runner, "assemble", side_effect=AssertionError("assembly forbidden")), \
             patch.object(runner, "_read_rows", side_effect=AssertionError("examples forbidden")), \
             patch.object(runner, "_whole_partition", side_effect=AssertionError("test loader forbidden")), \
             patch.object(runner, "_adapter", side_effect=AssertionError("model forbidden")), \
             patch.object(runner, "execute", side_effect=AssertionError("execution forbidden")):
            self.assertEqual(len(self.plan()["runs"]), 55)
            self.assertEqual(len(self.plan(resume=True)["runs"]), 55)
        subprocess.run([sys.executable, "-c",
                        "import sys; import src.experiments.adaptation_launch; "
                        "assert not set(sys.modules).intersection({'torch','catboost','tabpfn','tabicl','sklearn'})"],
                       check=True, cwd=ROOT)

    def test_rejects_draft_stale_and_incompatible_freezes(self):
        for key, value in (
            ("status", "draft"), ("protocol_sha256", "stale"), ("matrix_sha256", "stale"),
            ("manifest_metadata_sha256", "stale"), ("approved_budgets", [250]),
            ("reference_profile_sha256", {}), ("checkpoint_sha256", {}),
            ("approved_run_ids", self.freeze["approved_run_ids"][:-1]),
        ):
            with self.subTest(key=key):
                original = self.freeze[key]
                self.freeze[key] = value
                self.save_freeze()
                with self.assertRaises(ValueError):
                    self.plan()
                self.freeze[key] = original
        self.save_freeze()

    def test_checkpoint_override_only_changes_deployment(self):
        second = self.project / "renamed.checkpoint"
        shutil.copy(self.canonical, second)
        with patch.object(runner, "_verify_checkpoint") as verify:
            first = self.plan("tabpfn", tabpfn_checkpoint=self.canonical)
            other = self.plan("tabpfn", tabpfn_checkpoint=second)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(first["runs"][0]["specification"], other["runs"][0]["specification"])
        a = copy.deepcopy(first["runs"][0]["effective_configuration"])
        b = copy.deepcopy(other["runs"][0]["effective_configuration"])
        a["model_params"].pop("model_path")
        b["model_params"].pop("model_path")
        self.assertEqual(a, b)
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.plan("tabpfn")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            self.plan("tabpfn", tabpfn_checkpoint=second)

    def test_existing_outputs_refused_without_overwrite(self):
        plan = self.plan()
        output = Path(plan["runs"][10]["output_path"])
        output.mkdir(parents=True)
        marker = output / "retain"
        marker.write_text("original")
        current = self.plan()
        self.assertEqual(current["runs"][10]["output_status"], "existing")
        self.export(plan)
        with patch.object(subprocess, "run") as run, self.assertRaisesRegex(FileExistsError, output.name):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), "original")
        self.assertFalse(self.job_dir.exists())

    def test_changed_plan_code_inputs_and_model_are_rejected(self):
        plan = self.plan()
        bad = copy.deepcopy(plan)
        bad["runs"][0]["command"].append("--training-cap")
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.validate_plan(bad, "catboost")
        with self.assertRaisesRegex(ValueError, "submitted model"):
            launch.validate_plan(plan, "tabicl")
        self.canonical.write_text("changed input")
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.validate_plan(plan, "catboost")

    def test_layout_rejects_unexpected_entries_and_nested_logs(self):
        self.run_root.mkdir(parents=True)
        (self.run_root / "plans").mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.plan()
        with self.assertRaisesRegex(ValueError, "separate"):
            launch.outside_runs(self.run_root / "logs", self.run_root)
        with self.assertRaisesRegex(ValueError, "separate"):
            launch.outside_runs(self.run_root / "plan.json", self.run_root)
        with self.assertRaisesRegex(ValueError, "inside"):
            runner._output_path(self.project / "outside", "research", "fixture")

    def test_execution_sequence_failure_propagation_and_external_logs(self):
        plan = self.plan()
        self.export(plan)
        calls = []

        def fake_run(command, **kwargs):
            if command[0] == "/usr/bin/time":
                calls.append(command)
                if len(calls) == 2:
                    raise subprocess.CalledProcessError(7, command)
                run_id = command[command.index("--run-id") + 1]
                self.write_completed_run(self.by_id[run_id])
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run), \
             self.assertRaises(subprocess.CalledProcessError) as error:
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        self.assertEqual(error.exception.returncode, 7)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--training-cap", calls[0])
        self.assertTrue((self.job_dir / "plan.json").is_file())
        status = json.loads((self.job_dir / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_run_ids"], [plan["runs"][0]["run_id"]])
        self.assertEqual(status["current_run_id"], plan["runs"][1]["run_id"])
        self.assertEqual(status["invalid_run_ids"], [])
        self.assertEqual(status["remaining_run_ids"], [run["run_id"] for run in plan["runs"][1:]])

    def test_gpu_preflight_and_cpu_does_not_require_gpu(self):
        for model in ("catboost", "ft_transformer"):
            with self.subTest(model=model):
                self.export(self.plan(model))
                with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, \
                     patch.object(launch, "_classify", return_value=("verified_complete", None)):
                    launch.execute_plan(self.plan_path, model, self.job_dir / model)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(sum(command[0] == "/usr/bin/time" for command in commands), 55)
                self.assertEqual(any(command[0] == "nvidia-smi" for command in commands), model != "catboost")
                self.assertEqual(any("import torch" in part for command in commands for part in command),
                                 model != "catboost")

    def test_failed_gpu_preflight_retains_evidence_without_starting_runs(self):
        self.export(self.plan("ft_transformer"))
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[0] == "nvidia-smi":
                raise subprocess.CalledProcessError(6, command)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run), \
             self.assertRaises(subprocess.CalledProcessError) as error:
            launch.execute_plan(self.plan_path, "ft_transformer", self.job_dir)
        self.assertEqual(error.exception.returncode, 6)
        self.assertFalse(any(command[0] == "/usr/bin/time" for command in commands))
        self.assertTrue((self.job_dir / "job.log").exists())
        self.assertEqual(json.loads((self.job_dir / "status.json").read_text())["completed_run_ids"], [])

    def test_change_during_preflight_stops_before_runner(self):
        self.export(self.plan())
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[0] == "scontrol":
                self.canonical.write_text("changed during preflight")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run), \
             self.assertRaisesRegex(ValueError, "changed before execution"):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        self.assertFalse(any(command[0] == "/usr/bin/time" for command in commands))

    def test_cli_refuses_plan_export_inside_run_collection(self):
        with patch.object(launch, "make_plan", return_value=self.plan()), \
             patch.object(sys, "argv", [
                 "launch", "plan", "--manifest-dir", str(self.manifest_dir),
                 "--freeze", str(self.freeze_path), "--run-root", str(self.run_root),
                 "--export", str(self.run_root / "plan.json"),
             ]), self.assertRaises(SystemExit) as error:
            launch.main()
        self.assertEqual(error.exception.code, 1)
        self.assertFalse(self.run_root.exists())

    def test_run_root_symlink_change_after_preflight_is_rejected(self):
        self.export(self.plan())
        other = self.run_root.parent / "other"
        other.mkdir(parents=True)

        def fake_run(command, **kwargs):
            if command[0] == "scontrol":
                self.run_root.symlink_to(other, target_is_directory=True)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run) as run, \
             self.assertRaisesRegex(ValueError, "layout changed"):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        self.assertFalse(any(call.args[0][0] == "/usr/bin/time" for call in run.call_args_list))


    def test_expected_provenance_matches_reporting_contract(self):
        from src.reporting import adaptation as reporting
        for model, condition, source, budget in (
            ("catboost", "real_only", "none", 250), ("catboost", "mixed", "kaggle", 1000),
            ("ft_transformer", "synthetic_only", "source", 0), ("tabpfn", "synthetic_only", "kaggle", 0),
        ):
            spec = next(r for r in self.runs if r["root_seed"] == 42 and r["model"] == model
                        and r["condition"] == condition and r["source"] == source and r["budget"] == budget)
            with self.subTest(run_id=spec["run_id"]):
                mine = launch._expected_provenance(self.document, spec)
                theirs = reporting._expected_provenance(self.document, spec)
                for a, b in zip(mine, theirs):
                    self.assertEqual(a.to_dict("records"), b.to_dict("records"))

    def test_resume_skips_verified_complete_and_executes_missing(self):
        first = self.catboost_specs()[0]
        self.write_completed_run(first)
        plan = self.plan(resume=True)
        self.assertEqual(plan, self.plan(resume=True))
        by_id = {run["run_id"]: run for run in plan["runs"]}
        self.assertEqual(by_id[first["run_id"]]["output_status"], "verified_complete")
        self.assertIn("completed_metadata_sha256", by_id[first["run_id"]])
        self.assertEqual(sum(run["output_status"] == "pending" for run in plan["runs"]), 54)
        self.export(plan)
        executed = []

        def fake_run(command, **kwargs):
            if command[0] == "/usr/bin/time":
                run_id = command[command.index("--run-id") + 1]
                executed.append(run_id)
                self.write_completed_run(self.by_id[run_id])
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        self.assertEqual(len(executed), 54)
        self.assertNotIn(first["run_id"], executed)
        self.assertFalse((self.job_dir / f"{first['run_id']}.command.json").exists())
        status = json.loads((self.job_dir / "status.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertFalse(status["no_op"])
        self.assertEqual(status["skipped_verified_run_ids"], [first["run_id"]])
        self.assertEqual(status["completed_run_ids"], executed)
        self.assertEqual(status["remaining_run_ids"], [])
        self.assertEqual(status["invalid_run_ids"], [])

    def test_resume_with_all_runs_verified_complete_is_explicit_no_op(self):
        for spec in self.catboost_specs():
            self.write_completed_run(spec)
        plan = self.plan(resume=True)
        self.assertTrue(all(run["output_status"] == "verified_complete" for run in plan["runs"]))
        self.export(plan)
        with patch.object(subprocess, "run") as run:
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        run.assert_not_called()
        status = json.loads((self.job_dir / "status.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["no_op"])
        self.assertEqual(len(status["skipped_verified_run_ids"]), 55)
        self.assertEqual(status["completed_run_ids"], [])
        self.assertIn("No-op", (self.job_dir / "job.log").read_text())

    def test_blocked_output_classifications_and_reasons(self):
        specs = self.catboost_specs()
        self.write_completed_run(specs[0], mutate=lambda o, m: m.update(
            status="failed", error={"type": "RuntimeError", "message": "boom"}))
        self.write_completed_run(specs[1], mutate=lambda o, m: m.update(status="running"))
        missing_artifact = self.write_completed_run(specs[2])
        (missing_artifact / "predictions.csv.gz").unlink()
        altered = self.write_completed_run(specs[3])
        (altered / "fake.model").write_text("altered after completion")
        malformed = self.write_completed_run(specs[4])
        (malformed / "metadata.json").write_text("{not json")
        expected = {
            specs[0]["run_id"]: ("failed", "Recorded failure: boom"),
            specs[1]["run_id"]: ("incomplete", "running"),
            specs[2]["run_id"]: ("incomplete", "required run artifact is missing"),
            specs[3]["run_id"]: ("incompatible", "fingerprint differs"),
            specs[4]["run_id"]: ("incomplete", "JSONDecodeError"),
        }
        plan = self.plan(resume=True)
        by_id = {run["run_id"]: run for run in plan["runs"]}
        for run_id, (state, fragment) in expected.items():
            with self.subTest(run_id=run_id):
                self.assertEqual(by_id[run_id]["output_status"], state)
                self.assertNotEqual(by_id[run_id]["output_status"], "pending")
                self.assertIn(fragment, by_id[run_id]["blocking_reason"])
        with self.assertRaisesRegex(ValueError, "Blocked outputs"):
            launch.validate_plan(copy.deepcopy(plan), "catboost")

    def test_deep_verification_blocks_wrong_freeze_mode_identity_and_provenance(self):
        specs = self.catboost_specs()
        self.write_completed_run(specs[0], mutate=lambda o, m: m.update(
            freeze={**m["freeze"], "status": "draft"}))
        self.assertEqual(self.classify(specs[0]),
                         ("incompatible", "ValueError: recorded freeze differs from the approved freeze record"))
        self.write_completed_run(specs[1], mutate=lambda o, m: m.update(mode="development"))
        state, reason = self.classify(specs[1])
        self.assertEqual(state, "incompatible")
        self.assertIn("does not match", reason)
        valid = self.write_completed_run(specs[2])
        shutil.copytree(valid, self.run_root / specs[3]["run_id"])
        state, reason = self.classify(specs[3])
        self.assertEqual(state, "incompatible")
        self.assertIn("does not match", reason)
        tampered = self.write_completed_run(specs[4])
        frame = pd.read_csv(tampered / "fit_identities.csv")
        frame.loc[0, "identity"] = 999999
        frame.to_csv(tampered / "fit_identities.csv", index=False)
        self.rehash(tampered)
        state, reason = self.classify(specs[4])
        self.assertEqual(state, "incompatible")
        self.assertIn("fit identities differ", reason)
        self.assertEqual(self.classify(specs[2]), ("verified_complete", None))

    def test_completed_artifact_change_after_planning_invalidates_execution(self):
        output = self.write_completed_run(self.catboost_specs()[0])
        self.export(self.plan(resume=True))
        (output / "fake.model").write_text("tampered after planning")
        with patch.object(subprocess, "run") as run, \
             self.assertRaisesRegex(ValueError, "changed since planning"):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        run.assert_not_called()
        self.assertFalse(self.job_dir.exists())

    def test_pending_output_created_after_resume_planning_is_not_overwritten(self):
        plan = self.plan(resume=True)
        self.export(plan)
        target = self.run_root / plan["runs"][0]["run_id"]
        target.mkdir(parents=True)
        marker = target / "keep.txt"
        marker.write_text("evidence from another job")
        with patch.object(subprocess, "run") as run, \
             self.assertRaisesRegex(ValueError, "changed since planning"):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), "evidence from another job")

    def test_newly_executed_run_with_invalid_artifacts_fails_job(self):
        plan = self.plan(resume=True)
        self.export(plan)
        calls = []

        def fake_run(command, **kwargs):
            if command[0] == "/usr/bin/time":
                calls.append(command)
                run_id = command[command.index("--run-id") + 1]
                (self.run_root / run_id).mkdir(parents=True)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run), \
             self.assertRaisesRegex(ValueError, "failed artifact verification"):
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        self.assertEqual(len(calls), 1)
        status = json.loads((self.job_dir / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_run_ids"], [])
        self.assertEqual(status["invalid_run_ids"],
                         [{"run_id": plan["runs"][0]["run_id"], "state": "incomplete",
                           "reason": "metadata.json is missing."}])

    def test_archive_moves_failed_run_and_enables_fresh_retry(self):
        spec = self.catboost_specs()[3]
        self.write_completed_run(spec, mutate=lambda o, m: m.update(
            status="failed", error={"type": "RuntimeError", "message": "boom"}))
        stale = self.plan(resume=True)
        self.export(stale)
        with self.assertRaisesRegex(ValueError, "Blocked outputs"):
            launch.validate_plan(copy.deepcopy(stale), "catboost")
        archive_dir = self.project / "results/adaptation/x01_v1/execution/archive/20261122"
        records = launch.archive_runs(
            manifest_dir=self.manifest_dir, freeze_path=self.freeze_path, run_root=self.run_root,
            archive_dir=archive_dir, run_ids=[spec["run_id"]], reason="Job 100 timed out mid-run.")
        destination = archive_dir / spec["run_id"]
        self.assertTrue((destination / "metadata.json").is_file())
        self.assertIn(spec["run_id"], (destination / "fake.model").read_text())
        self.assertFalse((self.run_root / spec["run_id"]).exists())
        record = json.loads((archive_dir / f"{spec['run_id']}.archive.json").read_text())
        self.assertEqual(records[0], record)
        self.assertEqual(record["run_id"], spec["run_id"])
        self.assertEqual(record["state"], "failed")
        self.assertEqual(record["reason"], "Job 100 timed out mid-run.")
        self.assertEqual(record["original_path"], str(self.run_root / spec["run_id"]))
        self.assertEqual(record["archive_path"], str(destination))
        self.assertIn("archived_at_utc", record)
        self.assertTrue(record["provenance"]["available"])
        self.assertEqual(record["provenance"]["error"]["message"], "boom")
        self.assertEqual(record["provenance"]["code_revision"]["commit"], "fixture-revision")
        with self.assertRaisesRegex(ValueError, "changed since planning"):
            launch.validate_plan(json.loads(self.plan_path.read_text()), "catboost")
        fresh = self.plan(resume=True)
        entry = next(run for run in fresh["runs"] if run["run_id"] == spec["run_id"])
        self.assertEqual(entry["output_status"], "pending")
        launch.validate_plan(copy.deepcopy(fresh), "catboost")

    def test_archive_refusals_preserve_evidence(self):
        specs = self.catboost_specs()
        verified = self.write_completed_run(specs[0])
        failed = self.write_completed_run(specs[1], mutate=lambda o, m: m.update(
            status="failed", error={"type": "RuntimeError", "message": "boom"}))
        corrupted = self.write_completed_run(specs[4])
        (corrupted / "fake.model").write_text("corrupted artifact")
        os.symlink(failed, self.run_root / specs[5]["run_id"])
        args = dict(manifest_dir=self.manifest_dir, freeze_path=self.freeze_path,
                    run_root=self.run_root, reason="documented reason")
        archive_dir = self.project / "execution/archive/a"
        for run_ids, allow, message in (
            ([specs[0]["run_id"]], False, "verified complete"),
            (["not_a_run"], False, "outside the approved matrix"),
            ([specs[2]["run_id"]], False, "no output to archive"),
            ([specs[1]["run_id"], specs[1]["run_id"]], False, "unique run IDs"),
            ([specs[5]["run_id"]], False, "symlink"),
            ([specs[4]["run_id"]], False, "incompatible/corrupted"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                launch.archive_runs(**args, archive_dir=archive_dir, run_ids=run_ids,
                                    allow_incompatible=allow)
        with self.assertRaisesRegex(ValueError, "documented reason"):
            launch.archive_runs(**{**args, "reason": "  "}, archive_dir=archive_dir,
                                run_ids=[specs[1]["run_id"]])
        with self.assertRaisesRegex(ValueError, "separate"):
            launch.archive_runs(**args, archive_dir=self.run_root / "archive",
                                run_ids=[specs[1]["run_id"]])
        collision = self.project / "execution/archive/b"
        (collision / specs[1]["run_id"]).mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            launch.archive_runs(**args, archive_dir=collision, run_ids=[specs[1]["run_id"]])
        self.assertTrue((verified / "metadata.json").is_file())
        self.assertTrue((failed / "metadata.json").is_file())
        self.assertTrue((corrupted / "metadata.json").is_file())
        records = launch.archive_runs(**args, archive_dir=archive_dir,
                                      run_ids=[specs[4]["run_id"]], allow_incompatible=True)
        self.assertEqual(records[0]["state"], "incompatible")
        self.assertTrue(records[0]["allow_incompatible"])
        self.assertTrue((archive_dir / specs[4]["run_id"] / "fake.model").is_file())

    def test_inventory_cli_reports_statuses(self):
        import io
        from contextlib import redirect_stdout
        specs = self.catboost_specs()
        self.write_completed_run(specs[0])
        self.write_completed_run(specs[1], mutate=lambda o, m: m.update(
            status="failed", error={"type": "RuntimeError", "message": "boom"}))
        argv = ["launch", "inventory", "--model", "catboost",
                "--project", str(self.project), "--python", sys.executable,
                "--manifest-dir", str(self.manifest_dir), "--freeze", str(self.freeze_path),
                "--run-root", str(self.run_root)]
        buffer = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(buffer):
            launch.main()
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["counts"],
                         {"verified_complete": 1, "failed": 1, "pending": 53})
        statuses = {run["run_id"]: run for run in report["runs"]}
        self.assertIn("boom", statuses[specs[1]["run_id"]]["reason"])
        self.assertIsNone(statuses[specs[0]["run_id"]]["reason"])


class BatchTests(unittest.TestCase):
    def test_cpu_gpu_allocations_and_wrapper_failure_exit(self):
        cpu = (ROOT / "scripts/x03_newton_cpu.sbatch").read_text()
        gpu = (ROOT / "scripts/x03_newton_gpu.sbatch").read_text()
        self.assertNotIn("#SBATCH --gres", cpu)
        self.assertIn("#SBATCH --gres=gpu:nvidia_h100_pcie:1", gpu)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/x03_newton_common.sh", project / "scripts")
            fake_python = project / "python"
            fake_python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CALLS\"\nexit 7\n")
            fake_python.chmod(0o755)
            calls = project / "calls"
            env = {**os.environ, "PROJECT_DIR": str(project), "PYTHON": str(fake_python),
                   "SLURM_JOB_ID": "99", "SLURM_CPUS_PER_TASK": "4", "CALLS": str(calls)}
            result = subprocess.run(["bash", str(ROOT / "scripts/x03_newton_cpu.sbatch"), "/plan.json"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertIn("catboost", calls.read_text().splitlines())
            invalid = subprocess.run(["bash", str(ROOT / "scripts/x03_newton_gpu.sbatch"),
                                      "catboost", "/plan.json"], env=env, capture_output=True)
            self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()
