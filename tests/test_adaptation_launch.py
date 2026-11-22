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

from src.experiments import adaptation_launch as launch
from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from test_adaptation_manifest import inputs

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = manifest.build_assignments(inputs(), {})
        cls.runs = matrix.build(cls.document, "a" * 64)

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
        (self.manifest_dir / "metadata.json").write_text("{}")
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
            return subprocess.CompletedProcess(command, 0)

        with patch.object(subprocess, "run", side_effect=fake_run), \
             self.assertRaises(subprocess.CalledProcessError) as error:
            launch.execute_plan(self.plan_path, "catboost", self.job_dir)
        self.assertEqual(error.exception.returncode, 7)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--training-cap", calls[0])
        self.assertTrue((self.job_dir / "plan.json").is_file())
        self.assertFalse(self.run_root.exists())
        status = json.loads((self.job_dir / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_run_ids"], [plan["runs"][0]["run_id"]])

    def test_gpu_preflight_and_cpu_does_not_require_gpu(self):
        for model in ("catboost", "ft_transformer"):
            with self.subTest(model=model):
                self.export(self.plan(model))
                with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
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
