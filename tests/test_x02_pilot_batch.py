import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.experiments.adaptation_matrix import CHECKPOINTS

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts/x02_newton_pilot.sbatch"

FAKE_PYTHON = r"""#!{python}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args and args[0].endswith("x02_pilot_summary.py"):
    os.execv({python_repr}, [{python_repr}, *args])
if args == ["-"]:
    sys.stdin.read()
    sys.exit(0)
if args[:3] == ["-m", "src.experiments.adaptation_manifest", "validate"]:
    sys.exit(0)
if args[:4] != ["-u", "-m", "src.experiments.adaptation_runner", "execute"]:
    sys.exit(91)
if any(os.environ.get(k) != v for k, v in {{
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}}.items()):
    sys.exit(92)
with open(os.environ["CALLS"], "a") as out:
    out.write(json.dumps(args) + "\n")
def value(name):
    return args[args.index(name) + 1]
run = value("--run-id")
directory = Path(value("--output-root")) / run
directory.mkdir()
model = next(m for m in ("catboost", "ft_transformer", "tabpfn", "tabicl") if run.startswith(m + "_"))
failed = os.environ.get("FAIL_FIRST") == "1"
metadata = {{
    "run_id": run, "mode": value("--mode"), "status": "failed" if failed else "completed",
    "specification": {{"model": model, "condition": "fixture", "source": "none", "budget": 250, "root_seed": 42}},
    "error": {{"type": "Fixture", "message": "intentional"}} if failed else None,
}}
(directory / "metadata.json").write_text(json.dumps(metadata))
sys.exit(7 if failed else 0)
"""


class PilotChecksumTests(unittest.TestCase):
    def test_batch_checks_the_recorded_sha256_pins(self):
        script = BATCH.read_text()
        block = re.search(
            r"(?ms)^printf '%s  %s\\n' \\\n(.*?)\|\s*sha256sum --check\s*$",
            script,
        )
        self.assertIsNotNone(block, "Missing checksum preflight command")
        tokens = shlex.split(block.group(1).replace("\\\n", " "))
        self.assertEqual(len(tokens), 6, "Expected exactly three checksum/path pairs")
        pairs = list(zip(tokens[::2], tokens[1::2]))
        self.assertEqual(
            [path for _, path in pairs],
            ["$MANIFEST_DIR/assignments.json", "$TABPFN_CHECKPOINT", "$TABICL_CHECKPOINT"],
        )
        for digest, path in pairs:
            with self.subTest(path=path):
                self.assertRegex(digest, r"^[0-9a-fA-F]{64}$")
        stage2c = (ROOT / "docs/results/x02_stage2c_validation.md").read_text()
        recorded = re.search(r"manifest assignment fingerprint was\s*`([0-9a-f]{64})`", stage2c)
        self.assertIsNotNone(recorded, "Stage 2C manifest fingerprint not found")
        self.assertEqual(pairs[0][0], recorded.group(1))
        self.assertEqual(pairs[1][0], CHECKPOINTS["tabpfn"])
        self.assertEqual(pairs[2][0], CHECKPOINTS["tabicl"])


class PilotBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("git", "scontrol", "nvidia-smi", "sha256sum"):
            path = self.bin / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        self.python = self.bin / "python-fixture"
        self.python.write_text(FAKE_PYTHON.format(python=sys.executable, python_repr=repr(sys.executable)))
        self.python.chmod(0o755)
        paths = [
            "results/adaptation/x01_v1/manifest/assignments.json",
            "results/adaptation/x01_v1/manifest/metadata.json",
            "data/processed/metadata.json", "results/nhanes_preparation/metadata.json",
            "data/processed/nhanes_common_16.csv", "data/processed/source_common.csv",
            "data/processed/kaggle_common.csv", "data/splits/nhanes_split_assignments.csv",
            "data/splits/source_split_assignments.csv", "data/splits/kaggle_split_assignments.csv",
            "configs/catboost_source_n2000.json", "configs/ft_transformer_source_n2000.json",
            "configs/tabpfn_source_n2000_float32_batch1024.json",
            "configs/tabicl_v2_source_n2000_none_fp32.json",
            "src/experiments/adaptation_manifest.py", "src/experiments/adaptation_matrix.py",
            "src/experiments/adaptation_runner.py",
            "tabpfn.checkpoint", "tabicl.checkpoint",
        ]
        for path in paths:
            file = self.project / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture")
        (self.project / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/x02_pilot_summary.py", self.project / "scripts")
        self.calls = self.root / "calls.jsonl"
        self.env = {
            **os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
            "PROJECT_DIR": str(self.project), "PYTHON": str(self.python),
            "TABPFN_CHECKPOINT": str(self.project / "tabpfn.checkpoint"),
            "TABICL_CHECKPOINT": str(self.project / "tabicl.checkpoint"),
            "SLURM_JOB_ID": "99", "SLURM_CPUS_PER_TASK": "4",
            "CALLS": str(self.calls),
        }

    def run_wave(self, wave, **env):
        return subprocess.run(
            ["bash", str(BATCH), wave], env={**self.env, **env},
            text=True, capture_output=True, timeout=30,
        )

    def test_first_wave_failure_stops_and_keeps_evidence(self):
        result = self.run_wave("first", FAIL_FIRST="1")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 1)
        root = self.project / "results/adaptation/x01_v1/development/stage2e_first_99"
        summary = json.loads((root / "summary.json").read_text())
        self.assertEqual([x["status"] for x in summary["runs"]],
                         ["failed", "not_started", "not_started", "not_started"])
        self.assertTrue((root / "logs/catboost_mixed_source_B250_S1800_seed42.time").exists())
        self.assertNotEqual(self.run_wave("first").returncode, 0)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)

    def test_timing_wave_uses_capped_development_runs_in_sequence(self):
        result = self.run_wave("timing-ft_transformer")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 6)
        for args in calls:
            self.assertEqual(args[args.index("--mode") + 1], "development")
            self.assertEqual(args[args.index("--training-cap") + 1], "5")
            self.assertEqual(args[args.index("--query-size") + 1], "446")
        self.assertIn("synthetic_only_kaggle", calls[-1][calls[-1].index("--run-id") + 1])

    def test_full_training_wave_uses_four_uncapped_runs_and_separate_output(self):
        earlier = self.project / "results/adaptation/x01_v1/development/stage2e_timing-catboost_99"
        earlier.mkdir(parents=True)
        marker = earlier / "keep.txt"
        marker.write_text("earlier evidence")

        result = self.run_wave("full-training")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        expected = [
            "catboost_mixed_kaggle_B1000_S1200_seed42",
            "catboost_synthetic_only_kaggle_B0_S2000_seed42",
            "ft_transformer_mixed_kaggle_B1000_S1200_seed42",
            "ft_transformer_synthetic_only_kaggle_B0_S2000_seed42",
        ]
        self.assertEqual(
            [args[args.index("--run-id") + 1] for args in calls],
            expected,
        )
        for args in calls:
            self.assertEqual(args[args.index("--mode") + 1], "development")
            self.assertEqual(args[args.index("--query-size") + 1], "446")
            self.assertNotIn("--training-cap", args)

        root = self.project / "results/adaptation/x01_v1/development/stage2e_full-training_99"
        self.assertEqual((root / "plan.txt").read_text().splitlines(), expected)
        self.assertTrue(marker.exists())
        self.assertNotEqual(root, earlier)
        self.assertTrue(all((root / "runs" / run_id).is_dir() for run_id in expected))

    def test_full_training_wave_failure_stops_and_keeps_nonzero_status(self):
        result = self.run_wave("full-training", FAIL_FIRST="1")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--training-cap", calls[0])
        root = self.project / "results/adaptation/x01_v1/development/stage2e_full-training_99"
        summary = json.loads((root / "summary.json").read_text())
        self.assertEqual(
            [item["status"] for item in summary["runs"]],
            ["failed", "not_started", "not_started", "not_started"],
        )

    def test_first_wave_uses_all_models_and_pinned_path_overrides(self):
        result = self.run_wave("first")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        ids = [args[args.index("--run-id") + 1] for args in calls]
        self.assertEqual(ids, [
            "catboost_mixed_source_B250_S1800_seed42",
            "ft_transformer_mixed_kaggle_B250_S1800_seed42",
            "tabpfn_real_only_none_B250_S0_seed42",
            "tabicl_mixed_source_B250_S1800_seed42",
        ])
        for args in calls:
            self.assertEqual(args[args.index("--mode") + 1], "development")
            self.assertEqual(args[args.index("--query-size") + 1], "64")
        for args in calls[:2]:
            self.assertEqual(args[args.index("--training-cap") + 1], "5")
        self.assertEqual(calls[2][calls[2].index("--checkpoint") + 1], self.env["TABPFN_CHECKPOINT"])
        self.assertEqual(calls[3][calls[3].index("--checkpoint") + 1], self.env["TABICL_CHECKPOINT"])

    def test_missing_input_fails_before_runner(self):
        (self.project / "tabpfn.checkpoint").unlink()
        result = self.run_wave("first")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing/unreadable file", result.stdout + result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
