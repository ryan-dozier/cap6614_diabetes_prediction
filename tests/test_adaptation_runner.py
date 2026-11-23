from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import DatasetSchema
from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from test_adaptation_manifest import inputs


SCHEMA = DatasetSchema("diagnosed_diabetes", ("value",), (), ("value",), ())


class FakeAdapter:
    artifact_name = "fake.model"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def reset_peak_memory(self): pass
    def synchronize(self): pass
    def resource_metadata(self): return {"fake": True}
    def metadata(self): return {"adapter": "fake", "persistence": "fixture"}

    def fit(self, X, y, X_val, y_val):
        self.calls.append((X.copy(), y.copy(), X_val.copy(), y_val.copy()))
        if self.fail:
            raise RuntimeError("fixture failure")

    def save(self, path): path.write_text("fake artifact")
    def predict_proba_batched(self, X, size): return np.full(len(X), 0.5)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.populations = inputs()
        cls.document = manifest.build_assignments(cls.populations, {})
        cls.runs = matrix.build(cls.document, "a" * 64)

    def spec(self, model, condition, source="none", budget=250):
        return next(r for r in self.runs if r["root_seed"] == 42 and r["model"] == model
                    and r["condition"] == condition and r["source"] == source and r["budget"] == budget)

    def assignment(self, domain):
        if domain == "nhanes":
            train = self.populations["nhanes/train"]["identity"]
            val = self.populations["nhanes/val"]["identity"]
            return pd.DataFrame({"seqn": pd.concat([train, val], ignore_index=True),
                                 "split": ["train"] * len(train) + ["val"] * len(val)})
        return pd.DataFrame()

    def read_rows(self, domain, split, ids, schema):
        if split == "test":
            raise AssertionError("test rows forbidden")
        pop = self.populations[f"{domain}/{split}"].set_index("identity")
        self.assertTrue(set(ids) <= set(pop.index))
        y = pd.Series(pop.loc[ids, "label"].to_numpy(), name=schema.target)
        X = pd.DataFrame({"value": np.asarray(ids, dtype=float)})
        p = pd.DataFrame({"domain": domain, "partition": split, "identity": ids})
        return X, y, p

    def source_validation(self, domain, split, schema):
        self.assertIn(domain, ("source", "kaggle"))
        self.assertEqual(split, "val")
        ids = list(range(5000, 5010))
        return (pd.DataFrame({"value": ids}), pd.Series([0, 1] * 5, name=schema.target),
                pd.DataFrame({"domain": domain, "partition": "val", "identity": ids}))

    def patches(self):
        return (
            patch.object(runner.data, "load_feature_schema", return_value=SCHEMA),
            patch.object(runner, "_assignments", side_effect=self.assignment),
            patch.object(runner, "_read_rows", side_effect=self.read_rows),
            patch.object(runner, "_whole_partition", side_effect=self.source_validation),
        )

    def test_matrix_counts_order_reuse_and_staleness(self):
        self.assertEqual(Counter(r["condition"] for r in self.runs), {"real_only": 60, "mixed": 120, "synthetic_only": 40})
        self.assertEqual(len({r["run_id"] for r in self.runs}), 220)
        self.assertEqual(self.runs, matrix.build(self.document, "a" * 64))
        real = self.spec("catboost", "real_only")
        mixed = self.spec("tabpfn", "mixed", "source")
        self.assertEqual(real["subsets"], mixed["subsets"][1:])
        self.assertEqual(len([r for r in self.runs if r["condition"] == "synthetic_only" and r["model"] == "catboost" and r["source"] == "source"]), 5)
        with patch.object(matrix.manifest, "_sha256_file", return_value="changed"), self.assertRaisesRegex(ValueError, "profile changed"):
            matrix.build(self.document, "a" * 64)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            matrix.select(self.runs, "not-a-run")

    def test_import_is_lazy_and_checkpoint_mismatch_fails(self):
        code = ("import sys; import src.experiments.adaptation_runner; "
                "assert not set(sys.modules).intersection({'torch','catboost','tabpfn','tabicl','sklearn'})")
        subprocess.run([sys.executable, "-c", code], check=True)
        with patch.object(manifest, "_sha256_file", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                runner._verify_checkpoint("tabpfn", {"model_path": "/unused/fixture"})
        with self.assertRaisesRegex(ValueError, "explicit local checkpoint"):
            runner._verify_checkpoint("tabpfn", {})

    def test_assembly_all_conditions_and_models(self):
        with self.patches()[0], self.patches()[1], self.patches()[2], self.patches()[3]:
            for model in manifest.MODELS:
                for condition, source, budget in (("real_only", "none", 250), ("mixed", "source", 250),
                                                   ("mixed", "kaggle", 250), ("synthetic_only", "source", 0)):
                    with self.subTest(model=model, condition=condition, source=source):
                        spec = self.spec(model, condition, source, budget)
                        out = runner.assemble(self.document, spec, "development", 20)
                        fit, selection, query = out["fit"], out["selection"], out["query"]
                        self.assertEqual(list(fit[0].columns), ["value"])
                        self.assertEqual(len(fit[0]), 200 if condition == "real_only" else 2000)
                        self.assertEqual(len(query[0]), 20)
                        expected_selection = (50 if budget else 10) if model in runner.TASK_TRAINED else 0
                        self.assertEqual(len(selection[0]), expected_selection)
                        self.assertEqual(out["label_accounting"]["nhanes_available"], budget)
                        self.assertEqual(out["label_accounting"]["nhanes_fit_or_context_used"], 200 if budget else 0)
                        self.assertEqual(out["label_accounting"]["nhanes_selection_used"], 50 if budget and expected_selection else 0)
                        self.assertEqual(out["label_accounting"]["nhanes_reserved_unused"], 50 if budget and not expected_selection else 0)
                        self.assertEqual(out["label_accounting"]["synthetic_selection_used"], 10 if not budget and expected_selection else 0)
                        domains = fit[2]["domain"].tolist()
                        if condition == "mixed":
                            self.assertEqual(domains, [source] * 1800 + ["nhanes"] * 200)
                        for domain in set(domains):
                            ids = fit[2].loc[fit[2]["domain"] == domain, "identity"].tolist()
                            self.assertEqual(ids, sorted(ids))
                        runner._disjoint(fit[2], selection[2], query[2])

    def test_query_overlap_and_reference_tampering(self):
        spec = self.spec("catboost", "mixed", "source")
        bad = dict(spec["subsets"][0], identity_sha256="bad")
        with self.assertRaisesRegex(ValueError, "reference"):
            runner._subset(self.document, bad)
        p = pd.DataFrame({"domain": ["nhanes"], "identity": [10]})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            runner._disjoint(p, p)
        with self.patches()[0], self.patches()[1], self.patches()[2], self.patches()[3]:
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                runner.assemble(self.document, spec, "development", 9999)

    def test_fake_fit_receives_only_permitted_validation_for_all_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_dir = root / "manifest"
            manifest_dir.mkdir()
            (manifest_dir / "metadata.json").write_text(json.dumps({"source_file_sha256": {"fixture": "hash"}}))
            output_root = root / "adaptation/x01_v1/development"
            with patch.object(runner, "RESULTS_DIR", root), \
                 patch.object(matrix, "load", return_value=(self.document, self.runs)), \
                 patch.object(runner, "_verify_checkpoint"), \
                 patch.object(manifest, "_file_fingerprints", return_value={"fixture": "hash"}), \
                 self.patches()[0], self.patches()[1], self.patches()[2], self.patches()[3]:
                for model in manifest.MODELS:
                    for condition, source, budget in (("real_only", "none", 250), ("mixed", "source", 250),
                                                       ("mixed", "kaggle", 250), ("synthetic_only", "source", 0)):
                        spec = self.spec(model, condition, source, budget)
                        adapter = FakeAdapter()
                        with self.subTest(model=model, condition=condition, source=source), \
                             patch.object(runner, "_adapter", return_value=adapter):
                            out = runner.execute(manifest_dir, spec["run_id"], "development", output_root, query_size=20)
                            fit_X, _, val_X, _ = adapter.calls[0]
                            expected = (50 if budget else 10) if model in runner.TASK_TRAINED else 0
                            self.assertEqual(len(val_X), expected)
                            self.assertEqual(len(fit_X), 200 if condition == "real_only" else 2000)
                            self.assertEqual(json.loads((out / "metadata.json").read_text())["status"], "completed")

    def test_research_gate(self):
        spec = self.spec("catboost", "real_only")
        with self.assertRaisesRegex(ValueError, "freeze"):
            runner.check_freeze(None, self.runs, spec["run_id"])
        with patch.object(runner.data, "load_feature_schema", side_effect=AssertionError("data must not load")):
            with self.assertRaisesRegex(ValueError, "freeze"):
                runner.assemble(self.document, spec, "research", 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freeze.json"
            record = {"schema_version": "x02_freeze_v1", "status": "approved", "protocol_version": matrix.PROTOCOL,
                      "protocol_sha256": manifest._sha256_file(runner.PROJECT_ROOT / "docs/plans/x01_adaptation_protocol.md"),
                      "evaluation_protocol_sha256": manifest._sha256_file(runner.PROJECT_ROOT / "docs/plans/e05_evaluation_protocol.md"),
                      "matrix_sha256": matrix.matrix_sha256(self.runs), "approved_by": "fixture", "approved_at": "fixture",
                      "feasibility_decision": "fixture", "feasibility_evidence": "fixture", "approval_reference": "fixture",
                      "approved_budgets": [250], "approved_run_ids": [r["run_id"] for r in self.runs if r["budget"] in (0, 250)]}
            path.write_text(json.dumps(record))
            self.assertEqual(runner.check_freeze(path, self.runs, spec["run_id"]), record)
            record["matrix_sha256"] = "stale"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "stale"):
                runner.check_freeze(path, self.runs, spec["run_id"])

    def test_execute_fake_success_failure_and_inventory(self):
        spec = self.spec("tabpfn", "mixed", "source")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_dir = root / "manifest"
            manifest_dir.mkdir()
            (manifest_dir / "metadata.json").write_text(json.dumps({"source_file_sha256": {"fixture": "hash"}}))
            adapter = FakeAdapter()
            output_root = root / "adaptation/x01_v1/development"
            with patch.object(runner, "RESULTS_DIR", root), \
                 patch.object(matrix, "load", return_value=(self.document, self.runs)), \
                 patch.object(runner, "_adapter", return_value=adapter), \
                 patch.object(runner, "_verify_checkpoint"), \
                 patch.object(manifest, "_file_fingerprints", return_value={"fixture": "hash"}), \
                 self.patches()[0], self.patches()[1], self.patches()[2], self.patches()[3]:
                out = runner.execute(manifest_dir, spec["run_id"], "development", output_root, query_size=20)
                meta = json.loads((out / "metadata.json").read_text())
                self.assertEqual(meta["status"], "completed")
                self.assertFalse(meta["research_comparison_eligible"])
                self.assertEqual(len(adapter.calls[0][2]), 0)
                self.assertEqual(runner.inventory(self.runs, output_root, "development")[
                    self.runs.index(spec)]["status"], "completed")
                with self.assertRaises(FileExistsError):
                    runner.execute(manifest_dir, spec["run_id"], "development", output_root, query_size=20)
                (out / "metrics.json").write_text("{}")
                self.assertEqual(runner.inventory(self.runs, output_root, "development")[
                    self.runs.index(spec)]["status"], "incompatible")
                other = self.spec("catboost", "real_only")
                with patch.object(runner, "_adapter", return_value=FakeAdapter(fail=True)):
                    with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                        runner.execute(manifest_dir, other["run_id"], "development", output_root, query_size=20)
                failed = json.loads((output_root / other["run_id"] / "metadata.json").read_text())
                self.assertEqual(failed["status"], "failed")
                states = runner.inventory(self.runs, output_root, "development")
                self.assertEqual(states[self.runs.index(other)]["status"], "failed")
                self.assertIn("missing", {entry["status"] for entry in states})
                with self.assertRaisesRegex(ValueError, "freeze"):
                    runner.execute(manifest_dir, other["run_id"], "research", root / "adaptation/x01_v1/research")

    def test_inventory_malformed_metadata_and_valid_neighbor(self):
        cases = [
            ("truncated", '{"status":', "incomplete"),
            ("non-object", "[]", "incompatible"),
            ("invalid-status", {"status": []}, "incompatible"),
            ("artifact-null", {"model_artifact": None}, "incompatible"),
            ("artifact-list", {"model_artifact": ["model"]}, "incompatible"),
            ("artifact-parent", {"model_artifact": "../model"}, "incompatible"),
            ("artifact-absolute", {"model_artifact": "/tmp/model"}, "incompatible"),
            ("artifact-dot", {"model_artifact": "."}, "incompatible"),
            ("artifact-backslash", {"model_artifact": "other\\model"}, "incompatible"),
            ("hash-list", {"output_sha256": []}, "incompatible"),
            ("hash-path", {"output_sha256": {"../file": "digest"}}, "incompatible"),
            ("hash-value", {"output_sha256": {"file": []}}, "incompatible"),
            ("eligibility", {"research_comparison_eligible": "false"}, "incompatible"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "adaptation/x01_v1/development"
            specs = [{"run_id": name} for name, _, _ in cases] + [{"run_id": "valid"}]
            with patch.object(runner, "RESULTS_DIR", root):
                for spec, (_, value, _) in zip(specs, cases):
                    directory = output_root / spec["run_id"]
                    directory.mkdir(parents=True)
                    base = {"status": "completed", "run_id": spec["run_id"], "mode": "development",
                            "specification": spec, "model_artifact": "model.bin"}
                    content = value if isinstance(value, str) else json.dumps({**base, **value})
                    (directory / "metadata.json").write_text(content)
                good = specs[-1]
                directory = output_root / good["run_id"]
                directory.mkdir()
                names = ["run_specification.json", "fit_identities.csv", "selection_identities.csv",
                         "query_identities.csv", "predictions.csv.gz", "metrics.json", "model.bin"]
                hashes = {}
                for name in names:
                    payload = name.encode()
                    (directory / name).write_bytes(payload)
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                (directory / "metadata.json").write_text(json.dumps({
                    "status": "completed", "run_id": good["run_id"], "mode": "development",
                    "specification": good, "model_artifact": "model.bin",
                    "research_comparison_eligible": False, "output_sha256": hashes,
                }))
                entries = runner.inventory(specs, output_root, "development")
                for entry, (name, _, expected) in zip(entries, cases):
                    self.assertEqual((entry["run_id"], entry["status"]), (name, expected))
                    self.assertTrue(entry.get("reason"))
                self.assertEqual(entries[-1], {"run_id": "valid", "status": "completed"})
                (output_root / "truncated" / "metadata.json").write_bytes(b"\xff")
                unreadable = runner.inventory([specs[0], good], output_root, "development")
                self.assertEqual(unreadable[0]["status"], "incomplete")
                self.assertIn("UnicodeDecodeError", unreadable[0]["reason"])
                self.assertEqual(unreadable[1]["status"], "completed")

    def test_inventory_preserves_running_failed_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "adaptation/x01_v1/development"
            specs = [{"run_id": state} for state in ("running", "failed", "missing")]
            for spec in specs[:2]:
                directory = output_root / spec["run_id"]
                directory.mkdir(parents=True)
                (directory / "metadata.json").write_text(json.dumps({
                    "status": spec["run_id"], "run_id": spec["run_id"],
                    "mode": "development", "specification": spec,
                }))
            with patch.object(runner, "RESULTS_DIR", root):
                entries = runner.inventory(specs, output_root, "development")
            self.assertEqual([entry["status"] for entry in entries], ["running", "failed", "missing"])

    def test_selected_row_reader_skips_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "split.csv"
            common = root / "common.csv"
            pd.DataFrame({"row_index": [0, 1, 2, 3], "seqn": [5, 6, 7, 8],
                          "split": ["train", "train", "val", "test"]}).to_csv(split, index=False)
            common.write_text("seqn,value,diagnosed_diabetes\n5,1,0\n6,2,1\n7,3,1\n8,not-a-number,do-not-parse\n")
            with patch.dict(runner.data.SPLIT_ASSIGNMENT_PATHS, {"nhanes": split}), \
                 patch.dict(runner.data.NHANES_COMMON_PATHS, {"strict_common_16": common}):
                X, y, p = runner._read_rows("nhanes", "train", [5, 6], SCHEMA)
                self.assertEqual(X["value"].tolist(), [1, 2])
                self.assertEqual(y.tolist(), [0, 1])
                self.assertEqual(p["identity"].tolist(), [5, 6])
                with self.assertRaisesRegex(ValueError, "partition"):
                    runner._read_rows("nhanes", "train", [8], SCHEMA)


if __name__ == "__main__":
    unittest.main()
