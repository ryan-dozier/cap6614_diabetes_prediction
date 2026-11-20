from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import adaptation_manifest as manifest
from src.experiments.identity import identity_fingerprints


def population(size: int, positives: int, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "identity": np.arange(start, start + size, dtype=np.int64),
        "label": np.r_[np.ones(positives, dtype=int), np.zeros(size - positives, dtype=int)],
    })


def inputs() -> dict[str, pd.DataFrame]:
    return {
        "nhanes/train": population(3014, 473),
        "nhanes/val": population(646, 102, 4000),
        "source/train": population(2600, 400),
        "kaggle/train": population(3000, 1200),
    }


class NestedSamplingTests(unittest.TestCase):
    def test_exact_quotas_nesting_streams_and_order(self):
        frame = inputs()["nhanes/train"]
        subsets = manifest.nested_sample(frame, (200, 400, 800), 42, 0)
        for size, positives in ((200, 31), (400, 63), (800, 126)):
            ids = subsets[size]["identity"]
            self.assertEqual(len(ids), size)
            self.assertEqual(len(set(ids)), size)
            self.assertEqual(ids.tolist(), sorted(ids))
            self.assertEqual(int(subsets[size]["label"].sum()), positives)
        self.assertLessEqual(set(subsets[200]["identity"]), set(subsets[400]["identity"]))
        self.assertLessEqual(set(subsets[400]["identity"]), set(subsets[800]["identity"]))
        reordered = manifest.nested_sample(frame.sample(frac=1, random_state=8), (200, 400, 800), 42, 0)
        for size in subsets:
            pd.testing.assert_frame_equal(subsets[size], reordered[size])
        other = manifest.nested_sample(frame, (200,), 42, 1)
        self.assertNotEqual(subsets[200]["identity"].tolist(), other[200]["identity"].tolist())
        self.assertNotEqual(subsets[200]["identity"].tolist(), manifest.nested_sample(frame, (200,), 43, 0)[200]["identity"].tolist())

    def test_invalid_inputs(self):
        frame = population(10, 3)
        invalid = [
            frame.assign(identity=[0] * 10),
            frame.assign(identity=[None] + list(range(1, 10))),
            frame.assign(label=[2] + [0] * 9),
            frame.assign(label=[float("nan")] + [0] * 9),
            frame.assign(label=[0] * 10),
        ]
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "identities|labels|classes"):
                manifest.nested_sample(bad, (2,), 42, 0)
        for size in (1, 11, True, 2.5):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "sizes"):
                manifest.nested_sample(frame, (size,), 42, 0)

    def test_half_up_and_feasibility(self):
        frame = population(10, 1)
        self.assertEqual(manifest.nested_sample(frame, (5, 10), 42, 0)[5]["label"].sum(), 1)
        self.assertEqual(manifest.nested_sample(population(4, 1), (2,), 42, 0)[2]["label"].sum(), 1)

    def test_mixed_order_keeps_equal_ids_in_separate_domains(self):
        self.assertEqual(
            manifest.mixed_identities("source", [3, 1], [3, 2]),
            [("source", 1), ("source", 3), ("nhanes", 2), ("nhanes", 3)],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            manifest.mixed_identities("source", [1, 1], [2])


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = inputs()
        cls.document = manifest.build_assignments(cls.data, {})

    def test_all_subsets_and_pairing(self):
        doc = self.document
        self.assertEqual(len(doc["subsets"]), 70)
        self.assertEqual(doc["models_share_assignments"], manifest.MODELS)
        self.assertIn("Synthetic rows first", doc["mixed_order"])
        for seed in manifest.SEEDS:
            entries = [item for item in doc["subsets"] if item["root_seed"] == seed]
            for budget, (fit, val, synthetic) in manifest.BUDGETS.items():
                real = next(s for s in entries if s["budget"] == budget and s["role"] == "fit")
                validation = next(s for s in entries if s["budget"] == budget and s["role"] == "validation")
                self.assertEqual((real["subset_size"], validation["subset_size"]), (fit, val))
                self.assertEqual((real["class_counts"], validation["class_counts"]), manifest.EXPECTED_NHANES[budget])
                self.assertLessEqual(set(real["participant_ids"]), set(self.data["nhanes/train"]["identity"]))
                self.assertLessEqual(set(validation["participant_ids"]), set(self.data["nhanes/val"]["identity"]))
                self.assertFalse(set(real["participant_ids"]) & set(validation["participant_ids"]))
                self.assertFalse((set(real["participant_ids"]) | set(validation["participant_ids"])) & set(range(5000, 5646)))
                for domain in ("source", "kaggle"):
                    mixed = next(s for s in entries if s["budget"] == budget and s["domain"] == domain)
                    anchor = next(s for s in entries if s["role"] == "synthetic_anchor" and s["domain"] == domain)
                    self.assertEqual(mixed["subset_size"], synthetic)
                    positive_count = (2 * synthetic * int(self.data[f"{domain}/train"]["label"].sum()) + len(self.data[f"{domain}/train"])) // (2 * len(self.data[f"{domain}/train"]))
                    self.assertEqual(mixed["class_counts"]["1"], positive_count)
                    self.assertLessEqual(set(mixed["participant_ids"]), set(anchor["participant_ids"]))
                    self.assertLessEqual(set(anchor["participant_ids"]), set(self.data[f"{domain}/train"]["identity"]))
            for domain, role in (("nhanes", "fit"), ("nhanes", "validation"), ("source", "synthetic_mixed"), ("kaggle", "synthetic_mixed")):
                seq = sorted((s for s in entries if s["domain"] == domain and s["role"] == role), key=lambda s: s["subset_size"])
                for first, second in zip(seq, seq[1:]):
                    self.assertLessEqual(set(first["participant_ids"]), set(second["participant_ids"]))
                for subset in seq:
                    ids = subset["participant_ids"]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(ids, sorted(ids))
                    self.assertEqual(identity_fingerprints(domain, subset["identity_column"], ids)["ordered_identity_sha256"], subset["ordered_identity_sha256"])

    def test_reproducibility_and_row_order(self):
        shuffled = {key: value.sample(frac=1, random_state=1) for key, value in self.data.items()}
        self.assertEqual(self.document, manifest.build_assignments(shuffled, {}))

    def test_tampering_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(manifest, "_reconstruct", return_value=(self.document, {"fixture": "hash"})):
            output = Path(tmp) / "manifest"
            manifest.generate(output)
            self.assertEqual(manifest.validate(output)["subsets"], 70)
            with self.assertRaises(FileExistsError):
                manifest.generate(output)
            path = output / "assignments.json"
            assignment = json.loads(path.read_text())
            assignment["subsets"][0]["participant_ids"][0] += 1
            path.write_bytes(manifest._json_bytes(assignment))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                manifest.validate(output)
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["assignment_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata_path.write_bytes(manifest._json_bytes(metadata))
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                manifest.validate(output)
            metadata["source_file_sha256"]["fixture"] = "tampered"
            metadata_path.write_bytes(manifest._json_bytes(metadata))
            with self.assertRaisesRegex(ValueError, "source-file"):
                manifest.validate(output)

    def test_tampered_subset_fingerprint_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(manifest, "_reconstruct", return_value=(self.document, {"fixture": "hash"})):
            output = Path(tmp) / "manifest"
            manifest.generate(output)
            path = output / "assignments.json"
            assignment = json.loads(path.read_text())
            assignment["subsets"][0]["identity_sha256"] = "0" * 64
            path.write_bytes(manifest._json_bytes(assignment))
            meta_path = output / "metadata.json"
            metadata = json.loads(meta_path.read_text())
            metadata["assignment_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            meta_path.write_bytes(manifest._json_bytes(metadata))
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                manifest.validate(output)

    def test_reader_skips_test_labels_and_rejects_split_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            common = {}
            for domain in ("nhanes", "source", "kaggle"):
                paths[domain] = root / f"{domain}_splits.csv"
                common[domain] = root / f"{domain}_common.csv"
                assignment = pd.DataFrame({"row_index": [0, 1, 2, 3], "split": ["train", "train", "val", "test"]})
                if domain == "nhanes":
                    assignment["seqn"] = [10, 11, 12, 13]
                assignment.to_csv(paths[domain], index=False)
                frame = pd.DataFrame({"diagnosed_diabetes": ["0", "1", "1", "test-label-must-not-be-read"]})
                if domain == "nhanes":
                    frame["seqn"] = [10, 11, 12, 13]
                frame.to_csv(common[domain], index=False)
            with patch.object(manifest.data, "SPLIT_ASSIGNMENT_PATHS", paths), \
                 patch.object(manifest.data, "COMMON_PATHS", common), \
                 patch.object(manifest.data, "NHANES_COMMON_PATHS", {"strict_common_16": common["nhanes"]}), \
                 patch.object(manifest.data, "load_feature_schema") as schema:
                schema.return_value.target = "diagnosed_diabetes"
                loaded, split_info = manifest.load_inputs()
                self.assertEqual(loaded["nhanes/val"]["identity"].tolist(), [12])
                self.assertEqual(loaded["source/train"]["label"].tolist(), [0, 1])
                self.assertEqual(split_info["nhanes/test"]["rows"], 1)
                broken = pd.read_csv(paths["nhanes"])
                broken.loc[3, "seqn"] = 10
                broken.to_csv(paths["nhanes"], index=False)
                with self.assertRaisesRegex(ValueError, "identities"):
                    manifest.load_inputs()


if __name__ == "__main__":
    unittest.main()
