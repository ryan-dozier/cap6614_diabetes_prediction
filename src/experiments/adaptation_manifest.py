"""Candidate A assignment generation and verification; no model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import data
from src.experiments.identity import identity_fingerprints
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = "x02_manifest_v1"
ALGORITHM = "x01_nested_v1"
SEEDS = (42, 43, 44, 45, 46)
STREAMS = {"nhanes/train": 0, "nhanes/val": 1, "source/train": 2, "kaggle/train": 3}
BUDGETS = {250: (200, 50, 1800), 500: (400, 100, 1600), 1000: (800, 200, 1200)}
MODELS = ["catboost", "ft_transformer", "tabpfn", "tabicl"]
EXPECTED_NHANES = {
    250: ({"0": 169, "1": 31}, {"0": 42, "1": 8}),
    500: ({"0": 337, "1": 63}, {"0": 84, "1": 16}),
    1000: ({"0": 674, "1": 126}, {"0": 168, "1": 32}),
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identities(values: pd.Series, name: str) -> np.ndarray:
    if values.isna().any() or values.dtype.kind not in "iu" or values.duplicated().any():
        raise ValueError(f"{name} identities must be nonmissing, unique integers.")
    return values.to_numpy(dtype=np.int64)


def nested_sample(frame: pd.DataFrame, sizes: tuple[int, ...], seed: int, stream: int) -> dict[int, pd.DataFrame]:
    """Draw nested, identity-sorted subsets from identity/label columns."""
    if type(seed) is not int or seed < 0 or type(stream) is not int or stream < 0:
        raise ValueError("Seed and stream must be nonnegative integers.")
    ids = _identities(frame["identity"], "Sampling")
    labels = frame["label"].to_numpy()
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Sampling labels must be nonmissing binary 0/1 values.")
    order = np.argsort(ids, kind="stable")
    ids, labels = ids[order], labels[order]
    classes = [ids[labels == label] for label in (0, 1)]
    if any(len(rows) == 0 for rows in classes):
        raise ValueError("Sampling requires both binary classes.")
    if not sizes or any(type(n) is not int or n < 2 or n > len(ids) for n in sizes):
        raise ValueError(f"Subset sizes must be integers between 2 and available rows ({len(ids)}).")
    permutations = [
        np.random.default_rng(np.random.SeedSequence([seed, stream, label])).permutation(rows)
        for label, rows in enumerate(classes)
    ]
    output: dict[int, pd.DataFrame] = {}
    previous = (0, 0)
    for n in sorted(set(sizes)):
        positives = (2 * n * len(classes[1]) + len(ids)) // (2 * len(ids))
        positives = min(max(positives, 1, n - len(classes[0])), n - 1, len(classes[1]))
        quotas = (n - positives, positives)
        if any(a < b for a, b in zip(quotas, previous)):
            raise ValueError("Class quotas are not nested across requested sizes.")
        previous = quotas
        selected = pd.DataFrame({
            "identity": np.concatenate([permutations[0][:quotas[0]], permutations[1][:quotas[1]]]),
            "label": np.repeat([0, 1], quotas),
        }).sort_values("identity", kind="stable").reset_index(drop=True)
        output[n] = selected
    return output


def mixed_identities(synthetic_domain: str, synthetic_ids: list[int], nhanes_ids: list[int]) -> list[tuple[str, int]]:
    """Return provenance order for mixed inputs without treating identities as features."""
    if synthetic_domain not in ("source", "kaggle"):
        raise ValueError("Mixed inputs require source or kaggle as the synthetic domain.")
    synthetic = _identities(pd.Series(synthetic_ids), synthetic_domain)
    real = _identities(pd.Series(nhanes_ids), "nhanes")
    return [(synthetic_domain, int(value)) for value in sorted(synthetic)] + [
        ("nhanes", int(value)) for value in sorted(real)
    ]


def _input_paths() -> list[Path]:
    return [
        data.METADATA_PATH, data.NHANES_METADATA_PATH,
        data.NHANES_COMMON_PATHS["strict_common_16"],
        data.COMMON_PATHS["source"], data.COMMON_PATHS["kaggle"],
        *data.SPLIT_ASSIGNMENT_PATHS.values(),
    ]


def _file_fingerprints() -> dict[str, str]:
    return {str(path.relative_to(PROJECT_ROOT)): _sha256_file(path) for path in _input_paths()}


def load_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read only selected train/val labels; test rows are skipped by the CSV parser."""
    schema = data.load_feature_schema("strict_common_16")
    inputs: dict[str, pd.DataFrame] = {}
    split_info: dict[str, Any] = {}
    for domain in ("nhanes", "source", "kaggle"):
        identity_column = "seqn" if domain == "nhanes" else "row_index"
        assignments = pd.read_csv(data.SPLIT_ASSIGNMENT_PATHS[domain], usecols=["row_index", identity_column, "split"])
        indices = _identities(assignments["row_index"], f"{domain} row_index")
        ids = _identities(assignments[identity_column], domain)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError(f"{domain} split row indices must match canonical row order.")
        if set(assignments["split"]) != {"train", "val", "test"}:
            raise ValueError(f"{domain} split assignments must contain train, val and test only.")
        for split in ("train", "val", "test"):
            split_ids = ids[assignments["split"].eq(split).to_numpy()]
            split_info[f"{domain}/{split}"] = {
                "rows": len(split_ids),
                **identity_fingerprints(domain, identity_column, sorted(split_ids)),
            }
        permitted = assignments["split"].isin(["train", "val"] if domain == "nhanes" else ["train"]).to_numpy()
        path = data.NHANES_COMMON_PATHS["strict_common_16"] if domain == "nhanes" else data.COMMON_PATHS[domain]

        def skip_row(line: int) -> bool:
            if line == 0:
                return False
            if line > len(permitted):
                raise ValueError(f"{domain} canonical data exceed split assignments.")
            return not permitted[line - 1]

        columns = [schema.target, "seqn"] if domain == "nhanes" else [schema.target]
        selected = pd.read_csv(path, usecols=columns, skiprows=skip_row)
        selected_assignments = assignments.loc[permitted].reset_index(drop=True)
        if len(selected) != len(selected_assignments):
            raise ValueError(f"{domain} canonical data and split assignments differ in size.")
        if domain == "nhanes" and not np.array_equal(selected["seqn"], selected_assignments["seqn"]):
            raise ValueError("NHANES canonical identities do not match split assignments.")
        for split in (("train", "val") if domain == "nhanes" else ("train",)):
            mask = selected_assignments["split"].eq(split)
            inputs[f"{domain}/{split}"] = pd.DataFrame({
                "identity": selected_assignments.loc[mask, identity_column].to_numpy(),
                "label": selected.loc[mask, schema.target].to_numpy(),
            })
    return inputs, split_info


def _subset(seed: int, budget: int, role: str, domain: str, split: str, rows: pd.DataFrame) -> dict[str, Any]:
    identity_column = "seqn" if domain == "nhanes" else "row_index"
    ids = [int(value) for value in rows["identity"]]
    return {
        "root_seed": seed, "budget": budget, "role": role, "domain": domain,
        "canonical_partition": split, "identity_column": identity_column,
        "subset_size": len(ids), "participant_ids": ids,
        "class_counts": {str(label): int((rows["label"] == label).sum()) for label in (0, 1)},
        **identity_fingerprints(domain, identity_column, ids),
    }


def build_assignments(inputs: dict[str, pd.DataFrame], split_info: dict[str, Any]) -> dict[str, Any]:
    subsets = []
    for seed in SEEDS:
        sampled = {
            key: nested_sample(inputs[key], sizes, seed, STREAMS[key])
            for key, sizes in (
                ("nhanes/train", (200, 400, 800)), ("nhanes/val", (50, 100, 200)),
                ("source/train", (1200, 1600, 1800, 2000)),
                ("kaggle/train", (1200, 1600, 1800, 2000)),
            )
        }
        for budget, (fit, val, synthetic) in BUDGETS.items():
            real = _subset(seed, budget, "fit", "nhanes", "train", sampled["nhanes/train"][fit])
            validation = _subset(seed, budget, "validation", "nhanes", "val", sampled["nhanes/val"][val])
            if (real["class_counts"], validation["class_counts"]) != EXPECTED_NHANES[budget]:
                raise ValueError(f"NHANES quotas do not match the approved budget {budget}.")
            subsets.extend((real, validation))
            for domain in ("source", "kaggle"):
                subsets.append(_subset(seed, budget, "synthetic_mixed", domain, "train", sampled[f"{domain}/train"][synthetic]))
        for domain in ("source", "kaggle"):
            subsets.append(_subset(seed, 0, "synthetic_anchor", domain, "train", sampled[f"{domain}/train"][2000]))
    return {
        "schema_version": SCHEMA_VERSION, "sampling_algorithm": ALGORITHM,
        "feature_set": "strict_common_16", "root_seeds": list(SEEDS), "streams": STREAMS,
        "models_share_assignments": MODELS,
        "reuse_policy": "One real fit and validation subset per seed and budget is shared by all models and real-only/mixed conditions; each synthetic subset is shared by all models; anchors are reused across budgets.",
        "mixed_order": "Synthetic rows first, NHANES rows second; ascending identity within each domain.",
        "identity_policy": "Identity is (domain, participant ID); provenance is never a predictor.",
        "quota_formula": "Clamp floor(n * positive_population / population_size + 0.5) to [max(1, n - negative_population), min(n - 1, positive_population)].",
        "nesting_policy": "One permutation per class, stream and root seed; class-wise prefixes for all sizes. Repetition overlap is permitted.",
        "split_identities": split_info, "subsets": subsets,
    }


def _revision() -> dict[str, Any]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout)
    code_paths = (Path(__file__), PROJECT_ROOT / "src/experiments/identity.py", Path(data.__file__))
    return {
        "commit": revision, "worktree_dirty": dirty,
        "code_file_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256_file(path) for path in code_paths},
    }


def _reconstruct() -> tuple[dict[str, Any], dict[str, str]]:
    before = _file_fingerprints()
    inputs, split_info = load_inputs()
    assignments = build_assignments(inputs, split_info)
    if before != _file_fingerprints():
        raise ValueError("Canonical input files changed during manifest generation or validation.")
    return assignments, before


def generate(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    assignments, fingerprints = _reconstruct()
    payload = _json_bytes(assignments)
    checked, checked_fingerprints = _reconstruct()
    if payload != _json_bytes(checked) or fingerprints != checked_fingerprints:
        raise ValueError("Manifest reconstruction failed before publication.")
    metadata = {
        "status": "complete", "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_sha256": hashlib.sha256(payload).hexdigest(),
        "source_file_sha256": fingerprints, "code_revision": _revision(),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    (output / "assignments.json").write_bytes(payload)
    (output / "metadata.json").write_bytes(_json_bytes(metadata))


def validate(output: Path) -> dict[str, Any]:
    metadata = json.loads((output / "metadata.json").read_text())
    payload = (output / "assignments.json").read_bytes()
    if metadata.get("status") != "complete" or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest is incomplete or has an unsupported schema version.")
    if hashlib.sha256(payload).hexdigest() != metadata.get("assignment_sha256"):
        raise ValueError("Assignment file fingerprint mismatch.")
    expected, fingerprints = _reconstruct()
    if fingerprints != metadata.get("source_file_sha256"):
        raise ValueError("Canonical source-file fingerprint mismatch.")
    if payload != _json_bytes(expected):
        raise ValueError("Assignments or subset fingerprints do not match the canonical reconstruction.")
    return {"subsets": len(expected["subsets"]), "repetitions": len(SEEDS), "assignment_sha256": metadata["assignment_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "validate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.output_dir)
    print(json.dumps(validate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
