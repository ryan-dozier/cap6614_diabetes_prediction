"""Build the Candidate A run matrix from a validated Stage 2A manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.experiments import adaptation_manifest as manifest
from src.paths import PROJECT_ROOT

VERSION = "x02_matrix_v1"
PROTOCOL = "x01_candidate_a_v1"
PROFILES = {
    "catboost": ("catboost_source_n2000.json", "c44fcec0e65a1bd5d08c535761dcf85efb4832716c210e26276801dbcad5259f"),
    "ft_transformer": ("ft_transformer_source_n2000.json", "f054c2dd5be7b61a50cb28bfa3dc8c2107e39d3264f7e1183e8f8d6d98ba6d37"),
    "tabpfn": ("tabpfn_source_n2000_float32_batch1024.json", "6230680ebb099c19a8b1fb2f747aabfd4145d32ff48d676d36ee1cde901f786e"),
    "tabicl": ("tabicl_v2_source_n2000_none_fp32.json", "dddecb41af2216474b92018f2ab9214147751a3d6b6fdadff9e3db39ccc070e2"),
}
CHECKPOINTS = {
    "tabpfn": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    "tabicl": "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
}


def profile(model: str) -> dict[str, Any]:
    name, expected = PROFILES[model]
    path = PROJECT_ROOT / "configs" / name
    if manifest._sha256_file(path) != expected:
        raise ValueError(f"Approved reference profile changed: {path}")
    config = json.loads(path.read_text())
    return {
        "reference_file": f"configs/{name}", "reference_sha256": expected,
        "model_params": config["model_params"], "prediction_batch_size": config["prediction_batch_size"],
        "early_stopping_rounds": config.get("early_stopping_rounds"),
        "checkpoint_sha256": CHECKPOINTS.get(model),
    }


def subset_key(subset: dict[str, Any]) -> dict[str, Any]:
    return {key: subset[key] for key in (
        "root_seed", "budget", "role", "domain", "canonical_partition",
        "subset_size", "identity_sha256", "ordered_identity_sha256",
    )}


def build(document: dict[str, Any], fingerprint: str) -> list[dict[str, Any]]:
    if document["schema_version"] != manifest.SCHEMA_VERSION or document["sampling_algorithm"] != manifest.ALGORITHM:
        raise ValueError("Unsupported manifest schema or algorithm.")
    indexed = {}
    for subset in document["subsets"]:
        key = (subset["root_seed"], subset["budget"], subset["role"], subset["domain"])
        if key in indexed:
            raise ValueError(f"Duplicate manifest subset: {key}")
        indexed[key] = subset_key(subset)
    runs = []
    for seed in manifest.SEEDS:
        for model in manifest.MODELS:
            model_profile = profile(model)
            for budget, (_, _, synthetic_size) in manifest.BUDGETS.items():
                for condition, source in (("real_only", "none"), ("mixed", "source"), ("mixed", "kaggle")):
                    references = [indexed[(seed, budget, "fit", "nhanes")], indexed[(seed, budget, "validation", "nhanes")]]
                    count = 0 if source == "none" else synthetic_size
                    if source != "none":
                        references.insert(0, indexed[(seed, budget, "synthetic_mixed", source)])
                    runs.append(_spec(seed, model, condition, source, budget, count, references, fingerprint, model_profile))
            for source in ("source", "kaggle"):
                refs = [indexed[(seed, 0, "synthetic_anchor", source)]]
                runs.append(_spec(seed, model, "synthetic_only", source, 0, 2000, refs, fingerprint, model_profile))
    ids = [run["run_id"] for run in runs]
    if len(runs) != 220 or len(set(ids)) != 220:
        raise ValueError("Candidate A matrix must contain exactly 220 unique runs.")
    return runs


def _spec(seed: int, model: str, condition: str, source: str, budget: int, count: int,
          refs: list[dict[str, Any]], fingerprint: str, model_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": VERSION, "protocol_version": PROTOCOL,
        "manifest_schema": manifest.SCHEMA_VERSION, "manifest_sha256": fingerprint,
        "run_id": f"{model}_{condition}_{source}_B{budget}_S{count}_seed{seed}",
        "model": model, "root_seed": seed, "condition": condition, "source": source,
        "budget": budget, "synthetic_size": count, "feature_set": "strict_common_16",
        "subsets": refs, "profile": model_profile,
    }


def load(manifest_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verification = manifest.validate(manifest_dir)
    payload = (manifest_dir / "assignments.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != verification["assignment_sha256"]:
        raise ValueError("Manifest changed during matrix construction.")
    doc = json.loads(payload)
    return doc, build(doc, verification["assignment_sha256"])


def matrix_sha256(runs: list[dict[str, Any]]) -> str:
    return hashlib.sha256(manifest._json_bytes(runs)).hexdigest()


def select(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    found = [run for run in runs if run["run_id"] == run_id]
    if len(found) != 1:
        raise ValueError(f"Unknown or duplicate run ID: {run_id}")
    return found[0]
