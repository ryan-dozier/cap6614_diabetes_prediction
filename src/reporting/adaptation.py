"""Validate and report approved Candidate A research artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import data
from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from src.experiments.metrics import binary_labels, evaluate_binary, probabilities
from src.experiments.results import write_json

METRICS = ("roc_auc", "average_precision", "brier_score", "log_loss")
SEEDS = manifest.SEEDS
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_REPLICATES = 2000
COHORT_COUNTS = (545, 101)


def _read_provenance(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["domain", "partition", "identity"]:
        raise ValueError(f"{path.name}: invalid provenance columns")
    if len(frame):
        if frame["identity"].isna().any() or frame["identity"].dtype.kind not in "iu":
            raise ValueError(f"{path.name}: identities must be nonmissing integers")
        if frame.duplicated(["domain", "identity"]).any():
            raise ValueError(f"{path.name}: duplicate domain-qualified identity")
    return frame


def _expected_provenance(doc: dict[str, Any], spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["domain", "partition", "identity"]
    fit: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for ref in spec["subsets"]:
        subset = runner._subset(doc, ref)
        rows = [{"domain": ref["domain"], "partition": ref["canonical_partition"], "identity": i}
                for i in subset["participant_ids"]]
        if ref["role"] == "validation":
            if spec["model"] in runner.TASK_TRAINED:
                selection.extend(rows)
        else:
            fit.extend(rows)
    if spec["condition"] == "synthetic_only" and spec["model"] in runner.TASK_TRAINED:
        assignments = runner._assignments(spec["source"])
        ids = sorted(int(x) for x in assignments.loc[assignments["split"].eq("val"), "row_index"])
        selection = [{"domain": spec["source"], "partition": "val", "identity": i} for i in ids]
    return pd.DataFrame(fit, columns=columns), pd.DataFrame(selection, columns=columns)


def _aligned_predictions(path: Path, expected_ids: list[int]) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    required = ["domain", "partition", "identity", "y_true", "y_probability", "y_pred"]
    if list(frame.columns) != required:
        raise ValueError(f"{path.name}: invalid prediction columns")
    manifest._identities(frame["identity"], path.name)
    if not frame["domain"].eq("nhanes").all() or not frame["partition"].eq("test").all():
        raise ValueError(f"{path.name}: predictions must be NHANES test participants")
    if set(frame["identity"]) != set(expected_ids) or len(frame) != len(expected_ids):
        raise ValueError(f"{path.name}: missing or extra test participant identities")
    labels = binary_labels(frame["y_true"])
    p = probabilities(frame["y_probability"])
    decisions = binary_labels(frame["y_pred"])
    if not np.array_equal(decisions, (p >= 0.5).astype(int)):
        raise ValueError(f"{path.name}: saved threshold decisions differ from probabilities")
    frame["y_true"] = labels
    return frame.set_index("identity").loc[expected_ids].reset_index()


def _validate_completed(path: Path, doc: dict[str, Any], spec: dict[str, Any],
                        runs: list[dict[str, Any]], freeze: dict[str, Any],
                        source_hashes: dict[str, str], test_ids: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("matrix_sha256") != matrix.matrix_sha256(runs) or meta.get("freeze") != freeze:
        raise ValueError("matrix fingerprint or approved freeze differs")
    if meta.get("prepared_file_sha256") != source_hashes:
        raise ValueError("canonical source/split fingerprints differ")
    if json.loads((path / "run_specification.json").read_text()) != spec:
        raise ValueError("run specification artifact differs from approved matrix")
    config = meta["effective_configuration"]
    if not isinstance(config, dict) or not isinstance(config.get("model_params"), dict):
        raise ValueError("effective configuration and model parameters must be objects")
    params = config["model_params"]
    device = params.get("device") if spec["model"] != "catboost" else None
    checkpoint = params.get("model_path") if spec["model"] in matrix.CHECKPOINTS else None
    if spec["model"] != "catboost" and (not isinstance(device, str) or not device.strip()):
        raise ValueError("recorded device must be a nonempty string")
    if spec["model"] in matrix.CHECKPOINTS and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("recorded checkpoint path must be a nonempty string")
    if config != runner._configuration(spec, device, checkpoint, None):
        raise ValueError("effective scientific configuration differs from approved profile")
    if spec["model"] in matrix.CHECKPOINTS:
        if meta["model"]["checkpoint"]["sha256"] != matrix.CHECKPOINTS[spec["model"]]:
            raise ValueError("recorded checkpoint fingerprint differs from approved checkpoint")
    schema = data.load_feature_schema("strict_common_16")
    if {"domain", "row_index", "seqn", "id", schema.target}.intersection(schema.features):
        raise ValueError("identity or target columns cannot be model features")
    if meta.get("feature_order") != list(schema.features):
        raise ValueError("feature order differs from approved schema")
    fit, selection = _expected_provenance(doc, spec)
    for role, expected in (("fit", fit), ("selection", selection)):
        actual = _read_provenance(path / f"{role}_identities.csv")
        if actual.to_dict("records") != expected.to_dict("records"):
            raise ValueError(f"{role} identities differ from manifest assignments")
    query = _read_provenance(path / "query_identities.csv")
    expected_query = pd.DataFrame({"domain": "nhanes", "partition": "test", "identity": test_ids})
    if query.to_dict("records") != expected_query.to_dict("records"):
        raise ValueError("query provenance differs from ordered canonical test identities")
    runner._disjoint(fit, selection, query)
    real_fit = int((fit["domain"] == "nhanes").sum())
    real_selection = int((selection["domain"] == "nhanes").sum())
    accounting = {
        "nhanes_available": spec["budget"], "nhanes_fit_or_context_used": real_fit,
        "nhanes_selection_used": real_selection,
        "nhanes_reserved_unused": spec["budget"] - real_fit - real_selection,
        "synthetic_fit_used": len(fit) - real_fit,
        "synthetic_selection_used": len(selection) - real_selection,
        "query_labels_for_metrics": len(test_ids), "query_label_category": "research_evaluation",
    }
    if meta.get("label_accounting") != accounting or accounting["nhanes_reserved_unused"] < 0:
        raise ValueError("label accounting differs from assigned identities")
    return meta, _aligned_predictions(path / "predictions.csv.gz", test_ids)


def stratified_indices(labels: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    labels = binary_labels(labels)
    classes = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in classes) or type(replicates) is not int or replicates < 1:
        raise ValueError("Bootstrap requires both classes and a positive replicate count")
    rng = np.random.default_rng(seed)
    return np.stack([np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in classes])
                     for _ in range(replicates)]).astype(np.int64)


def _four_metrics(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    values = np.asarray([roc_auc_score(y, p), average_precision_score(y, p),
                         brier_score_loss(y, p), log_loss(y, p, labels=[0, 1])], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap metric")
    return values


def bootstrap_effects(labels: np.ndarray, mixed: np.ndarray, comparator: np.ndarray,
                      indices: np.ndarray) -> np.ndarray:
    """Replicate mean within-seed improvements, using common participant draws."""
    labels = binary_labels(labels)
    if mixed.shape != comparator.shape or mixed.ndim != 2 or mixed.shape[1] != len(labels):
        raise ValueError("Paired probability arrays must have matching seed/participant dimensions")
    if indices.ndim != 2 or indices.shape[1] != len(labels):
        raise ValueError("Bootstrap indices have invalid dimensions")
    effects = np.empty((len(indices), len(METRICS)))
    direction = np.array([1, 1, -1, -1])
    for i, draw in enumerate(indices):
        effects[i] = np.mean([(_four_metrics(labels[draw], a[draw]) -
                               _four_metrics(labels[draw], b[draw])) * direction
                              for a, b in zip(mixed, comparator)], axis=0)
    return effects


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate(manifest_dir: Path, freeze_path: Path, run_root: Path, output: Path) -> Path:
    """Generate an immutable research report from saved predictions only."""
    started = time.perf_counter()
    doc, runs = matrix.load(manifest_dir)
    anchor = next(s for s in runs if s["budget"] == 0)
    freeze = runner.check_freeze(freeze_path, runs, anchor["run_id"])
    approved = [s for s in runs if s["run_id"] in freeze["approved_run_ids"]]
    runner._output_path(run_root, "research", approved[0]["run_id"])
    if not run_root.is_dir():
        raise ValueError(f"Research run root does not exist: {run_root}")
    output = output.resolve()
    if output == run_root.resolve() or output.is_relative_to(run_root.resolve()):
        raise ValueError("Report output must be separate from research run artifacts")
    if output.exists():
        raise FileExistsError(f"Report output already exists: {output}")
    unexpected = [p.name for p in run_root.iterdir() if p.is_dir() and p.name not in freeze["approved_run_ids"]]
    if unexpected:
        raise ValueError(f"Unapproved run directories: {sorted(unexpected)}")
    assignments = runner._assignments("nhanes")
    test_ids = sorted(int(x) for x in assignments.loc[assignments["split"].eq("test"), "seqn"])
    if len(test_ids) != sum(COHORT_COUNTS):
        raise ValueError("Canonical research cohort must contain 646 participants")
    source_hashes = json.loads((manifest_dir / "metadata.json").read_text())["source_file_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    inventory: list[dict[str, Any]] = []
    valid: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    labels: np.ndarray | None = None
    try:
        for spec in approved:
            path = run_root / spec["run_id"]
            status, reason = runner._inventory_status(path, spec, "research")
            if status == "completed":
                try:
                    meta, pred = _validate_completed(path, doc, spec, runs, freeze, source_hashes, test_ids)
                    y = pred["y_true"].to_numpy()
                    if tuple(int((y == c).sum()) for c in (0, 1)) != COHORT_COUNTS:
                        raise ValueError("research cohort class counts must be 545 negative and 101 positive")
                    if labels is not None and not np.array_equal(labels, y):
                        raise ValueError("participant labels conflict across runs")
                    labels = y
                    valid[spec["run_id"]] = (meta, pred)
                except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                    status, reason = "incompatible", f"{type(exc).__name__}: {exc}"
            inventory.append({"run_id": spec["run_id"], "model": spec["model"], "condition": spec["condition"],
                              "source": spec["source"], "budget": spec["budget"], "seed": spec["root_seed"],
                              "status": status, "reason": reason or ""})
        _write_csv(output / "inventory.csv", inventory, list(inventory[0]))
        invalid = [item for item in inventory if item["status"] == "incompatible"]
        if invalid:
            raise ValueError(f"Incompatible research artifacts: {len(invalid)}; see inventory.csv for run-specific reasons")
        _produce(output, approved, valid, labels, freeze, manifest_dir, freeze_path, run_root, started)
    except Exception as exc:
        write_json(output / "report_status.json", {"status": "failed", "error": str(exc)})
        raise
    return output


def _produce(output: Path, approved: list[dict[str, Any]],
             valid: dict[str, tuple[dict[str, Any], pd.DataFrame]], labels: np.ndarray | None,
             freeze: dict[str, Any], manifest_dir: Path, freeze_path: Path, run_root: Path,
             started: float) -> None:
    specs = {(s["model"], s["condition"], s["source"], s["budget"], s["root_seed"]): s for s in approved}
    run_metrics: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for spec in approved:
        run_id = spec["run_id"]
        if run_id not in valid:
            continue
        pred = valid[run_id][1]
        result = evaluate_binary(pred["y_true"], pred["y_probability"], 0.5, 10)
        run_metrics[run_id] = result
        metric_rows.append({"run_id": run_id, "model": spec["model"], "condition": spec["condition"],
                            "source": spec["source"], "budget": spec["budget"], "seed": spec["root_seed"],
                            **{m: result[m] for m in METRICS},
                            "f1": result["f1"], "precision": result["precision"], "recall": result["recall"],
                            "ece": result["ece"], "tn": result["confusion_matrix"]["tn"],
                            "fp": result["confusion_matrix"]["fp"], "fn": result["confusion_matrix"]["fn"],
                            "tp": result["confusion_matrix"]["tp"],
                            "calibration_bins_json": json.dumps(result["calibration_bins"], separators=(",", ":"))})
    _write_csv(output / "run_metrics.csv", metric_rows,
               ["run_id", "model", "condition", "source", "budget", "seed", *METRICS,
                "f1", "precision", "recall", "ece", "tn", "fp", "fn", "tp", "calibration_bins_json"])
    comparisons: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    needed: set[str] = set()
    for model in manifest.MODELS:
        for budget in sorted(freeze["approved_budgets"]):
            for source in ("source", "kaggle"):
                for contrast in ("real_only", "synthetic_only"):
                    mixed_ids = [specs[(model, "mixed", source, budget, seed)]["run_id"] for seed in SEEDS]
                    comparator_ids = [specs[(model, contrast, "none" if contrast == "real_only" else source,
                                             budget if contrast == "real_only" else 0, seed)]["run_id"]
                                      for seed in SEEDS]
                    available = [seed for seed, a, b in zip(SEEDS, mixed_ids, comparator_ids)
                                 if a in valid and b in valid]
                    missing = sorted({run for run in mixed_ids + comparator_ids if run not in valid})
                    cell = {"model": model, "budget": budget, "source": source, "contrast": contrast,
                            "status": "complete" if not missing else "incomplete",
                            "expected_seeds": ",".join(map(str, SEEDS)),
                            "available_seeds": ",".join(map(str, available)),
                            "missing_run_ids": ";".join(missing)}
                    comparisons.append(cell)
                    for seed, a, b in zip(SEEDS, mixed_ids, comparator_ids):
                        if a not in valid or b not in valid:
                            continue
                        for metric in METRICS:
                            x, y = run_metrics[a][metric], run_metrics[b][metric]
                            delta = (x - y) * (1 if metric in METRICS[:2] else -1)
                            paired.append({**{k: cell[k] for k in ("model", "budget", "source", "contrast", "status")},
                                           "seed": seed, "metric": metric, "mixed_run_id": a,
                                           "comparator_run_id": b, "mixed_metric": x, "comparator_metric": y,
                                           "improvement": delta})
                    if not missing:
                        needed.update(mixed_ids + comparator_ids)
    _write_csv(output / "comparisons.csv", comparisons,
               ["model", "budget", "source", "contrast", "status", "expected_seeds", "available_seeds", "missing_run_ids"])
    _write_csv(output / "paired_differences.csv", paired,
               ["model", "budget", "source", "contrast", "status", "seed", "metric",
                "mixed_run_id", "comparator_run_id", "mixed_metric", "comparator_metric", "improvement"])
    draws = None
    boot: dict[str, np.ndarray] = {}
    if needed:
        if labels is None:
            raise ValueError("Complete comparisons require validated labels")
        draws = stratified_indices(labels, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        np.save(output / "bootstrap_indices.npy", draws, allow_pickle=False)
        cached: dict[bytes, np.ndarray] = {}
        for run_id in sorted(needed):
            p = valid[run_id][1]["y_probability"].to_numpy()
            key = p.tobytes()
            if key not in cached:
                cached[key] = np.stack([_four_metrics(labels[draw], p[draw]) for draw in draws])
            boot[run_id] = cached[key]
    summary: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    for cell in comparisons:
        if cell["status"] != "complete":
            continue
        model, budget, source, contrast = (cell[k] for k in ("model", "budget", "source", "contrast"))
        a = [specs[(model, "mixed", source, budget, seed)]["run_id"] for seed in SEEDS]
        b = [specs[(model, contrast, "none" if contrast == "real_only" else source,
                    budget if contrast == "real_only" else 0, seed)]["run_id"] for seed in SEEDS]
        for index, metric in enumerate(METRICS):
            x = np.array([run_metrics[r][metric] for r in a])
            y = np.array([run_metrics[r][metric] for r in b])
            sign = 1 if metric in METRICS[:2] else -1
            differences = sign * (x - y)
            samples = sign * np.mean(np.stack([boot[u][:, index] - boot[v][:, index]
                                                for u, v in zip(a, b)]), axis=0)
            low, high = np.quantile(samples, [0.025, 0.975], method="linear")
            summary.append({**{k: cell[k] for k in ("model", "budget", "source", "contrast")},
                            "metric": metric, "n_seeds": len(SEEDS), "mixed_mean": x.mean(),
                            "mixed_sd": x.std(ddof=1), "comparator_mean": y.mean(),
                            "comparator_sd": y.std(ddof=1), "mean_improvement": differences.mean(),
                            "improvement_sd": differences.std(ddof=1), "ci_lower": low, "ci_upper": high})
            distribution.extend({**{k: cell[k] for k in ("model", "budget", "source", "contrast")},
                                 "metric": metric, "replicate": n, "mean_improvement": value}
                                for n, value in enumerate(samples))
    _write_csv(output / "summary.csv", summary,
               ["model", "budget", "source", "contrast", "metric", "n_seeds", "mixed_mean", "mixed_sd",
                "comparator_mean", "comparator_sd", "mean_improvement", "improvement_sd", "ci_lower", "ci_upper"])
    _write_csv(output / "bootstrap_effects.csv", distribution,
               ["model", "budget", "source", "contrast", "metric", "replicate", "mean_improvement"])
    incomplete = [c for c in comparisons if c["status"] == "incomplete"]
    lines = ["# Candidate A research adaptation report", "",
             f"Completed runs: {len(valid)} / {len(approved)}. Complete comparisons: {len(comparisons) - len(incomplete)} / {len(comparisons)}.",
             "", "## Complete primary comparisons (ROC-AUC)", "",
             "| Model | Budget | Source | Mean improvement | 95% interval |",
             "|---|---:|---|---:|---:|"]
    primary = [row for row in summary if row["contrast"] == "real_only" and row["metric"] == "roc_auc"]
    for row in primary:
        lines.append(f"| {row['model']} | {row['budget']} | {row['source']} | "
                     f"{row['mean_improvement']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
    if not primary:
        lines.append("| No complete comparisons | | | | |")
    lines.extend(["", "## Incomplete comparisons", ""])
    if not incomplete:
        lines.append("None.")
    for cell in incomplete:
        lines.append(f"- {cell['model']} B{cell['budget']} {cell['source']} vs {cell['contrast']}: "
                     f"available seeds [{cell['available_seeds']}], expected [{cell['expected_seeds']}]; "
                     f"missing/failed run IDs: {cell['missing_run_ids']}")
    lines.extend(["", "## Interpretation", "",
                  "Positive effects indicate improvement. Synthetic-only comparisons are pipeline contrasts, "
                  "not isolated effects of synthetic fitting rows. See CSV tables for all cells and seed values.",
                  "Intervals are pointwise participant-bootstrap intervals conditional on the five fitted models "
                  "and observed cohort/class composition, not full training-sample uncertainty or simultaneous intervals.",
                  "This unweighted participant bootstrap ignores NHANES survey weights, strata and clusters; "
                  "it does not support national-population inference. Crossing zero does not establish equivalence.",
                  "Run standard deviations combine sampling and model variability and are not confidence intervals."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "x02_adaptation_report_v1", "status": "completed",
        "manifest_sha256": approved[0]["manifest_sha256"], "matrix_sha256": freeze["matrix_sha256"],
        "manifest_metadata_sha256": manifest._sha256_file(manifest_dir / "metadata.json"),
        "reporting_code_sha256": manifest._sha256_file(Path(__file__)),
        "freeze_sha256": manifest._sha256_file(freeze_path), "run_root": str(run_root.resolve()),
        "manifest_dir": str(manifest_dir.resolve()), "freeze_path": str(freeze_path.resolve()),
        "approved_run_count": len(approved), "completed_run_count": len(valid),
        "complete_comparisons": len(comparisons) - len(incomplete), "incomplete_comparisons": len(incomplete),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
                      "rng": "numpy.default_rng", "class_counts": {"0": COHORT_COUNTS[0], "1": COHORT_COUNTS[1]},
                      "interval": "pointwise 95% percentile, linear interpolation",
                      "indices_sha256": manifest._sha256_file(output / "bootstrap_indices.npy") if draws is not None else None},
        "metrics": {"primary": "roc_auc", "secondary": list(METRICS[1:]),
                    "log_loss": "sklearn log_loss float64 clipping", "threshold": 0.5, "calibration_bins": 10},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": version("scikit-learn")},
        "code_revision": manifest._revision(), "reporting_seconds": time.perf_counter() - started,
        "run_provenance": {r: {"metadata_sha256": manifest._sha256_file(run_root / r / "metadata.json"),
                               "code_revision": meta.get("code_revision"), "host": meta.get("host"),
                               "effective_configuration": meta.get("effective_configuration")}
                           for r, (meta, _) in valid.items()},
    }
    write_json(output / "report_metadata.json", metadata)
    write_json(output / "report_status.json", {"status": "completed"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.manifest_dir, args.freeze, args.run_root, args.output_dir))


if __name__ == "__main__":
    main()
