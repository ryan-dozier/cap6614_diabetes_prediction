"""Verify a trusted development run's provenance and same-batch save/reload."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments import adaptation_manifest as manifest
from src.experiments import adaptation_matrix as matrix
from src.experiments import adaptation_runner as runner
from src.paths import RESULTS_DIR

# Existing FT/TabPFN smoke criteria; TabICL uses 64 FP32 epsilons.
# CatBoost native CPU reload uses an absolute 1e-12 engineering criterion.
TOLERANCES = {
    "catboost": (0.0, 1e-12),
    "ft_transformer": (1e-5, 1e-6),
    "tabpfn": (1e-5, 1e-6),
    "tabicl": (0.0, 64 * float(np.finfo(np.float32).eps)),
}


def _reload(model: str, artifact: Path, schema: Any, metadata: dict[str, Any], output: Path) -> Any:
    if model == "catboost":
        from catboost import CatBoostClassifier

        native = CatBoostClassifier()
        native.load_model(str(artifact), format="cbm")

        def predict(X: pd.DataFrame, batch_size: int) -> np.ndarray:
            frame = X.copy()
            for name in schema.categorical_features:
                frame[name] = frame[name].astype(str)
            classes = np.asarray(native.classes_)
            if classes.shape != (2,) or set(classes) != {0, 1}:
                raise ValueError("Reloaded CatBoost classes are not binary 0/1.")
            index = int(np.flatnonzero(classes == 1)[0])
            return np.concatenate([
                np.asarray(native.predict_proba(frame.iloc[start:start + batch_size]))[:, index]
                for start in range(0, len(frame), batch_size)
            ])

        return predict
    device = metadata["effective_configuration"]["model_params"]["device"]
    if model == "ft_transformer":
        from src.models.ft_transformer import FTTransformerAdapter
        restored = FTTransformerAdapter.load(artifact, device=device)
    elif model == "tabpfn":
        from src.models.tabpfn import TabPFNAdapter
        restored = TabPFNAdapter.load(artifact, device=device)
    elif model == "tabicl":
        from src.models.tabicl import TabICLAdapter
        restored = TabICLAdapter.load(artifact, device=device, diagnostic_path=output / "native_reload.log")
    else:
        raise ValueError(f"Unsupported model: {model}")
    return restored.predict_proba_batched


def _provenance(path: Path, expected: pd.DataFrame) -> None:
    observed = pd.read_csv(path)
    if list(observed.columns) != ["domain", "partition", "identity"]:
        raise ValueError(f"{path.name}: invalid provenance columns.")
    if observed.to_dict("records") != expected.to_dict("records"):
        raise ValueError(f"{path.name}: ordered provenance differs from reconstructed assignments.")


def verify(manifest_dir: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    development = (RESULTS_DIR / "adaptation/x01_v1/development").resolve()
    run_dir, output = run_dir.resolve(), output.resolve()
    if not run_dir.is_relative_to(development) or run_dir == development:
        raise ValueError("Verification requires a development run directory.")
    if not output.is_relative_to(development) or output == development or output.is_relative_to(run_dir):
        raise ValueError("Verification output must be separate from the run under the development root.")
    if output.exists():
        raise FileExistsError(f"Verification output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    report: dict[str, Any] = {"status": "running", "run_directory": str(run_dir)}
    from src.experiments.results import write_json
    start = time.perf_counter()
    try:
        write_json(output / "verification.json", report)
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if not isinstance(metadata, dict) or metadata.get("mode") != "development" or metadata.get("research_comparison_eligible") is not False:
            raise ValueError("Only completed development artifacts are eligible for verification.")
        doc, runs = matrix.load(manifest_dir)
        spec = matrix.select(runs, run_dir.name)
        if runner._inventory_status(run_dir, spec, "development")[0] != "completed":
            raise ValueError("Run inventory or artifact fingerprints are not completed and valid.")
        if metadata["matrix_sha256"] != matrix.matrix_sha256(runs):
            raise ValueError("Matrix fingerprint differs from the current validated matrix.")
        before = manifest._file_fingerprints()
        if before != metadata["prepared_file_sha256"]:
            raise ValueError("Canonical inputs differ from run provenance.")
        config = metadata["effective_configuration"]
        params = config["model_params"]
        model = spec["model"]
        device = params.get("device") if model != "catboost" else None
        checkpoint = params.get("model_path") if model in matrix.CHECKPOINTS else None
        expected_config = runner._configuration(spec, device, checkpoint, config["development_training_cap"])
        if config != expected_config:
            raise ValueError("Effective configuration differs from the approved development profile.")
        assembled = runner.assemble(doc, spec, "development", metadata["query_size_requested"])
        if before != manifest._file_fingerprints():
            raise ValueError("Canonical inputs changed during verification.")
        for role in ("fit", "selection", "query"):
            _provenance(run_dir / f"{role}_identities.csv", assembled[role][2])
        if metadata["label_accounting"] != assembled["label_accounting"]:
            raise ValueError("Run label accounting differs from reconstructed assignments.")
        if metadata["feature_order"] != list(assembled["schema"].features):
            raise ValueError("Feature order differs from reconstructed schema.")
        if {"domain", "identity", "seqn", "row_index", "id"} & set(metadata["feature_order"]):
            raise ValueError("Provenance columns cannot be model features.")
        prediction = pd.read_csv(run_dir / "predictions.csv.gz", float_precision="round_trip")
        query = assembled["query"]
        if prediction[["domain", "partition", "identity"]].to_dict("records") != query[2].to_dict("records"):
            raise ValueError("Saved prediction identity order differs from the development query.")
        if not np.array_equal(prediction["y_true"].to_numpy(), query[1].to_numpy()):
            raise ValueError("Saved prediction labels differ from the development query.")
        reference = prediction["y_probability"].to_numpy()
        if reference.shape != (len(query[0]),) or not np.isfinite(reference).all() or ((reference < 0) | (reference > 1)).any():
            raise ValueError("Saved probabilities are invalid.")
        if not np.array_equal(prediction["y_pred"].to_numpy(), (reference >= 0.5).astype(int)):
            raise ValueError("Saved threshold decisions do not match probabilities.")
        runner._verify_checkpoint(model, params)
        tick = time.perf_counter()
        predict = _reload(model, run_dir / metadata["model_artifact"], assembled["schema"], metadata, output)
        reload_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        actual = np.asarray(predict(query[0], config["prediction_batch_size"]))
        prediction_seconds = time.perf_counter() - tick
        if actual.shape != reference.shape or not np.isfinite(actual).all() or ((actual < 0) | (actual > 1)).any():
            raise ValueError("Reload probabilities are invalid.")
        rtol, atol = TOLERANCES[model]
        differences = np.abs(actual - reference)
        flips = int(np.count_nonzero((actual >= 0.5) != (reference >= 0.5)))
        passed = bool(np.allclose(actual, reference, rtol=rtol, atol=atol) and flips == 0)
        report.update(
            status="passed" if passed else "failed", run_id=spec["run_id"],
            manifest_sha256=spec["manifest_sha256"], matrix_sha256=metadata["matrix_sha256"],
            model=model, condition=spec["condition"], source=spec["source"], seed=spec["root_seed"],
            budget=spec["budget"], effective_configuration=config, label_accounting=metadata["label_accounting"],
            fit_rows=len(assembled["fit"][0]), selection_rows=len(assembled["selection"][0]),
            query_rows=len(query[0]), identity_and_artifact_checks="passed",
            batch_size=config["prediction_batch_size"], rtol=rtol, atol=atol,
            max_absolute_difference=float(differences.max()), threshold_disagreements=flips,
            reload_seconds=reload_seconds, prediction_seconds=prediction_seconds,
            run_operation_seconds=metadata["operation_seconds"], run_resources=metadata["resources"],
            research_comparison_eligible=False,
        )
        if not passed:
            raise ValueError("Reload probabilities exceed the fixed tolerance or change a threshold decision.")
        report["total_seconds"] = time.perf_counter() - start
        write_json(output / "verification.json", report)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      total_seconds=time.perf_counter() - start)
        write_json(output / "verification.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest_dir, args.run_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("run_id", "status", "max_absolute_difference", "threshold_disagreements")}))


if __name__ == "__main__":
    main()
