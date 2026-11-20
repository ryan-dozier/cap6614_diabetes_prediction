"""Extract resource evidence from a Stage 2E pilot without predictive metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def summarize(root: Path) -> dict[str, Any]:
    plan = (root / "plan.txt").read_text().splitlines()
    if not plan or len(plan) != len(set(plan)) or any(not re.fullmatch(r"[A-Za-z0-9_]+", item) for item in plan):
        raise ValueError("Invalid pilot run plan.")
    runs = []
    for run_id in plan:
        run_dir = root / "runs" / run_id
        metadata_path = run_dir / "metadata.json"
        item: dict[str, Any] = {"run_id": run_id, "status": "not_started", "error": None}
        time_path = root / "logs" / f"{run_id}.time"
        if time_path.exists():
            timing = time_path.read_text()
            rss = re.search(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$", timing, re.MULTILINE)
            elapsed = re.search(r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", timing, re.MULTILINE)
            item["time_max_rss_kib"] = int(rss.group(1)) if rss else None
            item["time_elapsed"] = elapsed.group(1) if elapsed else None
        else:
            item.update(time_max_rss_kib=None, time_elapsed=None)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("run_id") != run_id or metadata.get("mode") != "development":
                raise ValueError(f"Unexpected run identity or mode: {metadata_path}")
            spec = metadata["specification"]
            config = metadata.get("effective_configuration", {})
            model = metadata.get("model", {})
            training = model.get("training") or {}
            history = training.get("training_history") or []
            log_path = root / "logs" / f"{run_id}.log"
            catboost_iterations = None
            if spec["model"] == "catboost" and log_path.exists():
                observed = re.findall(r"^(\d+):\s+test:", log_path.read_text(), re.MULTILINE)
                if observed:
                    catboost_iterations = max(map(int, observed)) + 1
            selection = metadata.get("label_accounting", {})
            item.update(
                status=metadata.get("status", "unknown"),
                error=metadata.get("error"),
                model=spec["model"], condition=spec["condition"], source=spec["source"],
                budget=spec["budget"], seed=spec["root_seed"],
                training_cap=config.get("development_training_cap"),
                catboost_iterations_observed_in_log=catboost_iterations,
                catboost_best_iteration=model.get("best_iteration"),
                catboost_saved_trees=model.get("fitted_tree_count"),
                ft_epochs_executed=training.get("epochs_executed"),
                ft_stopping_reason=training.get("stopping_reason"),
                ft_validation_seconds=sum(row["validation_duration_seconds"] for row in history) if history else None,
                label_accounting=selection,
                validation_population=metadata.get("validation_population"),
                operation_seconds=metadata.get("operation_seconds"),
                total_duration_seconds=metadata.get("total_duration_seconds"),
                resources=metadata.get("resources"),
                artifact_bytes=sum(path.stat().st_size for path in run_dir.iterdir() if path.is_file()),
            )
        elif run_dir.exists():
            item.update(status="incomplete", error="Run directory exists without metadata.json")
        runs.append(item)
    return {"schema_version": "x02_stage2e_summary_v1", "pilot_directory": str(root), "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.pilot_directory), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
