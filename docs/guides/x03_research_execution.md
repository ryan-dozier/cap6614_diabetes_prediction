# X03: Candidate A research launch planning, resume, and retry

X03A implements plan-only validation and four logical model jobs; X03B adds
validated resume, inventory classification, and explicit archive-before-retry.
**No research jobs or test evaluations were performed while preparing this
launcher.** X03 experiments remain pending.

## Scope and launch review

Use the [approved freeze](../plans/x01_execution_freeze.json), unchanged
[X01 scientific protocol](../plans/x01_adaptation_protocol.md) and
[E05 analysis conventions](../plans/e05_evaluation_protocol.md). The validated
matrix contains 220 unique runs, 55 per model: 15 real-only, 30 mixed and 10
synthetic-only anchors. Budgets are 250/500/1000, seeds 42-46 and sources
`source`/`kaggle`. Real-only comparators and anchors are reused across contrasts,
not retrained for each source/budget. No pilot outputs become research results.

Preparation began on `feat/x02-adaptation-pipeline`, revision `f8f383d`;
X02 [PR #16](https://github.com/ryan-dozier/cap6614_diabetes_prediction/pull/16)
was open awaiting review. This is not the launch revision. Before any actual
submission, check X02's review/merge status, explicitly choose/review the
deployment revision, transfer these changes through the normal repository
workflow and regenerate plans on Newton. No automatic merge or branch switch
is part of the launcher. Record the actual launch revision in the retained job
logs. A dirty worktree is recorded, not automatically rejected; untracked Slurm
logs and ignored results are not scientific changes. Relevant source files,
profiles, protocols, manifest and prepared inputs are fingerprinted.

Historical future-tense wording in the frozen X01/E05 documents is not a new
approval gate: the current approved freeze and feasibility record supersede it.
Do not edit those fingerprinted protocols merely to update status wording.

## Output layout

Use this single canonical collection for all four jobs:

```text
results/adaptation/x01_v1/
  manifest/                 existing validated assignments
  research/runs/<RUN_ID>/    only canonical per-run artifacts
  execution/plans/           reviewed launch-plan exports
  execution/jobs/<MODEL>_<JOBID>/  plans, commands, logs, timings, status, accounting
```

The existing runner permits roots inside `research/`; a different explicit
`--run-root` must still satisfy that restriction and be shared by all four
model plans. The reporting module rejects unexpected directories in its run
collection. Never put plans, scheduler logs, reports or retry archives inside
`research/runs/`. Reports also need a separate destination, as described in the
[reporting guide](adaptation_reporting.md).

Planning does not create the canonical root or run directories. Job directories
are created exclusively; runners create per-run directories only when execution
begins. Slurm startup errors appear in submission-directory `slurm-<JOBID>.out`.

## Generate and inspect plans (no model execution)

Run from the Newton repository with its explicit environment Python. Create only
the separate plan-export directory, not a research run directory:

```bash
cd /home/ry482284/cap6616/cap6614_diabetes_prediction
PYTHON=/home/ry482284/.conda/envs/synthetic-tabular-ai/bin/python
mkdir -p results/adaptation/x01_v1/execution/plans
PLAN_DIR="$PWD/results/adaptation/x01_v1/execution/plans"
COMMON=(--project "$PWD" --python "$PYTHON"
  --manifest-dir "$PWD/results/adaptation/x01_v1/manifest"
  --freeze "$PWD/docs/plans/x01_execution_freeze.json"
  --run-root "$PWD/results/adaptation/x01_v1/research/runs")
TABPFN=/home/ry482284/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors
TABICL="$PWD/checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt"

"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model catboost --export "$PLAN_DIR/catboost.json"
"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model ft_transformer --export "$PLAN_DIR/ft_transformer.json"
"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model tabpfn --tabpfn-checkpoint "$TABPFN" --export "$PLAN_DIR/tabpfn.json"
"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model tabicl --tabicl-checkpoint "$TABICL" --export "$PLAN_DIR/tabicl.json"

# Optional full-grid inspection plan: not accepted for single-model execution.
"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model all --tabpfn-checkpoint "$TABPFN" --tabicl-checkpoint "$TABICL" \
  --export "$PLAN_DIR/all.json"
```

Exports are exclusive: existing plan files are not overwritten. Use a new plan
filename when regenerating. Without `--export`, the complete plan is printed to
stdout; with it, the compact summary is printed. An empty-collection CatBoost
summary has `model: catboost`, `mode: research`, `run_count: 55`, `pending: 55`,
`existing_run_ids: []`, plus grid counts, run root and plan/freeze SHA-256.

Inspect each exported JSON's `runs`: ordered IDs, exact argument lists,
specifications/subset fingerprints, effective settings and output locations.
Order follows the existing matrix (ascending seeds; existing model, budget and
condition order), filtered by the selected model. No manually maintained run-ID
list exists. Deployment paths are separate from the unchanged scientific
specifications; checkpoint paths may move only when their pinned bytes match.
Use `--project` only for the source tree from which the module is imported;
change directory/run the deployed module to plan a different repository.

Manifest validation reconstructs assignments using permitted train/validation
labels and split identities. It does not parse/materialize canonical test
examples or outcomes. Opaque whole-file hashing, including test-containing
canonical files, remains enabled. Model libraries are not imported for planning.

## Proposed allocations

Based on the [measured feasibility evidence](../results/x02_stage2e_feasibility.md):

| Model job | CPUs | GPU | RAM | Walltime ceiling |
| --- | ---: | --- | --- | --- |
| CatBoost | 4 | None | 64G | 3 hours |
| FT-Transformer | 4 | One H100 PCIe | 64G | 6 hours |
| TabPFN | 4 | One H100 PCIe | 64G | 2 hours |
| TabICL | 4 | One H100 PCIe | 64G | 2 hours |

These are planning choices, not measured requirements or guaranteed runtime
bounds. CatBoost's upper serial allowance was 1.83 h, FT's 3.67 h and each
foundation model's 0.92 h; ceilings add startup/variability margins. Retaining
64G avoids an untested memory reduction despite observed host RSS around
2.70 GiB. GPU allocator peaks are not total device consumption. CPU-only
CatBoost preserves the fixed four-thread CPU scientific profile while avoiding
an unused GPU reservation. Account concurrency is unverified: submit/review
one job at a time initially. Reporting/bootstrap and queue delay are separate,
unmeasured costs; the four ceilings total 13 hours, not an estimated duration.

## Manual submission, only after launch review

The CPU entry point has no GPU request or CUDA/nvidia-smi requirement. The GPU
entry point admits only FT/TabPFN/TabICL, requests the verified H100 resource
and requires both nvidia-smi identity and CUDA availability checks. Common
numerical thread counts are four; offline flags and explicit Python are retained.
Research commands pass the freeze explicitly and never pass `--training-cap`.
Normal training maximums and early stopping/checkpoint selection remain intact.

```bash
# From the repository root, after reviewing the generated deployment plans:
sbatch scripts/x03_newton_cpu.sbatch "$PLAN_DIR/catboost.json"
sbatch --time=06:00:00 scripts/x03_newton_gpu.sbatch ft_transformer "$PLAN_DIR/ft_transformer.json"
sbatch --time=02:00:00 scripts/x03_newton_gpu.sbatch tabpfn "$PLAN_DIR/tabpfn.json"
sbatch --time=02:00:00 scripts/x03_newton_gpu.sbatch tabicl "$PLAN_DIR/tabicl.json"
```

These are four separate manual decisions, not an instruction to submit all at
once. GPU jobs may optionally add `--exclude=evc40` while the prior "No devices
were found" incident remains unresolved. This is a temporary submission-time
precaution, not proof the node is permanently broken; confirm with cluster
support and remove the exclusion when resolved. Never bypass GPU checks.

`PROJECT_DIR`, `PYTHON` and `JOB_ROOT` environment overrides are supported by the
batch wrapper. Python/project must agree with the reviewed plan. Manifest,
freeze, run-root and checkpoint locations are supplied through the plan CLI.
The launcher reconstructs and compares the entire plan before execution, checks
selected-output absence, and rechecks fingerprints before every run. Changed
inputs/code/settings/paths/commands require a new reviewed plan, not silent
substitution. Checkpoint fingerprints are validated before any model runs.

## Monitoring and accounting

Replace the job ID/model, preserving the batch and other step accounting:

```bash
JOB=REPLACE_WITH_JOB_ID
MODEL=catboost
JOB_DIR="results/adaptation/x01_v1/execution/jobs/${MODEL}_${JOB}"
squeue -j "$JOB"
scontrol show job "$JOB"
sacct -j "$JOB" --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,ReqTRES%80,AllocTRES%80,MaxRSS,MaxRSSNode,MaxRSSTask -P
# After completion, retain accounting outside the run collection:
sacct -j "$JOB" --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,ReqTRES%80,AllocTRES%80,MaxRSS,MaxRSSNode,MaxRSSTask -P > "$JOB_DIR/slurm_accounting.txt"
cat "$JOB_DIR/status.json"
tail -n 100 "$JOB_DIR/job.log"
tail -n 100 slurm-"$JOB".out
```

Job evidence includes plan/freeze fingerprints, revision/worktree state,
environment versions, allocation, GPU identity (GPU models), each command as
JSON, per-run stdout/stderr and `/usr/bin/time -v` evidence. A failed command
stops subsequent runs and propagates its nonzero exit status. Slurm timeout,
OOM or node loss may prevent final status writing; missing/incomplete status
is not success. Consult scheduler accounting and retained logs.

## Existing outputs, inventory, resume, and archive-before-retry (X03B)

Ordinary (non-resume) plans keep the safe default: any existing selected output
refuses the whole model job and lists the affected IDs. Nothing is skipped,
archived, deleted or overwritten. Recovery is explicit and split into three
deliberate steps — inspect, archive (only if needed), then regenerate a resume
plan — each separate from execution.

### Inventory and status inspection

Classify every expected output location for one model without reading any
prediction values:

```bash
"$PYTHON" -m src.experiments.adaptation_launch inventory "${COMMON[@]}" --model catboost
```

Statuses per run ID:

- `pending`: missing; eligible to execute.
- `verified_complete`: passes the full research artifact contract below;
  eligible to skip under an explicit resume plan.
- `failed` / `incomplete`: blocks execution until explicitly archived
  (includes outputs still marked `running`; confirm the owning job ended first).
- `incompatible`: corrupted or contract-violating; blocks with a per-run
  reason and stays blocked for manual investigation.

Before a completed run may be skipped it must pass, in order: the runner's
inventory contract (exact run identity, research mode, approved specification,
required artifacts present, every recorded artifact SHA-256 re-verified), then
the deeper research checks mirrored from the reporting gate — current matrix
fingerprint, exact approved-freeze match, canonical source/split fingerprints,
run-specification artifact equality, effective scientific configuration equal to
the approved profile (no training cap), pinned checkpoint fingerprints for
foundation models, approved feature order, fit/selection/query participant
provenance re-derived from the manifest and canonical test identities, and
label accounting. Prediction *values* are never parsed for resume decisions:
their bytes are integrity-covered by the verified recorded hashes, and the
final reporting gate revalidates prediction contents. Launcher commits are
recorded but not required to match across completed runs; only the frozen
scientific contracts gate compatibility. Unexpected entries in the canonical
run collection remain errors.

### Resume-plan generation

```bash
"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model catboost --resume \
  --export "$PLAN_DIR/catboost_resume_$(date +%Y%m%dT%H%M%S).json"
```

A resume plan retains the full 55-run model scope and records, per run ID, the
verified-complete runs to skip (with their pinned metadata fingerprints), the
missing runs to execute, and any blocking IDs with reasons. An invalid existing
run is never silently reclassified as missing. Submission uses the same batch
wrappers with the resume-plan file. Execution revalidates by reconstructing the
entire plan against current inputs and output state: changed completed-run
evidence, newly appeared pending outputs, changed scientific inputs/code, or
changed commands make the plan stale and refuse execution. Any blocked ID also
refuses execution. If every selected run is verified complete, the job is an
explicit recorded no-op (`status.json` has `no_op: true`) and invokes no model
or preflight command.

During execution, each pending run keeps the X03A per-run fingerprint and
output-absence rechecks, and the runner's exclusive-creation/no-overwrite
behavior. A zero subprocess exit is not success: each newly finished run is
verified against the same artifact contract before the job continues, and
`status.json` distinguishes `skipped_verified_run_ids`, `completed_run_ids`
(new this job), `invalid_run_ids`, `remaining_run_ids`, and `current_run_id`.
Concurrent same-model execution, or archival while a job is running, is
unsupported; run one job per model at a time.

### Inspecting blocked runs

Read the inventory output or the resume plan's `blocking_reason` fields; the
plan summary also lists a `blocked` map. For a failed run, consult the retained
per-run log, `.time` file, and `metadata.json` error before deciding anything:

```bash
jq '.runs[] | select(.status != "pending" and .status != "verified_complete")' inventory.json
```

### Explicit archive-before-retry

Archival is a separate, deliberate operation; planning and resume never archive
automatically. It requires exact run IDs, an explicit destination outside the
canonical run collection, and a documented reason:

```bash
"$PYTHON" -m src.experiments.adaptation_launch archive \
  --manifest-dir "$PWD/results/adaptation/x01_v1/manifest" \
  --freeze "$PWD/docs/plans/x01_execution_freeze.json" \
  --run-root "$PWD/results/adaptation/x01_v1/research/runs" \
  --archive-dir "$PWD/results/adaptation/x01_v1/execution/archive/$(date +%Y%m%dT%H%M%S)" \
  --reason "Job 123456 hit its walltime ceiling during this run; retrying unchanged inputs." \
  --run-id catboost_mixed_source_B500_S1600_seed43
```

Each ID is validated against the approved matrix and canonical run root, and its
state is rechecked immediately before moving. The operation refuses
verified-complete runs, missing outputs, symlinks, unsafe paths, destinations
inside the run collection, and existing destinations; nothing is deleted or
overwritten, and the original artifacts move intact to `<archive-dir>/<run_id>`
with a sibling `<run_id>.archive.json` recording the run ID, original and
archive locations, reason, classification, timestamp, and available metadata
provenance. Incompatible/corrupted outputs stay blocked for manual
investigation unless `--allow-incompatible` records a deliberate, documented
decision. Failed training commands are never retried automatically.

### Regenerating a resume plan after archival

Archival changes output state, so every earlier plan for that model is stale by
design and will refuse execution. Generate and review a fresh resume plan (the
archived run reappears as `pending`), then submit manually:

```bash
"$PYTHON" -m src.experiments.adaptation_launch plan "${COMMON[@]}" \
  --model catboost --resume --export "$PLAN_DIR/catboost_resume_after_archive.json"
sbatch scripts/x03_newton_cpu.sbatch "$PLAN_DIR/catboost_resume_after_archive.json"
```

### Final completion evidence

A model job is finished when its `status.json` reports `completed` with no
remaining/invalid runs, and `inventory` reports all 55 runs
`verified_complete`. The 220-run grid is complete when all four models reach
that state. Keep job logs, plans, accounting, and archives outside
`research/runs/`. This launcher-level verification does not evaluate research
outcomes; final E05 reporting separately revalidates predictions and remains a
pending gate.
