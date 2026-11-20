# X02 Stage 2E: Newton pilot and resource assessment

Status: **prepared, not executed or approved**. Newton model feasibility and
the final protocol freeze remain pending. This is not authorization for research
test execution. X02 and Stage 2E remain open.

## Verified setup (reported by Ryan)

Repository `/home/ry482284/cap6616/cap6614_diabetes_prediction`, branch
`feat/x02-adaptation-pipeline`; Python
`/home/ry482284/.conda/envs/synthetic-tabular-ai/bin/python`. Available module:
`anaconda/anaconda-2024.10`; the batch job uses the executable directly.
Job 1225847 on evc49 used account `course_cap6614`, partition/QOS `normal`,
one `gpu:nvidia_h100_pcie`, one node/task, four CPUs and 16G. It reported an
H100 PCIe (81,559 MiB), driver 580.178.04, Python 3.11.17 and PyTorch
2.14.1+cu126. Four model-library imports and CUDA matrix multiplication passed.
The allocation was released; neither adapter fitting nor pretrained checkpoint
loading was exercised.

`requirements-local.txt` installation and `pip check` passed;
`initial_setup.sh` completed with exit 0. Login-node thread limits were needed.
The transferred original manifest validated with 70 subsets/five repetitions
and assignment SHA-256
`fea3d3c14d7d32fb2ee371981b2e2c907cc9b1083ac56f4dfd5fdedaaacfb59d`.
The checkpoint hashes matched the pinned values:

| Model | Verified path | SHA-256 |
| --- | --- | --- |
| TabPFN | `/home/ry482284/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors` | `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3` |
| TabICL | `checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt` relative to the repository | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |

This 16G job proves neither host-RAM sufficiency for model runs nor runtime,
peak GPU usage, or concurrency. Do not repeat setup or data preparation.
The [Stage 2C record](../results/x02_stage2c_validation.md) is a local smoke
record, not a basis for Newton timing estimates.

## Pilot design

Each run uses seed 42, its exact Stage 2A assignments, approved X01 profiles,
precision, and checkpoint hashes. Only task-trained models receive the existing
development cap of five iterations/epochs. No fitting/context rows or permitted
selection rows are downsampled. Each allocation runs sequentially and stops at
the first failure. No pilot predictive metric may select configurations, models
or budgets.

| Wave | Runs | Purpose |
| --- | --- | --- |
| First, one job, 64 queries | CatBoost mixed Original B250; FT mixed Playground B250; TabPFN real-only B250; TabICL mixed Original B250 | Short integration and initial resource check across all four actual adapters, including both checkpoint loads and both synthetic sources. |
| Later, one separate job per model, 446 queries | B1000 real-only and both mixed sources; B500 mixed Original; both synthetic-only anchors (six/model, 24 total) | Largest real fit and selection sizes; source differences at fixed 2,000 fit/context rows; intermediate size; source-only selection paths; longer-query inference. |

`B250/B500/B1000` use 200/400/800 real fitting/context rows, with
50/100/200 reserved validation rows and 1,800/1,600/1,200 synthetic rows in
mixed runs. Task-trained positive-budget runs select on those 50/100/200 rows;
frozen models reserve but do not use them. Both task-trained synthetic anchors
use the complete same-source validation partition: 15,000 Original and 105,000
Playground rows as recorded in the existing assignments. FT evaluates it each
epoch, so five epochs can mean 75,000 or 525,000 selection-row evaluations;
actual counts and timing must come from the run. The anchors for both frozen
models measure 2,000-row context construction and inference without selection.
The later wave contains both anchor sources for every model, but does not repeat
every B250/B500 combination. Unsupported extrapolations remain uncertain.

Queries are distinct permitted NHANES validation rows outside the run's reserved
selection subset, with actual counts in `label_accounting`. At B1000 only
646 - 200 = 446 such rows remain. First-wave queries are 64; later queries are
446 even when more are available. No duplication simulates the 646-person test
cohort. These development labels are additional supervision outside the
research budget; artifacts are marked ineligible for research comparisons.
The validator and runner hash the whole canonical files as opaque bytes for
integrity, including bytes belonging to test rows. Ryan explicitly allowed this
integrity hashing; no test features or outcomes are parsed for the pilot, used
for model fitting, selection, prediction, or metrics. Split identity metadata
is also checked by the existing validator.

The pilot measures actual capped fitting/context preparation, query inference,
assembly, saving, metrics/artifact writing, host RSS, and available allocator
peaks. Full training schedules, early-stopping behavior, 646-row inference,
reload verification, full-grid reporting/bootstrap, retries, queue wait and
concurrency are **not measured by these runs**. FT validation time is separated
in its epoch history; CatBoost fit includes selection. Never infer the actual
full-run stopping epoch from a five-epoch/iteration cap.

## Allocation and first submission

The batch file requests one H100, four CPUs, **64G RAM and one hour** for the
first wave, with explicit account, QOS, partition and walltime. 64G is a
precautionary starting request, not a measured sufficiency claim: large input
assembly, foundation-model preprocessing and later full-source validation can
exceed it. The one-hour limit bounds the first check, not an asserted runtime.
Later timing jobs use the same resources but request **12 hours per model** as
an unverified upper allocation; even this may be too short, particularly for
FT's complete Playground selection path. Review first-wave evidence and cluster
limits before deciding whether to submit later jobs or change resource requests.
No CPU/GPU concurrency has been verified.

After pushing/pulling the scripts, on Newton:

```bash
cd /home/ry482284/cap6616/cap6614_diabetes_prediction
sbatch scripts/x02_newton_pilot.sbatch first
```

Record the job ID printed by `sbatch`. Default paths are those verified above.
Set absolute `PROJECT_DIR`, `PYTHON`, `MANIFEST_DIR`, `TABPFN_CHECKPOINT`, or
`TABICL_CHECKPOINT` in the submission environment to override them; e.g.
`PROJECT_DIR=/absolute/repo PYTHON=/absolute/env/bin/python sbatch ...`.
Use an explicit project environment executable; no interactive Conda activation
is needed. The profiles fix CatBoost to four threads, so the script rejects a
different CPU count rather than changing the science profile.

The job checks required files, software/CUDA availability, both checkpoint hashes,
the exact manifest fingerprint and full manifest validation before execution.
It sets numerical thread limits and offline flags, captures commands, software,
Git state, GPU/driver and `scontrol` job details, and refuses an existing wave
directory. Results, per-run logs, `/usr/bin/time -v` outputs and `summary.json`
are under:

```text
results/adaptation/x01_v1/development/stage2e_first_<JOBID>/
```

Slurm's submission-directory `slurm-<JOBID>.out` captures early failures even
before the evidence directory is created. A failed/interrupted run is retained;
never reuse its output directory. Slurm timeout/OOM or node failure can prevent
the exit trap or runner from writing final status; treat `running`, missing
metadata, or absent summary as incomplete, not success.

## Review gate and later submissions

Review the first job's exit, each adapter's actual fit/loading, resource use,
failure diagnostics and any non-default environment before proceeding. After
review and an explicit manual decision, submit the following individually; do
not submit the entire set automatically:

```bash
sbatch --time=12:00:00 scripts/x02_newton_pilot.sbatch timing-catboost
sbatch --time=12:00:00 scripts/x02_newton_pilot.sbatch timing-ft_transformer
sbatch --time=12:00:00 scripts/x02_newton_pilot.sbatch timing-tabpfn
sbatch --time=12:00:00 scripts/x02_newton_pilot.sbatch timing-tabicl
```

These are proposed timing jobs, not automatically approved after first-wave
success. Use their distinct job IDs when retrieving evidence. The script runs
the six listed conditions in order and aborts on the first failure. If evidence
indicates likely timeout, OOM or unavailable cluster limits, revise the plan
before submitting; retain failed evidence and rerun only in a new job directory.

## Evidence retrieval

From the submission directory, substitute the numeric job ID and the correct
wave. Include the batch and other job steps; do not use `sacct -X`, which hides
step-level MaxRSS. Accounting may be delayed or disabled:

```bash
JOB=REPLACE_WITH_JOB_ID
WAVE=first
ROOT="results/adaptation/x01_v1/development/stage2e_${WAVE}_${JOB}"
sacct -j "$JOB" --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,ReqTRES%80,AllocTRES%80,MaxRSS,MaxRSSNode,MaxRSSTask -P
scontrol show job "$JOB"
cat "$ROOT/summary.json"
sed -n '1,/^Running:/p' "$ROOT/job.log"
```

Return the full `sacct` output, `scontrol` output if still available, the compact
`summary.json`, and the displayed startup section of `job.log`. The summary
includes each run's status; `sacct` supplies the job's final status. If no root
exists, return `slurm-$JOB.out` instead. For failed or missing results,
also return the failing run's command, `.time`, the last 100 lines of its
`.log`, and the last 100 lines of `job.log` or `slurm-$JOB.out`:

```bash
tail -n 100 "$ROOT/job.log"
tail -n 100 slurm-"$JOB".out
# Replace RUN with the failing run ID shown in summary.json or plan.txt:
cat "$ROOT/logs/RUN.command" "$ROOT/logs/RUN.time"
tail -n 100 "$ROOT/logs/RUN.log"
```

If `summary.json` is missing after interruption but `plan.txt` exists, reconstruct
it to standard output without overwriting evidence:

```bash
/home/ry482284/.conda/envs/synthetic-tabular-ai/bin/python scripts/x02_pilot_summary.py "$ROOT"
```

The summary excludes predictive metrics and includes status/errors, actual
query/selection label counts, per-operation and total seconds, artifact bytes,
and runner resource metadata. CatBoost iteration counts are the highest
zero-based iteration printed plus one, not the selected iteration or saved tree
count; if the log lacks it, the count is null. FT records actual executed epochs,
stopping reason and sum of per-epoch selection seconds. The five-iteration
CatBoost cap is below patience 100, but the log is the execution evidence.
`/usr/bin/time` MaxRSS is a per-process high-water measurement, not necessarily
the summed memory of all children; `sacct` step MaxRSS is also not a guaranteed
aggregate of concurrent processes. Blank, null, missing, unsupported or delayed
accounting means **unavailable**, not zero. PyTorch allocator peaks are
process-local allocated/reserved memory, not total GPU use. `nvidia-smi` here
records device identity/capacity, not a sampled peak. Save full logs and failed
artifacts locally; only excerpts are needed for review.

## Resource decision after measurements

The full initial matrix is **60 real-only + 120 mixed + 40 synthetic-only =
220 unique runs**, with five seeds for every model/budget/source combination.
Real-only runs are reused for both source comparisons and anchors are reused
across budgets; never count contrast references as new jobs.

After all relevant evidence arrives, build a per-model/condition/budget cost
table from the observed assembly, fitting/context, prediction, save and artifact
times. Include manifest revalidation and batch startup overhead from total
elapsed time; avoid adding components twice. Include actual source-validation
workload and query sizes. Estimate 646-query inference separately with explicit
uncertainty: 446 is shorter, and fixed per-batch overhead, TabPFN batch 1,024,
TabICL batch 128, and remainder batches rule out an unquestioned linear
throughput assumption. Make conservative ranges for unmeasured B250/B500
conditions and seed variability, with observed ranges and explicit margins.

For capped task-trained fits, separate startup from observed epoch/iteration
work where possible. Explore conservative maximum-profile scenarios up to
CatBoost 1,000 iterations and FT 200 epochs, including selection each epoch,
but label scaling as uncertain, not as measured stopping behavior or a firm
upper bound. Obtain additional permitted development timing if needed rather
than asserting a stopping point. Frozen-model costs are not multiplied by task
training epochs. Add reporting/bootstrap, retry, failure and queue contingencies
as separate estimates; they are currently unmeasured. Use observed artifact
sizes for storage ranges, including predictions, logs, checkpoints, verification,
reports and retained failures, and distinguish bytes per run from shared inputs.

Compute serial GPU-allocation hours and CPU-hours from allocated resources, and
separate wall-clock scenarios for one job at a time and any concurrency
**actually verified** under account limits. Queue time and scaling remain
unknown until observed. Report uncertainties, assumptions and unresolved
measurements, not laptop-derived or fabricated Newton timings. If the measured
workload is infeasible, retain all four models and propose reducing the shared
budget grid first under R06, preserving anchors and repetitions where possible.
Any reduction and its scientific consequences require a recorded decision
before affected test evaluation. Freeze approval, final configuration/inventory,
and research execution remain pending.

## Local preparation checks

The pilot ID list was checked against the existing validated 220-run matrix.
Representative development assembly, without fitting any models, confirmed
64 queries for the first B250 case and 446 queries for B500, B1000, and both
task-trained anchor sources. These checks reconstructed selection populations
of 50/100/200, 15,000 and 105,000 rows as appropriate. These are engineering
checks, not Newton feasibility measurements.
