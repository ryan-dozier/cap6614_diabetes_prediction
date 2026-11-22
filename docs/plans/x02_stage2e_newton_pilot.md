# X02 Stage 2E: Newton pilot and resource assessment

Status: **in progress; final feasibility and protocol freeze remain pending**.
The transferred summaries show that all 28 existing Newton pilot runs completed
successfully; their local artifact hashes were validated. Evidence is under
`results/adaptation/x01_v1/development/stage2e_first_1225886/` and the four
`stage2e_timing-*` roots (jobs 1225910, 1225888, 1225895, and 1225898).
CatBoost and FT-Transformer training in those waves was capped at five
iterations/epochs, leaving full training and early-stopping costs uncertain.
This document adds one manually submitted full-training development wave; it is
not authorization for research test execution. X02 and Stage 2E remain open.

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
precision, and checkpoint hashes. Existing pilot waves cap task-trained models
at five iterations/epochs; the full-training wave applies no development cap.
No fitting/context rows or permitted selection rows are downsampled. Each
allocation runs sequentially and stops at the first failure. No pilot predictive
metric may select configurations, models or budgets.

| Wave | Runs | Purpose |
| --- | --- | --- |
| First, completed job 1225886, 64 queries | CatBoost mixed Original B250; FT mixed Playground B250; TabPFN real-only B250; TabICL mixed Original B250 | Short integration and initial resource check across all four actual adapters, including both checkpoint loads and both synthetic sources. |
| Completed timing jobs 1225910/1225888/1225895/1225898, 446 queries | B1000 real-only and both mixed sources; B500 mixed Original; both synthetic-only anchors (six/model, 24 total) | Largest real fit and selection sizes; source differences at fixed 2,000 fit/context rows; intermediate size; source-only selection paths; longer-query inference. |
| Full-training, one manual job, 446 queries | `catboost_mixed_kaggle_B1000_S1200_seed42`, `catboost_synthetic_only_kaggle_B0_S2000_seed42`, `ft_transformer_mixed_kaggle_B1000_S1200_seed42`, `ft_transformer_synthetic_only_kaggle_B0_S2000_seed42` | Measure normal training/early-stopping costs and the largest anchor validation workload. No development training cap; unchanged profiles permit up to 1,000 CatBoost iterations and 200 FT epochs. |

The full-training wave retains the original manifest assignments, development
mode, four-thread/resource profile, and early-stopping/checkpoint-selection
rules. It does not tune models or supply research evidence. The synthetic-only
Kaggle anchors retain the complete permitted 105,000-row validation population.

`B250/B500/B1000` use 200/400/800 real fitting/context rows, with
50/100/200 reserved validation rows and 1,800/1,600/1,200 synthetic rows in
mixed runs. Task-trained positive-budget runs select on those 50/100/200 rows;
frozen models reserve but do not use them. Both task-trained synthetic anchors
use the complete same-source validation partition: 15,000 Original and 105,000
Playground rows as recorded in the existing assignments. FT evaluates it each
epoch, so the existing five-epoch runs can mean 75,000 or 525,000 selection-row
evaluations; actual counts and timing must come from each run. The anchors for
both frozen models measure 2,000-row context construction and inference without
selection.
The later wave contains both anchor sources for every model, but does not repeat
every B250/B500 combination. Unsupported extrapolations remain uncertain.

Queries are distinct permitted NHANES validation rows outside the run's reserved
selection subset, with actual counts in `label_accounting`. At B1000 only
646 - 200 = 446 such rows remain. First-wave queries are 64; later and
full-training queries are 446 even when more are available. No duplication
simulates the 646-person test cohort. These development labels are additional
supervision outside the research budget; artifacts are marked ineligible for
research comparisons.
The validator and runner hash the whole canonical files as opaque bytes for
integrity, including bytes belonging to test rows. Ryan explicitly allowed this
integrity hashing; no test features or outcomes are parsed for the pilot, used
for model fitting, selection, prediction, or metrics. Split identity metadata
is also checked by the existing validator.

The existing capped runs measure capped fitting/context preparation, query
inference, assembly, saving, metrics/artifact writing, host RSS, and available
allocator peaks. The full-training wave additionally measures normal training
and early-stopping costs for its four configurations. 646-row inference, reload
verification, full-grid reporting/bootstrap, retries, queue wait and
concurrency remain **unmeasured**. FT validation time is separated in its epoch
history; CatBoost fit includes selection. Never infer a full-run stopping point
from the five-iteration/epoch cap.

## Allocation and full-training submission

The batch file requests one H100, four CPUs, **64G RAM and one hour by default**,
with explicit account, QOS, partition and walltime. 64G is a
precautionary starting request, not a measured sufficiency claim: large input
assembly, foundation-model preprocessing and later full-source validation can
exceed it. The one-hour limit bounds the first check, not an asserted runtime.
The full-training wave uses the same resource profile and must be submitted with
an explicit **12-hour walltime ceiling**. This is a precautionary allocation
limit, not an estimated runtime. No CPU/GPU concurrency has been verified.

After pushing/pulling the scripts, submit the new wave manually on Newton:

```bash
cd /home/ry482284/cap6616/cap6614_diabetes_prediction
sbatch --time=12:00:00 scripts/x02_newton_pilot.sbatch full-training
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
results/adaptation/x01_v1/development/stage2e_full-training_<JOBID>/
```

Slurm's submission-directory `slurm-<JOBID>.out` captures early failures even
before the evidence directory is created. A failed/interrupted run is retained;
never reuse its output directory. Slurm timeout/OOM or node failure can prevent
the exit trap or runner from writing final status; treat `running`, missing
metadata, or absent summary as incomplete, not success.

## Review gate

The full-training wave runs the four listed configurations sequentially and
aborts on the first failure. Use its unique job ID to retrieve evidence; retain
failed evidence and rerun only in a new job directory. Review the four results
before updating feasibility estimates. Feasibility estimates, final freeze
approval and Stage 2E completion remain pending until this wave is reviewed.

## Evidence retrieval

From the submission directory, substitute the numeric job ID and the correct
wave. Include the batch and other job steps; do not use `sacct -X`, which hides
step-level MaxRSS. Accounting may be delayed or disabled:

```bash
JOB=REPLACE_WITH_JOB_ID
WAVE=full-training
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
and runner resource metadata. It explicitly reports whether a development cap
was applied; `development_training_cap: null` with status `none` means no
development cap. FT records actual executed epochs, selected epoch, stopping
reason and sum of per-epoch selection seconds. CatBoost's selected iteration
(zero-based) and saved tree count are separate from actual iterations executed.
The summary reports CatBoost's actual count only when it is established by the
successful completion of a cap below early-stopping patience or a logged final
configured iteration; otherwise the count is null. Do not infer total CatBoost
iterations from saved tree count. Missing or incomplete execution evidence means
unavailable, not zero.
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

For task-trained fits, separate startup from observed epoch/iteration work where
possible. The full-training wave measures the normal training/early-stopping
cost for two CatBoost and two FT configurations, including the complete Kaggle
anchor validation workload. These runs do not tune models or supply research
evidence. Frozen-model costs are not multiplied by task training epochs. Add
reporting/bootstrap, retry, failure and queue contingencies as separate
estimates; they are currently unmeasured. Use observed artifact
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
