# X02 Stage 2E — Newton feasibility and proposed execution freeze

Status: **Approved by Ryan Dozier on 2026-11-22 (America/New_York).**
Prepared from transferred Newton evidence. The approved execution record
authorizes the defined research grid through the existing gate; no research
job was launched as part of approval. Candidate B remains deferred.

## Approved decision

Retain all four models (CatBoost, FT-Transformer, TabPFN, TabICL), all three
available-real-label budgets (250, 500, 1,000), both synthetic sources, and
five seeds (42–46). Retain the approved profiles, nested assignments, fixed
2,000-row mixtures, label accounting, checkpoint selection and E05 analysis.
There are 60 real-only, 120 mixed, and 40 synthetic-only runs: **220 unique
runs, 55 per model**. Anchors and real-only comparators are reused rather than
retrained for each contrast. No pilot predictive metric informed this decision.

The [approved machine-readable freeze](../plans/x01_execution_freeze.json)
contains the exact 220 run IDs and current protocol, manifest, matrix, profile,
checkpoint and evidence fingerprints. The original draft is retained as a
superseded, unapproved historical snapshot; use the approved file for execution.

## Measured evidence

Execution was performed by Ryan Dozier on Newton, using scripts prepared with
AI assistance. This audit checks transferred artifacts; it does not rerun
models or calculate predictive metrics. All 32 completed run directories pass
inventory/artifact SHA-256 checks and remain research-ineligible development
runs. Opaque whole-file integrity hashing is permitted; pilot modeling and
query evaluation used train/validation examples, not canonical test examples.

| Job | Wave | Completed runs | Allocated elapsed | Exit |
|---|---|---:|---:|---|
| 1225886 | First | 4 | 2:00 | 0:0 |
| 1225888 | FT timing | 6 | 1:35 | 0:0 |
| 1225895 | TabPFN timing | 6 | 1:32 | 0:0 |
| 1225898 | TabICL timing | 6 | 1:21 | 0:0 |
| 1225910 | CatBoost timing | 6 | 2:40 | 0:0 |
| 1231168 | Full-training startup failure | 0 | 0:02 | 6:0 |
| 1231230 | Full-training retry | 4 | 4:07 | 0:0 |

Successful allocated elapsed totals **13:15**, or **13:17 including the failed
startup**. All jobs reserved one H100, four CPUs and 64 GiB host RAM. This is
about 0.221 allocated GPU-hours and 0.886 allocated CPU-hours including the
failure; CatBoost itself executed on CPU. These are pilot allocation costs,
not measured GPU busy time or full-grid costs. Queue delays are separate.

Environment: Python 3.11.17, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1,
CatBoost 1.2.10, PyTorch 2.14.1+cu126, rtdl-revisiting-models 0.0.2,
TabPFN 9.2.0, TabICL 2.2.0. H100 PCIe capacity was 81,559 MiB as reported
by nvidia-smi. Observed driver versions include 580.105.08 and 580.178.04.
The capped pilots used revision 959a0fe; full-training used 6e53122459a0198625552d3290be1d9853967e47.
Untracked Slurm output files were logged; a dirty-worktree flag alone is not
proof of changed model code.

The first attempt at full training failed on evc40 when nvidia-smi reported
"No devices were found", before Python/model execution. Retrying with
`--exclude=evc40` succeeded on evc48. The root cause is unconfirmed; preserve
this failure as infrastructure evidence. Continue explicit GPU preflight and
avoid that node until its availability is confirmed; do not bypass the check.

## Training and memory observations

All first/timing task-trained runs were capped at five iterations/epochs.
The four follow-up runs omitted that development cap and retained normal
maximums and early stopping:

| Model / condition (Playground) | Process elapsed | Fit time | Training outcome |
|---|---:|---:|---|
| CatBoost mixed B1000 | 38.54 s | 7.88 s | Early stop; selected iteration 304; 305 saved trees |
| CatBoost synthetic-only | 8.82 s | 2.51 s | Early stop; selected iteration 33; 34 saved trees |
| FT mixed B1000 | 58.21 s | 46.44 s | 62 epochs; selected epoch 42 |
| FT synthetic-only | 23.83 s | 12.27 s | 25 epochs; selected epoch 8 |

CatBoost's exact executed counts are unavailable from sparse progress logging;
the selected iteration and retained tree count must not be used as counts of
training work. FT stopping patience and raw-AUC checkpoint selection are
separate policies, so selected epoch need not equal executed epochs minus
patience. The anchor used all 105,000 same-source validation rows. Earlier
source-anchor checks used 15,000 rows.

Frozen-model timing waves used 446 development queries: TabPFN process elapsed
12.91–14.15 s (prediction 0.43–0.63 s); TabICL 10.55–12.88 s (prediction
0.95–1.83 s). Both exercised 2,000-row mixed/anchor contexts and left reserved
real selection labels unused. TabICL reported no detected recovery or swap
fallback; that is not proof of identical internal scheduling.

The largest observed per-process host RSS across the pilots was about
2.70 GiB, and largest PyTorch reserved GPU memory about 2.50 GiB. Slurm sampled
MaxRSS may miss brief peaks and differ from /usr/bin/time. Allocator peaks are
not total GPU use. No tested configuration exhausted the 64 GiB host/H100
allocation. Retaining that already-tested allocation initially avoids an
untested memory reduction; it is not a claim that 64 GiB is necessary.

## Planning estimates, not measured bounds

Use one job at a time as the baseline: account concurrency limits have not
been established. The following rounded **per-run execution allowances**
include imports, integrity checks, assembly, training/context, prediction and
saving. They are engineering scenarios, not confidence intervals or guaranteed
upper bounds. Do not add those components a second time.

| Model | Runs | Allowance per run | Serial allocation allowance |
|---|---:|---:|---:|
| CatBoost | 55 | 30–120 s | 0.46–1.83 h |
| FT-Transformer | 55 | 60–240 s | 0.92–3.67 h |
| TabPFN | 55 | 20–60 s | 0.31–0.92 h |
| TabICL | 55 | 20–60 s | 0.31–0.92 h |
| Total | 220 | — | **1.99–7.33 h** |

Rationale: use margins over observed process times and normal-limit fits.
For FT, a simple 200/62 scaling of the mixed fit gives approximately 150 s
plus observed process overhead, within the 240 s planning allowance. Scaling
the 105,000-validation anchor from 25 to 200 epochs gives approximately 98 s
fit time. These linear scenarios are uncertain, include startup in fitting,
and are not empirical upper bounds. For CatBoost, the five-iteration mixed
and anchor fits of roughly 0.16–0.31 s suggest approximately 32–62 s if scaled
to 1,000 iterations, before startup; the 120 s allowance adds margin. The
5.01 s real-only startup fit and 40.50 s first-wave FT fit show why short
measurements cannot give firm bounds or known stopping epochs.

Foundation-model allowances cover observed startup variability and a larger
646-query research cohort. We have not evaluated that cohort during pilots.
TabPFN's query batch is 1,024; TabICL's is 128, so 446 versus 646 queries changes
TabICL from four to six batches. Unmeasured query shapes, seeds and data mixes
prevent claiming proportional throughput or identical peak memory.

If all runs retain one GPU/four CPUs, these scenarios imply 1.99–7.33 allocated
GPU-hours and 7.94–29.33 allocated CPU-hours. CPU-only CatBoost scheduling could
reduce GPU reservation but requires a compatible launcher; it does not change
CatBoost's scientific profile. No parallel speedup is assumed.

For scheduling, reserve roughly **10 serial allocation-hours** for model runs,
batch startup and retry contingency. This is a planning envelope, not a promise.
Queue time and report-generation/bootstrap CPU time remain unmeasured and are
excluded; provision a separate reporting job and measure it. The full 220-run
report can require substantial CPU work despite small individual cohorts.

The seven transferred evidence directories occupy 820,961,654 bytes (about
0.765 GiB). TabICL artifacts are about 111 MB per run; 55 such runs project to
about 6.1 GB decimal. Other models' observed artifacts are substantially
smaller, but full-length histories and CatBoost models can grow. Provision
**20 GiB for new research outputs/reporting/retries**, in addition to existing
raw/processed data, environments, shared checkpoints and these retained pilot
artifacts. This is a storage allowance, not an exact final size prediction.

## Evidence and review gate

Evidence lives under `results/adaptation/x01_v1/development/stage2e_*` and is
ignored by Git. The companion JSON records exact per-file evidence hashes,
including each completed run's metadata, which in turn records its artifact
hashes. Keep both Newton and local copies. This tracked document provides a
readable summary even when ignored evidence is not available in a fresh clone.

## Approval record

Ryan Dozier approved the feasibility assessment and subsequently instructed:
"approve. go on and update those, and any checklist pieces needed".
Approval was recorded on 2026-11-22 at 18:45:00 UTC (13:45 EST) in this project
conversation. This timestamp records the approval entry, not an independently
verified message-send timestamp.

- [x] Retain all four models, three budgets, five seeds and 220 unique runs.
- [x] Accept the documented resource allowances and limitations; no concurrency assumed.
- [x] Retain X01 label/selection policy and E05 paired evaluation unchanged.
- [x] Preserve infrastructure failure/retry evidence and GPU preflight safeguards.
- [x] Revalidate protocol, matrix, manifest and evidence hashes at finalization.

X01 execution freeze and X02 Stage 2E acceptance are complete. X03 research
execution remains pending. Pilot outputs must not be promoted into research
results. Reporting/bootstrap runtime, queue delays and actual research-run
variability remain unmeasured; approval accepts the documented estimates rather
than treating them as measured guarantees. No scientific profile was changed.
