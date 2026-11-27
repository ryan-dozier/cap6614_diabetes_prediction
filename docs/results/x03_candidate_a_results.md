# X03 Candidate A research results

Status: **Initial analysis draft, 2026-11-27 (America/New_York).**
Prepared with AI assistance from the transferred Newton report and execution
evidence. Ryan Dozier executed the jobs and supplied the completion records.
The initial diagnostic reviews of CatBoost seed 45, FT-Transformer, TabPFN
and TabICL are included below. This document does not
change the approved protocol or constitute the final paper interpretation.

## 1. Research question and scope

Does adding synthetic fitting/context examples improve NHANES performance
beyond the same real supervision alone?

The approved Candidate A experiment includes CatBoost, FT-Transformer,
TabPFN-3.5 and TabICLv2; available real-label budgets of 250, 500 and 1,000;
Original Synthetic (`source`) and Playground S5E12 (`kaggle`); and seeds
42–46. All models use `strict_common_16`.

Each budget includes the real fitting/context rows and reserved selection
labels: 200/50, 400/100 and 800/200 respectively. Mixed runs add 1,800, 1,600
or 1,200 synthetic rows to obtain 2,000 fitting/context rows. Foundation
models do not use the reserved selection labels. Real-only comparators use
the same real supervision as their paired mixtures.

The grid contains 60 real-only, 120 mixed and 40 synthetic-only runs.
Comparators are reused across contrasts. There are 24 primary mixed-versus-
real-only comparisons and 24 supporting mixed-versus-synthetic-only comparisons,
each evaluated with four metrics across five paired seeds.

Authoritative specifications:
- [Research contribution](../plans/r06_research_contribution.md)
- [X01 adaptation protocol](../plans/x01_adaptation_protocol.md)
- [E05 evaluation protocol](../plans/e05_evaluation_protocol.md)
- [Approved execution freeze](../plans/x01_execution_freeze.json)
- [Newton feasibility evidence](x02_stage2e_feasibility.md)
- [Research execution guide](../guides/x03_research_execution.md)

The approved freeze supersedes historical pending-feasibility wording in the
protocol documents. Candidate B remains deferred.

## 2. Execution and completion

Deployment revision: `3c1d1c2ee2cfb16719222868f155a6b3a1406e48`.

| Model | Newton job | Completed runs | Allocated elapsed | Allocation |
|---|---:|---:|---:|---|
| CatBoost | 1254823 | 55 | 00:06:24 | 4 CPUs, 64 GB RAM |
| FT-Transformer | 1254982 | 55 | 00:16:10 | 4 CPUs, 64 GB RAM, 1 H100 |
| TabPFN | 1255179 | 55 | 00:13:54 | 4 CPUs, 64 GB RAM, 1 H100 |
| TabICL | 1255398 | 55 | 00:11:22 | 4 CPUs, 64 GB RAM, 1 H100 |
| Total | | 220 | 00:47:50 | |

Every job reported Slurm `COMPLETED`, exit `0:0`, 55 newly completed
runs, zero skipped runs, zero remaining runs and zero invalid runs. The final
inventory independently classified all 220 runs as `verified_complete`.

Reporting job **1255599** completed with exit `0:0` in **00:35:12**.
Its report status is `completed`: 220/220 runs and 48/48 comparisons,
with no incomplete comparisons. Allocated elapsed times include startup,
validation and artifact handling; they are not pure model-training times.

Recorded fingerprints:
- Manifest assignments: `fea3d3c14d7d32fb2ee371981b2e2c907cc9b1083ac56f4dfd5fdedaaacfb59d`
- Matrix: `77d2429fbbbe840f5585ab4a66823d9423b9841026b9e0c504245a15ac937f24`
- Approved freeze: `cbec2c336ae966194be30d34924f0fb5e14f7dc61519f4e959b2a1b88c209dab`

## 3. Report consistency review

The local review independently reconstructed all 192 summary rows from the
220 unique per-run metric records. Mixed/comparator means, sample standard
deviations and paired mean effects matched the saved summaries. All 220
confusion matrices describe 646 participants, including 101 positive and
545 negative outcomes.

The saved bootstrap effects contain 192 metric/contrast cells with 2,000
replicates each (384,000 rows). Recomputed linear 2.5th/97.5th percentiles
matched all saved confidence bounds to floating-point precision. The saved
bootstrap-index file matches its recorded SHA-256.

These are report-consistency checks. The local review did not regenerate
predictions from saved models or independently recompute bootstrap metrics
from participant predictions. Those full research artifacts remain on Newton;
the reporting job performed its implemented prediction validation there.

## 4. Primary findings: incremental synthetic utility

Positive effects mean improvement: mixed minus comparator for ROC-AUC and
average precision, and comparator minus mixed for Brier score and log loss.

| Metric | Positive mean effects, out of 24 | Pointwise intervals entirely below zero | Intervals crossing or containing zero | Intervals entirely above zero |
|---|---:|---:|---:|---:|
| ROC-AUC | 0 | 21 | 3 | 0 |
| Average precision | 0 | 21 | 3 | 0 |
| Brier score | 1 | 18 | 6 | 0 |
| Log loss | 1 | 18 | 6 | 0 |

Neither synthetic source improves mean ROC-AUC or average precision over
matched real-only supervision in any tested model/budget cell. No primary
metric has a comparison whose pointwise interval is entirely above zero.

At budget 1,000, absolute mean ROC-AUC across five seeds is:

| Model | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| CatBoost | 0.8470 | 0.7733 | 0.8074 |
| FT-Transformer | 0.8564 | 0.7774 | 0.8205 |
| TabPFN | 0.8569 | 0.8493 | 0.8563 |
| TabICL | 0.8608 | 0.8050 | 0.8438 |

TabPFN comes closest to preserving real-only performance at the largest
budget. Its Playground mixture has mean ROC-AUC effect −0.000538 with interval
[−0.008943, 0.009138]. This does not establish equivalence.

TabICL has the highest observed real-only mean ROC-AUC at budget 1,000.
The reported contrasts do not formally test superiority between models.
Playground S5E12 produces smaller mean ROC-AUC losses than Original Synthetic
in every matched model/budget cell, but these intervals do not directly test
the difference between synthetic sources.

A suitable preliminary statement is:

> Across the tested models and real-label budgets, synthetic augmentation did
> not improve mean ROC-AUC or average precision beyond matched real-only
> supervision. Degradation depended on the synthetic source and model, with
> TabPFN showing the smallest losses at the largest budget.

This conclusion concerns the specified mixtures and model-selection procedures,
not synthetic data in general.

## 5. Supporting comparisons and seed variability

Relative to synthetic-only pipelines, mixed runs improve mean Brier score and
log loss in all 24 settings, with all corresponding pointwise intervals above
zero. For ROC-AUC, 16/24 mean effects are positive (10 intervals above zero,
3 below and 11 crossing zero). For average precision, 14/24 means are positive
(7 intervals above zero, 2 below and 15 crossing zero).

Thus, incorporating real supervision helps probability scores relative to the
synthetic-only pipelines, while these synthetic additions do not demonstrate
incremental benefit beyond the matched real-only baseline. Synthetic-only
contrasts are pipeline comparisons: task-trained anchor selection uses source
validation rather than the budgeted NHANES selection set.

Across the 120 primary paired seed comparisons, ROC-AUC decreases in 115 and
average precision decreases in 116. All five seeds decrease in 20/24 settings
for each of those metrics. Brier score worsens in 109/120 and log loss in
110/120. These overlapping comparisons are descriptive counts, not independent
statistical trials.

## 6. CatBoost seed 45: early checkpoint selection

The only positive primary mean effects for Brier score and log loss occur in
CatBoost with Playground S5E12 at budget 1,000. The mean improvements are
0.001598 for Brier score and 0.003926 for log loss; both intervals cross zero.

Four paired seeds worsen on each metric. Seed 45 improves substantially and
drives the positive average. Its real-only test Brier score is 0.150567,
compared with approximately 0.099–0.106 for the other four seeds. Its real-only
log loss is 0.484047; its ROC-AUC remains 0.840540.

The transferred training logs show:

| Seed | Real-only selected trees | Best validation ROC-AUC |
|---|---:|---:|
| 42 | 94 | 0.8138 |
| 43 | 54 | 0.8570 |
| 44 | 52 | 0.8744 |
| 45 | 7 | 0.8185 |
| 46 | 80 | 0.8720 |

Seed 45 peaks at zero-based iteration 6. Training continues through the
100-iteration patience window and restores the seven-tree model. Its matched
Playground mixture selects 103 trees.

The adapter optimizes log loss during fitting but selects the saved checkpoint
using validation AUC. AUC evaluates ranking rather than the quality of
probability estimates. The unusually early selected model is therefore a
plausible contributor to the elevated probability losses, but the logs alone
do not establish causality. CatBoost's console label `test:` refers to the
provided validation set, not the held-out research test outcomes.

There is no evidence here of a failed run or protocol violation. **Retain seed
45 unchanged**; do not exclude or rerun it based on its research results.

Suggested wording:

> The small mean improvement in CatBoost probability scores at budget 1,000
> was driven by one repetition whose real-only model selected an unusually
> early checkpoint. The other four repetitions showed degradation.

This diagnosis used transferred training logs and report metrics. The saved
model and full run metadata have not been inspected locally.

## 7. FT-Transformer: consistent degradation across seeds

The detailed review covers all 55 FT-Transformer training logs and the saved
per-run metrics and comparison summaries. All six primary source/budget
settings have negative mean effects on ROC-AUC, average precision, Brier
score and log loss. All 24 corresponding pointwise intervals lie below zero.

Mean ROC-AUC across five seeds:

| Real-label budget | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| 250 | 0.8136 | 0.7555 | 0.7580 |
| 500 | 0.8479 | 0.7569 | 0.7893 |
| 1,000 | 0.8564 | 0.7774 | 0.8205 |

Other mean metrics at budget 1,000:

| Metric | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| Average precision (higher is better) | 0.5021 | 0.3464 | 0.4113 |
| Brier score (lower is better) | 0.1053 | 0.1369 | 0.1204 |
| Log loss (lower is better) | 0.3344 | 0.4192 | 0.3770 |

### Training and checkpoint selection

All 55 runs stopped through patience after 21-101 epochs; none reached the
200-epoch ceiling. Seven mixed runs selected epoch 1 or 2, all at budgets
250 or 500. At budget 1,000, mixed runs selected epochs 10-42, yet both
synthetic sources still reduced ROC-AUC and average precision in every seed.

Early checkpoint selection may contribute to some low-budget behavior, but
it cannot explain the entire pattern. More epochs being available would not
by itself change the frozen patience-based stopping rule. There is no evidence
that the configured maximum epoch ceiling truncated these runs.

The adapter selects the highest raw validation ROC-AUC checkpoint, while
patience uses its separate minimum-improvement reference. Consequently,
executed epochs need not always equal the selected epoch plus the patience
length. The logs are consistent with ordinary completion; the full training
histories and model artifacts have not been independently inspected locally.

### Seed variability and source differences

ROC-AUC and average precision each worsen in 28 of the 30 paired
mixed-versus-real-only seed comparisons. The only improvements occur at
budget 250. At budgets 500 and 1,000, both metrics worsen for every paired
seed and both synthetic sources. These are overlapping descriptive
comparisons, not independent statistical trials.

Playground S5E12 has better mean ROC-AUC than Original Synthetic at every
budget, but does not uniformly have better probability scores. Its mean log
loss is worse at budgets 250 (0.6345 versus 0.6262) and 500 (0.5314 versus
0.5041). It should not be described as universally the better synthetic source.

Suggested wording:

> FT-Transformer showed consistently lower performance under synthetic
> augmentation across all tested source/budget settings and all four
> evaluation metrics. Several low-budget mixtures selected very early
> checkpoints, but degradation persisted at the largest budget across all
> seeds for ROC-AUC and average precision. The findings therefore cannot be
> attributed solely to an isolated repetition or the maximum epoch ceiling.

No rerun, exclusion or tuning change is indicated by this review. These
observations describe the frozen method's behavior; they do not establish
the causal mechanism behind degradation.

Evidence: the report's `summary.csv` and `run_metrics.csv`, all 55 logs in
`execution/jobs/ft_transformer_1254982/`, and checkpoint-selection logic in
`src/models/ft_transformer.py`. Paths are relative to the artifact and
repository locations documented below.

## 8. TabPFN: smaller augmentation penalties at larger budgets

The review covers the saved TabPFN comparison summaries, per-seed metrics,
all 55 per-run execution logs and the retained launch plan. All six primary
source/budget settings have negative mean effects on all four metrics.
No mean effect demonstrates improvement over matched real-only supervision.

Mean ROC-AUC across five seeds:

| Real-label budget | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| 250 | 0.8323 | 0.7702 | 0.8193 |
| 500 | 0.8509 | 0.8206 | 0.8389 |
| 1,000 | 0.8569 | 0.8493 | 0.8563 |

The ROC-AUC penalties narrow as the budget increases. This simultaneously
increases real context and reduces the synthetic share, so the trend cannot
be attributed to either change in isolation.

Other mean metrics at budget 1,000:

| Metric | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| Average precision (higher is better) | 0.5209 | 0.4983 | 0.5089 |
| Brier score (lower is better) | 0.09870 | 0.10116 | 0.09961 |
| Log loss (lower is better) | 0.31335 | 0.31962 | 0.31457 |

### Uncertainty and seed variability

At budget 1,000, all four Playground effect intervals cross zero. Its mean
ROC-AUC effect is -0.000538 with interval [-0.008943, 0.009138]. This supports
describing a small observed difference, not equivalence or noninferiority.
For Original Synthetic at this budget, the average-precision interval remains
entirely below zero: mean effect -0.022596, interval [-0.042888, -0.000812].

At budgets 250 and 500, both sources reduce ROC-AUC in every seed. At budget
1,000, Original Synthetic improves ROC-AUC in one of five seeds and Playground
improves it in two of five. Overall, ROC-AUC decreases in 27 of 30 paired
comparisons. These are overlapping descriptive comparisons, not independent
statistical trials.

No single anomalous repetition drives the overall negative ROC-AUC pattern.
For Playground at the largest budget, the seed-level differences are small
and vary in direction. Crossing-zero intervals elsewhere also do not erase
the negative point estimates: for example, every Playground seed at budget
250 has lower ROC-AUC despite the mean-effect interval crossing zero.

### Execution checks

All 55 per-run logs contain only their output paths, with no logged warnings
or tracebacks. The retained launch plan specifies the approved TabPFN-3.5
checkpoint, CUDA, float32 inference, four estimators, prediction batches of
1,024, memory-saving mode and `fit_preprocessors` fitting mode. No development
training cap or early-stopping setting is applied.

There is no task-training epoch or early-stopping explanation to investigate
as there is for CatBoost and FT-Transformer. These checks use retained plans
and logs; the full saved model metadata and artifacts have not been
independently inspected locally. Absence of logged warnings alone does not
establish the absence of every possible numerical issue.

Suggested wording:

> TabPFN showed decreasing augmentation penalties as the real-label budget
> increased. At the largest budget, the Playground mixture approached
> real-only performance across all four metrics, although no mean improvement
> was observed and the intervals do not establish equivalence.

No rerun, exclusion or tuning change is indicated by this review.

Evidence: the report's `summary.csv` and `run_metrics.csv`, and logs and
`plan.json` in `execution/jobs/tabpfn_1255179/`. Paths are relative to the
artifact locations documented below.

## 9. TabICL: consistent losses despite strong real-only performance

The review covers all 55 per-run execution logs, the retained launch plan,
and the saved comparison summaries and per-seed metrics. All six primary
source/budget settings have negative mean effects on all four metrics.
Of the 24 corresponding pointwise intervals, 23 lie entirely below zero.
The exception is the Playground Brier-score effect at budget 1,000, whose
interval narrowly crosses zero.

Mean ROC-AUC across five seeds:

| Real-label budget | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| 250 | 0.8330 | 0.7229 | 0.8056 |
| 500 | 0.8543 | 0.7586 | 0.8237 |
| 1,000 | 0.8608 | 0.8050 | 0.8438 |

At budget 1,000, the Playground mean ROC-AUC effect is -0.016968 with interval
[-0.029669, -0.003509]. Original Synthetic has a larger negative effect:
-0.055769 with interval [-0.076304, -0.034761]. Unlike TabPFN's Playground
comparison at this budget, TabICL's ROC-AUC interval remains below zero.
This is a descriptive comparison of within-model effects, not a formal test
of a difference between models.

Other mean metrics at budget 1,000:

| Metric | Real-only | Mixed: Original Synthetic | Mixed: Playground S5E12 |
|---|---:|---:|---:|
| Average precision (higher is better) | 0.5204 | 0.3980 | 0.4627 |
| Brier score (lower is better) | 0.09915 | 0.11163 | 0.10378 |
| Log loss (lower is better) | 0.31297 | 0.35399 | 0.32717 |

### Seed variability and source differences

Every one of the 30 paired mixed-versus-real-only seed comparisons worsens on
all four metrics. There is no isolated repetition driving the direction of
the result. These overlapping comparisons remain descriptive counts, not
independent statistical trials.

Playground S5E12 has better mean scores than Original Synthetic across all
tested budgets and all four metrics. The existing intervals do not directly
test Playground against Original Synthetic, so this observation should not
be presented as a formal source-superiority result.

The penalty is smaller at the largest budget than at the smallest, but the
trend is not strictly monotonic. Playground's mean ROC-AUC loss increases
from 0.027410 at budget 250 to 0.030586 at budget 500 before decreasing to
0.016968 at budget 1,000. Increasing the budget changes both real context
size and synthetic share, as described in the interpretation boundaries.

### Execution checks

All 55 per-run logs contain only output paths, with no logged warnings or
tracebacks. The retained plan specifies the approved TabICLv2 checkpoint,
CUDA, eight estimators, normalization method `none`, AMP disabled, CPU
offloading, KV cache disabled and prediction batches of 128. No development
training cap or early-stopping setting is applied.

There is no task-training epoch or checkpoint-selection explanation to
investigate. The logs and planned settings show no apparent operational issue,
but absence of warnings alone does not rule out every numerical issue.
The full saved model metadata and artifacts remain on Newton and have not
been independently inspected locally.

Suggested wording:

> TabICL achieved strong real-only performance, but synthetic augmentation
> reduced every evaluated metric in every paired repetition. Playground S5E12
> produced smaller losses than Original Synthetic, although degradation
> remained evident at the largest real-label budget.

No rerun, exclusion or tuning change is indicated by this review.

Evidence: the report's `summary.csv` and `run_metrics.csv`, and logs and
`plan.json` in `execution/jobs/tabicl_1255398/`. Paths are relative to the
artifact locations documented below.

## 10. Interpretation boundaries

- Intervals are pointwise 95% participant-bootstrap percentile intervals,
  conditional on the five fitted models and observed cohort/class composition.
  They do not capture full training-sample uncertainty or simultaneous coverage.
- Multiple comparisons are not adjusted; the counts above are descriptive.
- NHANES survey weights, strata and clusters are not incorporated. No
  national-population inference is supported.
- An interval crossing zero does not establish equivalence or noninferiority.
- Run standard deviations combine sampling/model variability and are not
  confidence intervals.
- Increasing the real-label budget increases real fitting rows while reducing
  the synthetic share of the fixed 2,000-row mixture. Effects cannot be
  attributed to either change in isolation.
- Historical work already examined this canonical test cohort. These results
  are exploratory rather than a previously untouched confirmatory evaluation.
- Brier score and log loss assess probability quality; their differences alone
  do not isolate calibration as the causal mechanism.
- Source differences and label prevalence may help motivate explanations, but
  these comparisons do not identify the cause of degradation.
- The results apply to these two synthetic datasets, this harmonized feature
  set, cohort, budgets and frozen model profiles.

## 11. Artifact locations and reproducibility

Artifacts are locally generated and Git-ignored; their paths below are local
references and will not resolve for a fresh clone without copying the results.
The tracked protocols, freeze and execution/reporting code describe the workflow.

Newton repository:
`/home/ry482284/cap6616/cap6614_diabetes_prediction`
(physical prefix `/lustre/fs1/home/ry482284/cap6616/cap6614_diabetes_prediction`).

Local repository:
`/home/noob/dev/ucf/fall2026/cap6614/cap6614_project`.

Relative to either repository:
- Report: `results/adaptation/x01_v1/reports/full_grid_3c1d1c2/`
- Launch plans and final inventory: `results/adaptation/x01_v1/execution/`
- Per-job logs, commands, timings and status:
  `results/adaptation/x01_v1/execution/jobs/<model>_<job_id>/`
- Full canonical research artifacts (retained on Newton):
  `results/adaptation/x01_v1/research/runs/<run_id>/`

The report and execution directories were copied locally. Principal report
files are `summary.csv`, `run_metrics.csv`, `paired_differences.csv`,
`comparisons.csv`, `inventory.csv`, `bootstrap_effects.csv`,
`bootstrap_indices.npy`, `report_metadata.json`, `report_status.json`
and `report.md`. Report accounting is retained as
`execution/report_1255599_accounting.txt`; model accounting is in each job
directory.

The report metadata records `worktree_dirty: true`. Untracked Slurm logs
were present in the supplied deployment status; this flag alone does not
demonstrate a source-code modification. Retain the recorded file fingerprints
and job Git-status evidence when documenting the execution revision.

## 12. Remaining review

- [x] Confirm all 220 runs and 48 comparisons completed.
- [x] Reconcile saved means, standard deviations and paired effects.
- [x] Check saved percentile bounds and bootstrap-index checksum.
- [x] Review all four metrics and descriptive seed-level direction.
- [x] Inspect CatBoost seed-45 training/checkpoint-selection logs.
- [x] Review FT-Transformer diagnostics in detail.
- [x] Review TabPFN diagnostics in detail.
- [x] Review TabICL diagnostics in detail.
- [ ] Prepare absolute-performance and paired-effect figures across budgets.
- [ ] Review the final interpretation with the complete paper and related work.
- [ ] Reconcile project-level checklist status and execution/PR attribution.

No additional training, exclusions, tuning or protocol amendments are proposed
by this initial review.
