# FT-Transformer matched-budget predictive results

## Status, attribution, and scope

**M10 complete:** both frozen strict_common_16, n2000, seed-42 predictive
runs completed on 2026-10-27; their saved artifacts passed post-run validation.
Execution: **Ryan Dozier**. Artifact validation and this documentation were
prepared with AI assistance. This automated audit does not establish a new
human review, PR, commit, or merge.

This is a descriptive, single-seed comparison, not evidence of statistical
superiority, convergence, or a causal generator/pretraining effect. No training,
inference, tuning, split regeneration, dependency installation, or saved-artifact
modification was performed during the audit. No reports were overwritten.
**E04b update (2026-10-27): four-model automated reporting is implemented and validated; merge pending.**

Implementation acceptance is [M09 / Stage 4B](../plans/ft_transformer_integration_plan.md#21-stage-4b-final-review-and-acceptance).
[Stage 3](ft_transformer_stage3_results.md) is historical train/validation-only
feasibility evidence, not the predictive evaluation or its runtime measurements.
The [TabICL results review](tabicl_predictive_results.md) documents the preceding
comparison group.

## Explicit run selection and artifacts

Here `source` means Original Synthetic and `kaggle` means Playground S5E12.
Both are synthetic; `nhanes` is real NHANES 2015–2016 survey data.

| Training domain | Frozen configuration | Completed FT run |
|---|---|---|
| Original Synthetic | [source config](../../configs/ft_transformer_source_n2000.json) | [ft_transformer_source_strict_common_16_n2000_float32_seed42](../../results/predictive/ft_transformer_source_strict_common_16_n2000_float32_seed42) |
| Playground S5E12 | [Playground config](../../configs/ft_transformer_kaggle_n2000.json) | [ft_transformer_kaggle_strict_common_16_n2000_float32_seed42](../../results/predictive/ft_transformer_kaggle_strict_common_16_n2000_float32_seed42) |

Each directory contains exactly the expected ten nonempty files:
`config.json`, `metadata.json`, `metrics.json`, `metrics.csv`,
`calibration_bins.csv`, `training_sample.csv.gz`, `model.ft_transformer.pt`,
and `predictions_{source,kaggle,nhanes}_test.csv.gz`.
For direct provenance references, see the saved
[source metadata](../../results/predictive/ft_transformer_source_strict_common_16_n2000_float32_seed42/metadata.json),
[Playground metadata](../../results/predictive/ft_transformer_kaggle_strict_common_16_n2000_float32_seed42/metadata.json),
[source metrics](../../results/predictive/ft_transformer_source_strict_common_16_n2000_float32_seed42/metrics.json),
and [Playground metrics](../../results/predictive/ft_transformer_kaggle_strict_common_16_n2000_float32_seed42/metrics.json).
These result directories are **ignored local-only evidence**, not files supplied
by a Git clone; this results document and the configurations are tracked.

Baselines were explicitly selected from the existing
[six-run manifest](../../configs/reports/predictive_n2000_seed42_with_tabicl.json),
not by modification time or newest-directory matching:

| Training | CatBoost | TabPFN | TabICLv2 |
|---|---|---|---|
| source | [catboost_source_strict_common_16_n2000_seed42](../../results/predictive/catboost_source_strict_common_16_n2000_seed42) | [tabpfn_v35_source_strict_common_16_n2000_float32_seed42](../../results/predictive/tabpfn_v35_source_strict_common_16_n2000_float32_seed42) | [tabicl_v2_source_strict_common_16_n2000_none_fp32_uncached_seed42](../../results/predictive/tabicl_v2_source_strict_common_16_n2000_none_fp32_uncached_seed42) |
| kaggle | [catboost_kaggle_strict_common_16_n2000_seed42](../../results/predictive/catboost_kaggle_strict_common_16_n2000_seed42) | [tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024](../../results/predictive/tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024) | [tabicl_v2_kaggle_strict_common_16_n2000_none_fp32_uncached_seed42](../../results/predictive/tabicl_v2_kaggle_strict_common_16_n2000_none_fp32_uncached_seed42) |

Full-training-split CatBoost runs are excluded.

## Validation checklist and evidence boundaries

- [x] **Completion and configuration:** both metadata statuses are `completed`;
  saved configuration equals metadata configuration and the real loader's frozen
  research configuration, including canonical output path. Requested parameters
  resolve exactly to the accepted full research defaults; metadata agrees.
  Model/training/sampling seed 42, train budget 2,000, same-domain validation,
  all three test domains, float32, explicit `cuda:0`, prediction batch 256,
  threshold 0.5, and ten calibration bins are unchanged.
- [x] **Architecture/optimization:** three blocks, width 192, eight heads,
  ReGLU width 256, dropout attention/FFN/residual 0.2/0.1/0.0, AdamW with upstream
  parameter groups, learning rate 1e-4, weight decay 1e-5, training batch 256,
  validation batch 1024, max_epochs 200, patience 20, min_delta 0.001.
  Resolved settings establish the requested policy; code inspection establishes
  how parameter groups and training-only transformations are actually applied.
- [x] **Schema/target:** all eight runs have identical ordered strict_common_16
  schema, numeric/categorical definitions, and binary target
  `diagnosed_diabetes`. Feature order:
  `age`, `alcohol_consumption_per_week`, `bmi`, `cardiovascular_history`,
  `cholesterol_total`, `diastolic_bp`, `ethnicity`, `family_history_diabetes`,
  `gender`, `hdl_cholesterol`, `heart_rate`, `hypertension_history`,
  `physical_activity_minutes_per_week`, `sleep_hours_per_day`, `smoking_status`,
  `systolic_bp`. Categorical order: ethnicity, gender, smoking_status.
- [x] **Matched samples:** training files have exactly 2,000 unique identities
  and contiguous sample_order 0–1999. Ordered identities, both SHA-256 identity
  fingerprints, sampling algorithm/seed/class counts, full-training provenance,
  prepared-file fingerprints, and schema agree with all three baselines per
  training domain. Source training counts are 800 negative / 1,200 positive;
  Playground counts are 753 / 1,247.
- [x] **Full validation and disjointness:** source validation has 15,000 rows
  (6,000 / 9,000); Playground has 105,000 (39,554 / 65,446).
  Predictive validation summaries and fingerprints match saved Stage 3D identity
  files exactly. Those files were independently counted/hashed and are disjoint
  from the matching predictive training samples. Same-domain test identities
  are disjoint from both. The runner also explicitly checks full train/val/test
  disjointness using domain-qualified identities.
- [x] **Evaluation alignment:** all six FT prediction files have unique,
  nonmissing identities and row indices, correct domain/test tags, binary labels,
  finite probabilities in [0,1], and y_pred equal to probability >= 0.5.
  Eighteen FT-to-baseline population checks passed: two training domains times
  three evaluation domains times three baselines. Identity sets and per-identity
  labels agree, with no duplicates or missing rows. Alignment used row_index
  within source/kaggle and seqn within NHANES, never positional assumptions.
  The existing baseline comparison validator also passed its cross-run checks.
- [x] **Metrics:** recomputed all six evaluations with the unchanged
  [evaluate_binary](../../src/experiments/metrics.py) definitions. Full-precision
  JSON AUC/AP/F1/Brier and additional prevalence, precision, recall, log loss,
  ECE, threshold, confusion counts, warnings, and calibration bins agree.
  Maximum absolute scalar JSON metric discrepancy: **0.0** for each run.
  Scalar metrics CSV and calibration-bin CSV values also agree within the
  existing reporting tolerance, relative 1e-8 / absolute 1e-8. This allows
  serialization/numerical roundoff, not six-decimal terminal rounding.
- [x] **Selection/persistence:** finite contiguous histories; independently
  reconstructed earliest raw-AUC maximum and the separate min_delta patience
  counter. Both stop at the first exhausted patience counter. CPU-only
  `torch.load(..., map_location="cpu", weights_only=True)` inspection found
  format-v1 metadata, matching schema/preprocessing/settings/version/history,
  and finite float32 weights. Selected weights and preprocessing are exactly
  equal to the corresponding independently validated Stage 3D checkpoint,
  including the ordered raw-training-data/labels fingerprint. This is a
  comparison of independently saved runs, not checkpoint reuse by the runner.
- [x] **Artifact preservation:** SHA-256 checks before/after the audit confirmed
  68 selected artifact files unchanged (20 FT files and 48 baseline files
  covered by the existing validator). No experiment output was repaired.

Training identity fingerprints (unordered / ordered):

| Training | SHA-256 | Ordered SHA-256 |
|---|---|---|
| source | `239741018e0fa7b477a8985fdf7d214c05cd43b708b28ab061946e709b4eb9a0` | `7d81147cce93cd5c56b2e66484f9f8902ae4ef1784bab270c32016e43f44d170` |
| kaggle | `d4eca71ee3d52a04818645f1eb01c33ea26eeea0f2bde38bfab8cbb8ebabd63e` | `612fcfb9dae65a36b591f44065baef3cf629b028435737db4a5480b09469fe3b` |

Selected model file SHA-256:

- source: `bbcfa30ffe9f9427635788b3aba3878171e864dd62fc026635ca4134b52efcaa`
- kaggle: `f1a5bd0fc450d84cbd04fc4ab3a78b892472f9afeec73a773fa59988bdc59e5c`

### What requires code inspection, and what is not recorded

The [runner](../../src/experiments/runner.py) passes only selected training and
full same-domain validation data to fit, saves the fitted adapter, then loads
test partitions for prediction/metrics. The [adapter](../../src/models/ft_transformer.py)
fits StandardScaler and sorted categorical vocabularies on training only;
validation uses those transforms without refitting. Modal fallback is
training-only with lexical tie breaking. It snapshots detached raw-best weights,
restores that snapshot before marking fit complete, and uses that fitted model
for saving and prediction. Thus the **selected checkpoint**, not the last epoch,
is the model evaluated by this execution path. No test data enters preprocessing,
selection, patience, or parameter choice in that path. Saved artifacts alone
cannot prove the historical execution path or exclude unrecorded external
human decisions; no inference was rerun to independently bind weights to each
saved probability.

Both runs record Python 3.11.17, Torch 2.14.1+cu126, rtdl-revisiting-models 0.0.2,
NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.17.1, Linux/WSL2,
`cuda:0`, and NVIDIA GeForce RTX 4060 Laptop GPU. Package/model versions and
prepared-data hashes are present, but **the runner does not record an execution
Git revision or dirty-worktree snapshot**. Audit inspection was on clean `main`
at `9c2cc1b`; that is not claimed to be the execution revision. Nor does this audit
prove that prepared source files remain identical today: it compared the saved
fingerprints and identity evidence, without regenerating splits or loading
dataset features. Training sample files contain identities, not labels;
class counts/data-label fingerprints, common prepared hashes, and code provide
the corresponding training-label evidence rather than a cross-model saved
training-label vector comparison.

## Evaluation populations

All models and both training domains share each evaluation population:

| Test domain | Rows | Positive | Negative | Positive prevalence |
|---|---:|---:|---:|---:|
| source | 15,000 | 8,999 | 6,001 | 0.599933 |
| kaggle | 105,000 | 65,446 | 39,554 | 0.623295 |
| nhanes | 646 | 101 | 545 | 0.156347 |

## Predictive results and matched-budget comparisons

Values below are rounded to six decimals **after** full-precision validation.
Higher ROC-AUC/AP/F1 is favorable; lower Brier is favorable. AP is average
precision, not trapezoidal PR area. F1 uses the fixed threshold 0.5.
TabPFN is the saved v3.5 float32 profile; TabICL is the saved v2
none-normalization/FP32/uncached profile, not vendor-default normalization.

### Original Synthetic training (source)

| Test | Model | ROC-AUC | AP | F1@0.5 | Brier |
|---|---|---:|---:|---:|---:|
| source | CatBoost | 0.649753 | 0.733946 | 0.739310 | 0.226477 |
| source | TabPFN | 0.649322 | 0.733937 | 0.719145 | 0.224178 |
| source | TabICL | 0.650641 | 0.734682 | 0.720414 | 0.224525 |
| source | FT-Transformer | 0.647146 | 0.731943 | 0.673593 | 0.226632 |
| kaggle | CatBoost | 0.665150 | 0.768570 | 0.765748 | 0.223829 |
| kaggle | TabPFN | 0.672194 | 0.773193 | 0.760486 | 0.215273 |
| kaggle | TabICL | 0.673041 | 0.774195 | 0.759511 | 0.214436 |
| kaggle | FT-Transformer | 0.669817 | 0.771530 | 0.713338 | 0.218200 |
| nhanes | CatBoost | 0.758361 | 0.342324 | 0.312693 | 0.322067 |
| nhanes | TabPFN | 0.787864 | 0.383181 | 0.351852 | 0.318468 |
| nhanes | TabICL | 0.805995 | 0.412307 | 0.343363 | 0.349823 |
| nhanes | FT-Transformer | 0.796748 | 0.384444 | 0.369048 | 0.302800 |

### Playground training (kaggle)

| Test | Model | ROC-AUC | AP | F1@0.5 | Brier |
|---|---|---:|---:|---:|---:|
| source | CatBoost | 0.639871 | 0.728238 | 0.696781 | 0.228685 |
| source | TabPFN | 0.648997 | 0.735163 | 0.683784 | 0.229245 |
| source | TabICL | 0.650522 | 0.737121 | 0.693207 | 0.230376 |
| source | FT-Transformer | 0.644701 | 0.728897 | 0.637549 | 0.234699 |
| kaggle | CatBoost | 0.679632 | 0.778949 | 0.761122 | 0.212627 |
| kaggle | TabPFN | 0.698164 | 0.790732 | 0.757768 | 0.208161 |
| kaggle | TabICL | 0.687861 | 0.784304 | 0.759702 | 0.210577 |
| kaggle | FT-Transformer | 0.681985 | 0.779535 | 0.711341 | 0.215492 |
| nhanes | CatBoost | 0.769062 | 0.374393 | 0.354478 | 0.342607 |
| nhanes | TabPFN | 0.780870 | 0.375148 | 0.362205 | 0.313163 |
| nhanes | TabICL | 0.781324 | 0.356710 | 0.349624 | 0.351982 |
| nhanes | FT-Transformer | 0.728749 | 0.322926 | 0.382872 | 0.279903 |

### Within-population differences: FT minus baseline

Each row compares the **same training domain and same test population**.
Positive AUC/AP/F1 differences favor FT; negative Brier differences favor FT.
Differences are computed before rounding, not from the displayed tables.

| Train | Test | Baseline | Delta AUC | Delta AP | Delta F1 | Delta Brier |
|---|---|---|---:|---:|---:|---:|
| source | source | CatBoost | -0.002607 | -0.002003 | -0.065717 | +0.000155 |
| source | source | TabPFN | -0.002176 | -0.001994 | -0.045552 | +0.002454 |
| source | source | TabICL | -0.003494 | -0.002739 | -0.046820 | +0.002107 |
| source | kaggle | CatBoost | +0.004668 | +0.002960 | -0.052410 | -0.005629 |
| source | kaggle | TabPFN | -0.002377 | -0.001663 | -0.047149 | +0.002928 |
| source | kaggle | TabICL | -0.003223 | -0.002665 | -0.046173 | +0.003765 |
| source | nhanes | CatBoost | +0.038387 | +0.042121 | +0.056354 | -0.019268 |
| source | nhanes | TabPFN | +0.008884 | +0.001264 | +0.017196 | -0.015668 |
| source | nhanes | TabICL | -0.009247 | -0.027863 | +0.025685 | -0.047024 |
| kaggle | source | CatBoost | +0.004830 | +0.000659 | -0.059232 | +0.006014 |
| kaggle | source | TabPFN | -0.004296 | -0.006266 | -0.046235 | +0.005454 |
| kaggle | source | TabICL | -0.005821 | -0.008224 | -0.055658 | +0.004322 |
| kaggle | kaggle | CatBoost | +0.002353 | +0.000586 | -0.049780 | +0.002865 |
| kaggle | kaggle | TabPFN | -0.016179 | -0.011196 | -0.046427 | +0.007331 |
| kaggle | kaggle | TabICL | -0.005876 | -0.004769 | -0.048361 | +0.004915 |
| kaggle | nhanes | CatBoost | -0.040312 | -0.051467 | +0.028394 | -0.062704 |
| kaggle | nhanes | TabPFN | -0.052121 | -0.052222 | +0.020667 | -0.033260 |
| kaggle | nhanes | TabICL | -0.052575 | -0.033784 | +0.033247 | -0.072078 |

### Between FT training domains: Playground-trained minus source-trained

The test population is held fixed within each row; training domain changes.

| Test | Delta AUC | Delta AP | Delta F1 | Delta Brier |
|---|---:|---:|---:|---:|
| source | -0.002445 | -0.003046 | -0.036044 | +0.008067 |
| kaggle | +0.012168 | +0.008005 | -0.001997 | -0.002708 |
| nhanes | -0.067999 | -0.061518 | +0.013824 | -0.022896 |

## Selection, runtime, and resource scope

| Training | Executed / selected epochs | Selected validation AUC | Stop | Last patience-reset epoch |
|---|---:|---:|---|---:|
| source | 23 / 3 | 0.6456892962962963 | patience | 3 |
| kaggle | 29 / 13 | 0.6816919093141097 | patience | 9 |

Both counters end at 20. Source reset epochs: 1, 2, 3. Playground reset epochs:
1, 2, 3, 5, 6, 9. Playground's patience reference is 0.6812525872644797;
epoch 13 improves the raw best but by less than 0.001 relative to that reference,
so it selects a checkpoint without resetting patience. The configured cap
remains 200, not 23/29 or 3/13. Patience stopping is not a convergence claim.

Actual predictive-run measurements, seconds:

| Training | Fit | Sum epoch training | Sum epoch validation | Fit overhead | Predict source | Predict kaggle | Predict nhanes | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 16.791094 | 9.128437 | 4.802546 | 2.860112 | 0.528355 | 4.251376 | 0.043525 | 31.978238 |
| kaggle | 53.595667 | 10.123639 | 41.220106 | 2.251921 | 0.521132 | 3.823625 | 0.037005 | 66.430342 |

Fit is the runner's synchronized fit call, including preprocessing, model setup,
all training/validation, checkpoint handling, and restoration. Epoch durations
are synchronized training and full-validation blocks; validation includes
validation inference/loss/full-population AUC. Fit overhead is fit minus those
two epoch-duration sums, not a separately timed operation. Prediction durations
cover synchronized adapter prediction calls, excluding data loading, metric
calculation, and output writing. Total is the saved perf_counter workflow
duration, including other runner work up to completion metadata construction;
it is not inferred from timestamps. Save/reload timings are not separately
recorded in these predictive runs; no audit reload inference was timed.
Do not substitute Stage 3 feasibility times.

Both final resource records report allocator peaks of 191,702,016 bytes allocated
(182.82 MiB) / 304,087,040 reserved (290 MiB), with free/total memory at query
7,103,053,824 / 8,585,216,000 bytes. Scope is **since the adapter reset inside fit
before model construction, continuing through subsequent save/prediction**;
these final snapshots cannot isolate fit-only or per-domain prediction peaks.
PyTorch allocator measurements are not total GPU/process memory; free memory
is a point-in-time device query, not memory dedicated to this model. No CPU peak
or competing-process history is recorded. Runtime differences are descriptive
local observations, not controlled cross-model compute comparisons.

## Interpretation, limitations, and next steps

**Ranking.** Source-trained FT ranks NHANES substantially better than
Playground-trained FT in this seed: AUC 0.796748 versus 0.728749 and AP 0.384444
versus 0.322926. It exceeds source-trained CatBoost/TabPFN on both NHANES ranking
metrics, but is below source-trained TabICL. Playground-trained FT is below all
three corresponding NHANES baselines in AUC/AP. On both synthetic test sets,
FT trails both pretrained profiles in AUC/AP for each training domain; its
comparisons to CatBoost vary by training/test domain.

**Thresholded classification.** Both FT runs have lower F1@0.5 than all three
matched baselines on the synthetic test sets, yet higher F1 on NHANES.
Playground-trained FT's NHANES F1 is higher than source-trained FT by 0.013824
despite worse AUC/AP. F1 at one threshold is not evidence of better ranking.

**Probability quality and prevalence.** Playground-trained FT has lower
NHANES Brier (0.279903 versus 0.302800), despite lower AUC/AP. Both FT runs have
lower NHANES Brier than their respective three baselines. This is lower mean
squared probability error, **not proof of better calibration alone**: Brier
reflects calibration and discrimination/resolution as well as outcome
uncertainty. NHANES prevalence is 15.63%, versus about 60–62% in the synthetic
tests; AP's no-skill reference and probability-error interpretation depend on
prevalence. Thus cross-domain AP/Brier magnitudes are not comparable as if class
balance were fixed. For descriptive scale only, a constant predictor equal to
the observed NHANES test prevalence has Brier p(1-p)=0.131902, below every model
here; this is a retrospective reference, not a fitted/evaluated deployable
baseline or permission to calibrate on the test set. Within-NHANES comparisons
hold population/prevalence fixed, so the ranking/Brier tradeoff is genuine in
these saved predictions, not evidence of statistical superiority.

**Unequal supervision/compute.** FT and CatBoost use full same-domain validation
labels for selection/stopping; TabPFN/TabICL adapters use training context only
and do not fit/select/calibrate on validation labels. FT's 15,000/105,000
validation rows therefore add unequal label access beyond the common 2,000
training rows. External pretraining, architecture, ensembles, devices, and query
batches also differ (CatBoost 256; TabPFN source 256/Playground 1024; TabICL 128;
FT 256). Equal training budgets do not match supervision or compute.

Only one model/sampling seed is represented; NHANES has just 101 positives.
No confidence intervals, repeat-seed uncertainty, survey-design-adjusted
population inference, adaptation, artifact ablations, or new calibrator are
established. Do not select hyperparameters or thresholds from these test results.

At M10 completion, the next task was to extend reporting model-profile validation,
manifest/run selection, tables, figures, labels, and tests for FT, preserving
legacy reports and fresh-output protection. That separate task is now implemented under E04b (see follow-up below). Predefine uncertainty/repeat-run and later adaptation/ablation
protocols separately. M10's completion is limited to these two frozen predictive
runs and their artifact validation, not those downstream research/reporting tasks.

## Audit execution and documentation checks

The read-only audit used
`/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python` with the unchanged
[reporting validation utilities](../../src/reporting/validation.py), real configuration
loader, identity fingerprint helpers, and metric definitions. It validated the
six baseline runs using `validate_comparison` with the explicit existing manifest,
then applied compatible lower-level checks to FT without extending reporting
schemas. CPU checkpoint inspection allocated no CUDA model.
Documentation-only changes require no model tests or full-suite rerun;
the 184-test Stage 4B result is historical, not a new test execution.
Local Markdown links/anchors, intended result/config paths, ignore status, and
`git diff --check` were checked for this documentation update.

## E04b reporting follow-up

The [eight-run manifest](../../configs/reports/predictive_n2000_seed42_with_ft_transformer.json) now generates the [four-model report](../../results/reports/predictive_n2000_seed42_with_ft_transformer/report.md). This separately implemented follow-up preserves M10 artifacts and earlier reports. Eight runs and 21 population checks passed; no model execution occurred. Human review and merge remain pending.
