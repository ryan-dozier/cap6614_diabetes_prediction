# X01: NHANES adaptation protocol

Prepared: November 12, 2026.
Status: **Implementation specification approved by Ryan Dozier on November 12, 2026; E05 and Newton feasibility gates remain open.**
Prepared with AI assistance. No adaptation samples or experiments have been
created by this planning update.

This working plan implements Candidate A from the
[approved contribution](r06_research_contribution.md).
The matrix, seed policy, deterministic sampling specification, model-selection
settings, and evaluation/output specification are approved. Implementation is
not yet completed. E05 uncertainty choices and Newton feasibility remain open.

## 1. Agreed initial scope

Include CatBoost, FT-Transformer, TabPFN-3.5, and TabICLv2 at shared supported
real-label budgets. Compare real-only, mixed with Original Synthetic, and mixed
with Playground S5E12. Retain synthetic-only zero-target-label anchors for both
sources. Mixed conditions contain exactly 2,000 fitting/context rows, with
synthetic counts set to 2,000 minus the real fitting count. No additional mixture
ratios, synthetic quantities, or retrieval methods are planned.

Use strict_common_16 and preserve the existing participant split boundaries.
The initial positive budget grid is 250, 500, and 1,000 total available NHANES
labels. This replaces the earlier proposed percentage grid for the initial
experiment matrix; it is a limited-budget learning curve, not coverage through
100% of available NHANES training data.

If Newton timing evidence requires a smaller matrix, reduce the shared budget
grid before considering model exclusions, as agreed in R06. Document any change
before the affected test evaluations. Candidate B and RQ4 are explicitly deferred
for now, not completed. They are excluded from implementation planning and run
counts below; resuming them requires a separate scope decision.

## 2. Existing NHANES counts

Inspected from the saved split CSV files on November 12, 2026:

| Partition | Participants | Negative labels | Positive labels |
| --- | ---: | ---: | ---: |
| Training | 3,014 | 2,541 | 473 |
| Validation | 646 | 544 | 102 |
| Test | 646 | 545 | 101 |

These counts describe the existing cohort, not a nationally representative
prevalence estimate. No split is regenerated or pooled in this plan.

## 3. Available-label budget and validation policy

B counts unique NHANES labels made available to the procedure, including
fitting/context and validation. Allocate 80% to fitting/context participants
drawn only from the existing training partition, and 20% to validation
participants drawn only from the existing validation partition.

| Available labels B | Fitting/context b | Validation v | Approximate validation positives |
| --- | ---: | ---: | ---: |
| 250 | 200 | 50 | 8 |
| 500 | 400 | 100 | 16 |
| 1,000 | 800 | 200 | 32 |

Approximate positive counts illustrate feasibility, not frozen class quotas.
Section 9.1 specifies the approved deterministic allocation/rounding rule;
implementation must verify actual counts before creating manifests. Preserve approximate partition class
proportions; do not artificially balance the samples to 50/50.

Within each budget and repetition:
- All four models receive the same fitting/context participant identities.
- Real-only and both mixed conditions share those real fitting participants.
- CatBoost and FT-Transformer use the same reserved NHANES validation subset
  for early stopping/checkpoint selection.
- Frozen TabPFN and TabICLv2 use only fitting/context participants; reserved
  validation labels are unused by these model procedures.
- No full NHANES validation partition is supplied outside the declared budget.
- Test participants are used only for final evaluation, never fitting or selection.

Describe the comparison as matched real fitting participants and equal available
real-label budgets. Actual label utilization differs across model families.
Validation examples must not enter foundation-model context or be added to a
post-selection refit without a separately recorded protocol amendment.

Keep one validation policy for the main experiment. Cross-validation and a
fixed-training-schedule sensitivity are not part of this initial plan.
The smallest validation subset has few positives and may give noisy checkpoint
selection; repeats do not make each individual validation estimate precise.
Prefer removing an unsupported tiny budget over introducing a special policy.

Synthetic-only means no NHANES labels used for fitting or selection. Its
checkpoint-selection policy must therefore remain source-based or otherwise
target-label-free; it cannot use the mixed condition's NHANES validation labels.

## 4. Participant sampling

Use stratified sampling without replacement within each canonical partition.
This is retrospective experimental sampling using known diagnoses. The budget
measures labels exposed to the model procedure, not the cost of prospectively
discovering unknown labels.

Within each repetition, use nested fitting subsets and independently nested
validation subsets: the smaller budget's participant set is contained in the
larger budget's set for each role. Do not move participants between roles.

Across models and synthetic-source conditions, reuse the same real participant
assignments. Across repetitions, vary participant draws using recorded seeds;
overlap between repetitions is permitted and expected. Do not claim independent
evaluation cohorts because all runs share the fixed test participants.

Five repetitions use root seeds 42, 43, 44, 45, and 46. Use each repetition\'s
root seed for model randomness and derive separate deterministic sampling streams
for real fitting, validation, Original Synthetic, and Playground. The approved stream
mapping, class quotas, and shuffle/order rules are specified in Section 9.1. The implementation must preserve
nesting while meeting those quotas rather than independently sampling every
budget and assuming the subsets are nested.

Require both classes in fitting and validation subsets. Do not increase B
silently to repair an infeasible sample. Real-only at B=0 is inapplicable;
unsupported model/condition combinations must be recorded explicitly.

## 5. Synthetic allocation and approved run matrix

| Available real labels | Real fitting/context | Synthetic addition | Mixed total | Reserved real validation |
| --- | ---: | ---: | ---: | ---: |
| 250 | 200 | 1,800 | 2,000 | 50 |
| 500 | 400 | 1,600 | 2,000 | 100 |
| 1,000 | 800 | 1,200 | 2,000 | 200 |

Run each mixture separately with Original Synthetic and Playground. Each source
also has a 2,000-row synthetic-only anchor per model and repetition. Real-only
conditions use the same real fitting rows without synthetic additions.

Within each repetition, sample a stratified 2,000-row pool from each source's
canonical training partition. Share each pool across all four models and take
nested stratified 1,800/1,600/1,200-row subsets. Preserve source class proportions,
rather than forcing equal classes. Change the sampled pools across repetitions.
Section 9.1 specifies class allocation and deterministic ordering.

The mixed-versus-real-only contrast measures augmentation; mixed-versus-synthetic
at the same 2,000-row total measures replacement. Reuse the same predictions for
both contrasts. As real budgets increase, synthetic quantities decrease; the
learning curve does not hold synthetic quantity constant.

| Condition | Calculation | Unique runs |
| --- | --- | ---: |
| Real-only | 4 models x 3 budgets x 5 repetitions | 60 |
| Mixed | 4 models x 3 budgets x 2 sources x 5 repetitions | 120 |
| Synthetic-only anchors | 4 models x 2 sources x 5 repetitions | 40 |
| Total Candidate A | | **220** |

This excludes pilots, retries, and deferred Candidate B. Anchors are not repeated
for every positive budget. Existing runs may be reused only after checking exact
protocol/configuration and sample-identity equivalence; no reuse is assumed here.

Varying sample draws and model randomness together measures combined run-to-run
variability. There is no separate model-seed sweep and no claim to isolate those
variance sources. All repetitions share the fixed test participants; five repeats
do not create five independent evaluation cohorts. E05 still needs to specify
paired effects and test-sample uncertainty.

Newton timing will validate feasibility. If two positive budgets are retained
instead of three, the same four models and five repetitions require 160 runs.
Which budget to remove, if necessary, must be decided from resources and scientific
coverage before affected test evaluation, not from test performance.

## 6. Sampling manifest deliverable

Before fitting models, save a reproducible assignment manifest with:
- Protocol/schema version and code revision.
- Canonical dataset/split fingerprints and feature-set identifier.
- Repetition identifier, sampling seed, budget, and role (fitting or validation).
- Participant IDs, class counts, and ordered identity fingerprints.
- Sampling algorithm, rounding rule, and nesting policy.
- Declared available and used label counts for each model procedure.

Validation requirements:
- Fitting IDs are unique and belong only to the canonical training partition.
- Validation IDs are unique and belong only to the canonical validation partition.
- Neither role contains a test ID, and the two roles do not overlap.
- Counts match B=b+v, both classes are represented, and class quotas match policy.
- Smaller sets are subsets of larger sets within each repetition and role.
- All paired model/source conditions use identical real assignments and ordering.
- Recreating the manifest with the same inputs and seed reproduces assignments.

This manifest is a planned X02 implementation artifact. It does not exist yet.

## 7. Decisions required to complete X01

- [x] Record the initial three positive budgets plus zero-label synthetic anchors.
- [x] Agree on shared 80/20 fitting/validation allocation within B.
- [x] Preserve canonical partition boundaries and fixed test participants.
- [x] Agree on stratified, nested real-participant sampling across paired conditions.
- [x] Record manifest requirements and basic feasibility counts.
- [x] Fix mixed total size at 2,000 and synthetic additions at 1,800/1,600/1,200.
- [x] Define stratified source pools, nested subsets, and anchor reuse across budgets.
- [x] Approve five repetitions with root seeds 42–46 and no separate model-seed sweep.
- [x] Record the 220-run Candidate A matrix and defer Candidate B.
- [x] Freeze exact synthetic ordering and checks for historical anchor equivalence.
- [x] Freeze exact class allocations, seed derivation, and ordering/nesting algorithm.
- [x] Specify model-specific training caps, patience/checkpoint criteria, frozen
      foundation-model settings, preprocessing, and source-validation access.
- [x] Confirm evaluation-domain coverage and artifact/run naming.
- [ ] Finalize paired uncertainty, multiplicity, and survey treatment with E05;
      repetition count and joint sampling/model variability policy are agreed.
- [ ] Measure representative train/validation-only Newton workloads and finalize
      the feasible shared budget grid.
- [ ] Review and approve the complete protocol before affected test evaluations.

Next planning work: agree on the E05 analysis. X02 can implement the approved
sampler, run matrix, and train/validation-only Newton pilot path. The pilot
must precede final resource-based budget decisions and research test execution.

## 8. Amendment and review record

| Date | Decision | Rationale and prior evidence |
| --- | --- | --- |
| 2026-11-12 | Record Ryan's accepted initial 250/500/1,000-label grid, canonical 80/20 split sourcing, and stratified nested sampling direction | Keep the initial matrix manageable while retaining all four models. Saved NHANES counts support the proposed allocations; exact quotas and Newton feasibility are pending. |
| 2026-11-12 | Replace the earlier percentage-grid proposal for initial Candidate A planning | Resource/simplicity decision informed by cohort counts and previously inspected baseline runtimes/results. No adaptation experiments were executed as part of this update. |

| 2026-11-12 | Ryan reviewed the initial draft and approved fixed 2,000-row mixtures and five repetitions, seeds 42–46 | Candidate A totals 220 planned runs; retain all models and reduce shared budgets first if necessary. |
| 2026-11-12 | Defer Candidate B/RQ4 for now | Keep the course-project workload focused on Candidate A; no mitigation runs or completed RQ4 claim. |

| 2026-11-12 | Ryan explicitly approved Section 9 | Deterministic sampling, model/source-validation settings, NHANES-only evaluation, and output specification accepted. E05 and Newton feasibility remain open. |

Baseline test outcomes were already inspected during earlier project work.
This prospective adaptation plan does not erase that exposure or make future
runs on the same test participants independently confirmatory.

## 9. Approved implementation specification

Prepared November 12, 2026 after inspecting the existing sampler and four
reference configurations. Ryan Dozier reviewed and approved this section on
November 12, 2026. Approval covers the sampler, model/validation profiles, and
evaluation/output policy; it does not imply completed implementation or pilot runs.

### 9.1 Deterministic nested sampling

Use a versioned algorithm named x01_nested_v1, separate from the existing
select_training function. That function uses independent random choices and
does not guarantee nesting across budgets.

For each canonical input partition:
1. Validate unique participant identities and binary labels; sort by identity.
2. Compute the positive quota for requested size n using half-up proportional
   rounding: floor(n * positive_population / population_size + 0.5).
3. Constrain the quota to available class counts and require both classes.
   Reject unsupported sizes rather than silently changing the requested size.
4. Independently permute each class's identity-sorted rows once per repetition
   and partition. Take class-wise prefixes of the required lengths for every
   requested size. Verify that both class quotas are nondecreasing with size.
5. Sort the selected rows by identity before passing them to a model.

Proposed random-stream definition: NumPy default_rng initialized from
SeedSequence([root_seed, stream_id, class_label]), with fixed stream IDs:
0 = NHANES train; 1 = NHANES validation; 2 = Original Synthetic train;
3 = Playground train. Record this mapping and NumPy version in the manifest.
Do not use Python's process-dependent hash function or model-dependent streams.

The existing NHANES counts yield these exact proposed quotas:

| Budget | Fit negative | Fit positive | Validation negative | Validation positive |
| --- | ---: | ---: | ---: | ---: |
| 250 | 169 | 31 | 42 | 8 |
| 500 | 337 | 63 | 84 | 16 |
| 1,000 | 674 | 126 | 168 | 32 |

For each synthetic source, compute quotas from its full canonical training
partition for sizes 1,200, 1,600, 1,800, and 2,000. Use the same class-wise
permutations for all sizes, making the smaller subsets contained in the anchor
pool. Domain membership plus ID defines identity; equal numeric IDs from
different datasets are not duplicate participants.

For mixed inputs, use stable domain order (synthetic first, NHANES second),
then ascending identity within each domain. Use that exact ordering across all
models; FT-Transformer may subsequently shuffle batches under its recorded model
seed. IDs and domain membership are provenance, never additional model features.

The historical n2000 sampler differs from this algorithm. Do not assume its
seed-42 results are reusable anchors. Default to new anchors; reuse only after
sample/order and full protocol equivalence are demonstrated.

### 9.2 Reuse existing model profiles

Proposed policy: preserve the existing architectures, ensemble counts,
preprocessing, and training hyperparameters. Change the input assignments,
validation population, output location, and repetition seed for adaptation.

| Model | Proposed frozen profile |
| --- | --- |
| CatBoost | CPU, four threads; depth 7; learning rate 0.05; at most 1,000 iterations; Logloss objective; AUC checkpoint selection; early-stopping wait 100 |
| FT-Transformer | Three blocks, width 192, eight attention heads; existing ReGLU/dropout profile; AdamW learning rate 0.0001, weight decay 0.00001; training batch 256; at most 200 epochs; full permitted validation each epoch, batch 1,024; AUC checkpoint selection, patience 20, min_delta 0.001 |
| TabPFN | Existing base v3.5 checkpoint/hash; four estimators; float32; fit_preprocessors; memory-saving enabled; query batch 1,024 |
| TabICLv2 | Existing pinned v2 checkpoint/hash; eight estimators; none normalization; FP32; no AMP/FA3/KV cache; native ensemble batch one; CPU offloading; four preprocessing jobs; query batch 128 |

FT inference query batch remains 256, matching its reference configuration.
Preserve FT's distinction between selecting the highest raw validation AUC
checkpoint and using min_delta for the patience counter. Report the actual
selected/executed iterations or epochs. Retain existing no-class-weighting
policies; no tuning grid or probability calibrator is added.

Reference configurations:
[CatBoost](../../configs/catboost_source_n2000.json),
[FT-Transformer](../../configs/ft_transformer_source_n2000.json),
[TabPFN](../../configs/tabpfn_source_n2000_float32_batch1024.json),
[TabICLv2](../../configs/tabicl_v2_source_n2000_none_fp32.json).

For real-only and mixed CatBoost/FT runs, use only the budgeted NHANES validation
subset for selection, not source validation as well. For synthetic-only anchors,
use the complete same-source validation partition, matching the established
baseline selection policy; disclose these additional synthetic labels.
Consequently, mixed-versus-anchor comparisons change validation domain/access
for task-trained models as well as fitting composition. Describe them as
pipeline adaptation contrasts, not isolated effects of replacing fitting rows.
Frozen foundation-model anchors use no validation labels.

Fit preprocessing only on the actual fitting/context rows for each condition.
Do not fit scaling/category mappings on the full manifest pool, validation, or
test data. No refit on fitting plus validation after selecting a checkpoint.

Move checkpoint paths to the actual Newton location while preserving verified
content hashes. Keep precision and batching fixed for the initial pilot.
If execution settings need changes, document and validate them before final
runs rather than silently mixing profiles.

### 9.3 Evaluation and output proposal

Evaluate Candidate A on the fixed 646-row NHANES test cohort only. Source and
Playground test evaluation remain part of existing baseline work, not repeated
across the adaptation matrix. This directly targets RQ3 and avoids the large
synthetic-query workload. Candidate B is deferred.

Save the existing metric suite: ROC-AUC (primary), AP, Brier, log loss,
classification metrics at 0.5, and calibration summaries using the existing
10-bin definition. E05 must finalize interpretation and uncertainty; do not
select a winning condition or threshold from these test metrics.

Proposed artifact root: results/adaptation/x01_v1/.
Use a separate manifest folder and per-run folder names encoding model,
condition (real_only, synthetic_only, mixed), source (none, source, kaggle),
available target-label budget, synthetic count, and root seed. Examples:

- ft_transformer_mixed_source_B500_S1600_seed42
- tabicl_real_only_none_B250_S0_seed42
- catboost_synthetic_only_kaggle_B0_S2000_seed42

Record actual used label counts, fitting/context sizes, canonical split hashes,
ordered participant identities, checkpoint/package versions, device, code
revision, configuration, selected checkpoint, per-operation timing, completion
status, and predictions keyed by NHANES participant ID. Do not overwrite existing
runs; use separate execution roots for pilots and any revised protocol.

Source label access must be recorded separately from NHANES label access.
Retain adapter-supported persistence and identify any external checkpoint
dependencies. Count the 220 unique runs from the manifest before execution.

### 9.4 Review and remaining gates

- [x] Accept the proposed prefix sampler, quotas, streams, and ordering.
- [x] Accept reused model profiles and the disclosed source-validation policy
      for synthetic-only task-trained anchors.
- [x] Accept NHANES-only Candidate A evaluation and the output convention.
- [ ] Complete E05's paired uncertainty/reporting specification.
- [ ] Validate feasibility on Newton and freeze the final shared budget grid.

X02 can implement the approved protocol and a train/validation-only pilot path.
Final research test evaluation must wait for protocol and E05 approval.
A pilot requiring new sampling/runner support is not a reason to claim that the
full experiment protocol or research execution has already been completed.
