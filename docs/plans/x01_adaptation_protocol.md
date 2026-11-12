# X01: NHANES adaptation protocol

Prepared: November 12, 2026.
Status: **In progress; Candidate A matrix and five-repetition plan approved by Ryan Dozier on November 12, 2026.**
Prepared with AI assistance. No adaptation samples or experiments have been
created by this planning update.

This working plan implements Candidate A from the
[approved contribution](r06_research_contribution.md).
The full protocol is not yet frozen: exact sampling implementation,
training-selection settings, evaluation/output details, and E05 uncertainty
choices remain open. Repetition count and root seeds are approved.

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
X01 must specify the deterministic allocation/rounding rule and verify actual
counts before creating manifests. Preserve approximate partition class
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
for real fitting, validation, Original Synthetic, and Playground. Exact stream
mapping, class quotas, and shuffle/order rules remain to be finalized. The implementation must preserve
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
Exact class allocations and deterministic ordering remain to be specified.

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
- [ ] Freeze exact synthetic ordering and checks for historical anchor equivalence.
- [ ] Freeze exact class allocations, seed derivation, and ordering/nesting algorithm.
- [ ] Specify model-specific training caps, patience/checkpoint criteria, frozen
      foundation-model settings, preprocessing, and source-validation access.
- [ ] Confirm evaluation-domain coverage and artifact/run naming.
- [ ] Finalize paired uncertainty, multiplicity, and survey treatment with E05;
      repetition count and joint sampling/model variability policy are agreed.
- [ ] Measure representative train/validation-only Newton workloads and finalize
      the feasible shared budget grid.
- [ ] Review and approve the complete protocol before affected test evaluations.

Next planning work: freeze deterministic sampling rules and the model-specific
training/validation settings, then agree on the E05 analysis and Newton pilot.

## 8. Amendment and review record

| Date | Decision | Rationale and prior evidence |
| --- | --- | --- |
| 2026-11-12 | Record Ryan's accepted initial 250/500/1,000-label grid, canonical 80/20 split sourcing, and stratified nested sampling direction | Keep the initial matrix manageable while retaining all four models. Saved NHANES counts support the proposed allocations; exact quotas and Newton feasibility are pending. |
| 2026-11-12 | Replace the earlier percentage-grid proposal for initial Candidate A planning | Resource/simplicity decision informed by cohort counts and previously inspected baseline runtimes/results. No adaptation experiments were executed as part of this update. |

| 2026-11-12 | Ryan reviewed the initial draft and approved fixed 2,000-row mixtures and five repetitions, seeds 42–46 | Candidate A totals 220 planned runs; retain all models and reduce shared budgets first if necessary. |
| 2026-11-12 | Defer Candidate B/RQ4 for now | Keep the course-project workload focused on Candidate A; no mitigation runs or completed RQ4 claim. |

Baseline test outcomes were already inspected during earlier project work.
This prospective adaptation plan does not erase that exposure or make future
runs on the same test participants independently confirmatory.
