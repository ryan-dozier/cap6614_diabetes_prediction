# Research contribution (R06)

Prepared: November 10, 2026.
Status: **R06 complete: contribution and scope decisions approved; final planning rule recorded November 12, 2026.**
Prepared with AI assistance from the approved R04 questions and R05 reviews.
Ryan approved the central question, label accounting, and claim boundaries;
retained replacement comparisons subject to workload; and retained Candidate B
as a time-bounded follow-up. Candidate A retains all four models, with budget-grid
reduction prioritized if compute is constrained. Exact budgets and execution
protocols remain X01/E05 work; no experiments are implied by this completion.

## 1. Proposed contribution

We will evaluate whether two existing synthetic diabetes datasets improve
prediction on an observational NHANES cohort when only a limited number of
real-domain labels are available. Controlled comparisons will distinguish the
benefit of adding synthetic examples to a fixed real-label budget from the
benefit of replacing synthetic examples with real examples at a fixed total
training/context size. A bounded follow-up will evaluate whether domain-informed
feature removal improves transfer beyond matched removal controls.

The contribution is empirical: evidence about the utility and limitations of
particular synthetic sources under explicit information-access constraints.
It does not claim a new architecture, a new synthetic generator, or priority
for synthetic augmentation or feature removal.

The existing four-model comparisons establish context for this contribution.
TabPFN-3.5 and TabICLv2 represent pretrained tabular foundation models;
FT-Transformer is a task-trained transformer; CatBoost is the conventional
reference. Cross-model comparisons concern complete pipelines and cannot isolate
the causal effect of pretraining or attention.

## 2. Basis and relationship to the research questions

The [approved research questions](research_questions.md) establish RQ1/RQ2
as generator-shift and synthetic-to-real evaluation questions. Candidate A
operationalizes RQ3; Candidate B narrows the candidate intervention under RQ4.

The [R05 adaptation review](r05_adaptation_mitigation_review.md) and
[related-work draft](../paper/related_work_draft.md) distinguish:

- Training a generator on available real data and augmenting that same task.
- Pretraining a foundation model on synthetic tasks.
- Using an existing synthetic table alongside limited real-domain examples.

Our intervention concerns the third setting. Both Original Synthetic and
Playground S5E12 are synthetic tables acquired separately from NHANES. The
study asks whether those existing examples are useful without training a new
generator on NHANES. Context-retrieval and fine-tuning studies motivate examining
how a model is adapted, but this contribution does not require query-specific
retrieval or foundation-model weight updates.

The [model comparison](r05_model_comparison.md) documents differences in
training and validation access. The approved
[shift/evaluation discussion](../paper/shift_and_evaluation_discussion.md)
sets the interpretation limits. Detailed citations remain in those companion
documents and should accompany the corresponding claims in the final paper.

## 3. Candidate A: synthetic utility at a fixed real-label budget

### Question and working hypothesis

Does adding either synthetic source improve held-out NHANES prediction beyond
what is achievable with the same limited real supervision?

Consistent with RQ3/H3, we hypothesize that real-domain supervision can improve
on synthetic-only transfer and that synthetic examples may help at smaller real
budgets, with benefits depending on source, model, and budget. These are distinct
claims: improving on synthetic-only predictions does not prove that synthetic
examples improve on real-only training.

We do not require monotonic improvement or a positive result for every condition.
A reproducible lack of benefit or harmful effect remains a valid finding.

### Required comparison structure

Let B denote the total permitted number of labeled NHANES participants available
to a procedure, including labels used for fitting, selection, and calibration.
Let b denote the number actually used as training examples or labeled context.
These quantities can differ when some labels are reserved for validation.

| Contrast | Held constant | Interpretation |
| --- | --- | --- |
| Mixed versus real-only | B, target participant pool, allocation policy, and evaluation cohort; match the actual fitting/context participants where applicable | Incremental utility of adding synthetic examples to the available real supervision |
| Mixed versus synthetic-only at fixed total size N | N, synthetic source, model profile, and evaluation cohort | Effect of replacing synthetic examples with b real examples: b real plus N-b synthetic versus N synthetic |
| Original Synthetic mixed versus Playground mixed | Real participants, B, b, synthetic count, model profile, and evaluation cohort | Dependence of augmentation utility on the synthetic source |
| Real-only versus synthetic-only | Evaluation cohort and declared model profile | Reference comparison of different information sources; not an equal-real-label-budget contrast |

Synthetic-only uses zero NHANES labels and may serve as the same anchor across
positive-budget conditions. Real-only at zero labels is inapplicable. Tiny
positive budgets that cannot satisfy a model's class-support requirements must
also be handled explicitly, without silently increasing the budget.

For the fixed-N comparison, 0 <= b <= N. If a planned large budget exceeds N,
X01 must specify a separate size regime or mark that contrast inapplicable.
At b=N, the mixed condition contains no synthetic examples. The existing n2000
baseline does not automatically determine N for the adaptation study.

The fixed-total-size replacement contrast is retained subject to acceptable
workload. Reuse synthetic-only anchors and mixed predictions where the declared
sample sizes, profiles, and evaluation conditions match; do not create duplicate
runs merely because one prediction set supports multiple contrasts. Record any
resource-driven reduction before the affected test evaluation.

### Model and adaptation scope

**Accepted planning rule:** Candidate A will include CatBoost, FT-Transformer,
TabPFN-3.5, and TabICLv2 at a shared set of supported real-label budgets. If
measured compute requirements exceed available resources, reduce the budget
grid consistently across models before considering model exclusions. Finalize
reductions before evaluating the affected test conditions.

Preserve synthetic-only zero-label anchors and, where feasible, small,
intermediate, and larger positive budgets. Remove less informative intermediate
grid points before compromising the repeat analysis defined under E05. A coarser
grid yields a less detailed learning curve; record that limitation explicitly.
Handle unsupported conditions, such as samples without both classes, separately
from resource-driven scope reductions. Any eventual model exclusion would require
a new explicit scope decision rather than an automatic fallback.

This rule applies to Candidate A. Candidate B retains its separate bounded scope.

Before finalizing coverage under X01/E05, use a train/validation-only Newton
timing pilot to measure representative small and large budgets. Record fitting
or context-preparation time, inference throughput for the planned evaluation
sizes, peak memory, hardware, and feasible job concurrency under the allocation.
Estimate total workload, including repeats, retries, and analysis time. Keep
test participants out of this feasibility pilot and account for any target
labels used in development.

Freeze the final experiment matrix before evaluating affected test conditions.
Any coverage reduction must follow resource measurements and scientific
priorities, not observed test performance. Exact budgets and repeat counts remain
X01/E05 decisions; the six-budget/five-repeat example discussed in conversation
was illustrative, not an accepted run plan.

The proposed first mechanisms are:

- Frozen TabPFN/TabICLv2: change the labeled inference context.
- CatBoost/FT-Transformer: fit or refit on the declared real-only or mixed data.

These operations are model-specific. They are not identical optimization
procedures, even when their available examples match. Fine-tuning foundation-model
weights, generating new data, and calibration-only interventions are separate
scope additions, not prerequisites for Candidate A.

## 4. Candidate B: bounded domain-informed feature removal

### Question and working hypothesis

At one prespecified model/budget setting, does removing features that distinguish
a synthetic domain from NHANES improve prediction beyond the full-feature model
and generic removal controls?

The hypothesis is that a domain-based ranking may identify features whose removal
helps transfer beyond the effect of simply reducing the number of inputs.
A lower domain-classifier AUC alone does not support this hypothesis.

### Required controls

Use the primary strict_common_16 feature set and compare:

1. All features.
2. Removal by a domain-discrimination ranking.
3. Repeated random removal of the same number of features.
4. Source-predictive selection retaining the same number of features.

X01 must specify the ranking statistic, direction of source-predictive selection,
tie/correlation handling, removal count, and development data used. Refit the
task-trained pipeline or rebuild the foundation-model context after selection.
Keep participant identities, label budgets, and applicable settings aligned.

The planned initial scope is one model, one real-label budget, and one removal
count, with repeated random controls. Whether to apply it to both synthetic
sources is a pending scope decision. A full model-by-budget-by-removal-count
sweep is not proposed.

### Execution gate

Ryan accepted Candidate B as a bounded follow-up given sufficient time.
Proceed only after Candidate A's pipeline and analysis are stable and sufficient
time/compute remains for all controls and repeat analysis. Select the setting
from permitted development evidence or a declared scientific rationale, not the
best NHANES test result from Candidate A.

A resource-based deferral must be documented as a scope amendment to RQ4,
with its reason and prior result exposure recorded. Do not quietly drop Candidate B
because early results are negative or describe RQ4 as completed without testing it.

## 5. Information-access and evaluation requirements

- Preserve existing participant splits and strict_common_16 for the primary study.
- Use matched target draws within paired conditions; record IDs and sampling seeds.
- Count all target labels used for fitting, early stopping, tuning, or calibration.
  A full NHANES validation partition is not free supervision.
- Distinguish B from b in artifacts. X01 must define a practical selection policy
  for task-trained models and its comparability with frozen context models.
- Record synthetic sample counts, source validation access, external pretrained
  checkpoints, and compute separately from target-label budgets.
- Disclose unlabeled target-feature access, especially for Candidate B. Keep test
  participants out of preprocessing fits, context, feature ranking, and selection.
- Audit old feature rankings before reuse; prior existence does not establish that
  their construction obeyed the proposed data-access rules.
- Pair comparisons on the same held-out NHANES participants.

NHANES ROC-AUC and paired changes are the primary outcomes, consistent with RQ3/RQ4.
AP, Brier score, log loss, and fixed-threshold classification metrics provide
secondary evidence. Candidate B also reports synthetic-domain performance.
Calibration claims require the appropriate diagnostics, not Brier score alone.

E05 must specify primary contrast families, repeats, uncertainty, multiplicity,
and any practical-effect margins. These are needed before formal claims of
benefit; this draft does not select significance cutoffs or interval procedures.

## 6. Interpretation and success criteria

| Outcome | Supported interpretation |
| --- | --- |
| Mixed improves on matched real-only | Synthetic augmentation helps under the tested source/model/budget and declared uncertainty analysis |
| Mixed improves only on synthetic-only | Real-domain adaptation helps; incremental synthetic utility has not been established |
| Probability quality improves without ROC-AUC improvement | A narrower probability-quality benefit, not a ranking improvement |
| Benefit differs between synthetic sources | Source-dependent utility under the tested configurations |
| Domain-ranked removal improves beyond full features and matched controls | Evidence for the tested removal rule; no general proof that removed features are causal artifacts |
| Effects are negative or uncertain | Limits or inconclusive evidence for the tested intervention; no equivalence claim without a justified equivalence design |

The project succeeds by executing and reporting credible comparisons, including
negative findings. It does not require every advanced model to beat CatBoost.

Conclusions apply to the analyzed NHANES cohort and implemented profiles.
Harmonization, label construction, selection, and prevalence differences limit
causal explanations. Nationally representative or clinical-use claims require
additional justification beyond this design.

Baseline test results have already been inspected. Later HPC runs on the same
participants do not create independent confirmatory evidence. Freeze adaptation
choices before affected runs and date amendments with their rationale and
existing result exposure.

## 7. Decisions and deliverables after this review

| Destination | Decision/deliverable |
| --- | --- |
| R06 review | Central contribution and boundaries accepted; comparison/follow-up retention conditional on workload/time; all-four-model Candidate A scope and budget-first resource rule accepted |
| X01 | Exact budgets and participant counts, class support, nested or independent draws, synthetic counts, N regimes, shared budget coverage across all four models, and validation-label policy |
| X01 | Candidate B model, budget, source coverage, selection rules, removal count, and matched controls |
| E05 | Seeds/repeats, paired uncertainty, primary contrast families, multiplicity, practical-effect margins if used, and survey treatment |
| D07 | Provenance, measurement/label mappings, cohort selection, and applicable access conditions |
| HPC planning / X01 / E05 | Measure a train/validation-only Newton timing pilot; estimate full-matrix cost to select a feasible shared budget grid; freeze configurations, job inventory, and artifact requirements |
| F02 | Revise claims and related work to reflect completed experiments and limitations |

Approval of R06 does not mark these downstream tasks complete or authorize
test-driven configuration selection.

## 8. Focused review checklist

Ryan's decisions from the November 10 review and subsequent scope discussion,
reconciled November 12, 2026:

- [x] **Central contribution approved:** incremental synthetic utility beyond the
      same real supervision is the primary question.
- [x] **Comparison direction accepted conditionally:** retain augmentation and
      fixed-total-size replacement, provided workload is acceptable. Concrete
      sizes and any additional runs remain subject to X01/HPC planning.
- [x] **Bounded follow-up accepted conditionally:** retain Candidate B if time
      permits, with the documented execution gate and amendment rule.
- [x] **Model coverage approved:** retain all four models for Candidate A;
      use Newton timing evidence to reduce the shared budget grid first if needed.
      Exact budgets remain X01/E05 decisions.
- [x] **Label accounting approved:** all real labels used for fitting, selection,
      or calibration count; the concrete validation policy remains X01.
- [x] **Claim boundaries approved:** negative or uncertain results are acceptable.
      Course-project success depends on credible execution and reporting.

Exact budgets, repetition counts, and statistical procedures remain X01/E05
decisions. All six contribution-review items are resolved, including conditional
retention of the replacement contrast and Candidate B. R06 is complete at the
planning level. These checkmarks record decisions, not completed experiments or
measured HPC feasibility.

## Review record

| Date | Event | Status |
| --- | --- | --- |
| 2026-11-10 | Initial draft prepared from approved R04/R05 discussions, with AI assistance | Pending Ryan Dozier's review |
| 2026-11-10 | Ryan approved items 1, 5, and 6; conditionally retained items 2 and 3; requested further discussion of model coverage (item 4) | Partial approval recorded; model coverage pending |
| 2026-11-10 | Added provisional full-model coverage target and Newton timing gate following discussion | Planning proposal; no pilot or final matrix approved/executed |

| 2026-11-12 | Recorded Ryan\'s decision to retain all four Candidate A models and reduce the shared budget grid first if compute is constrained | Final R06 scope issue resolved; R06 complete; X01/E05 remain pending |

## November 12, 2026: Candidate A scope update

Ryan approved fixed 2,000-row mixed fitting/context sets, five repetitions with
root seeds 42–46, and all four models across the initial 250/500/1,000-label grid:
220 Candidate A runs, subject to Newton feasibility. See the
[X01 working protocol](x01_adaptation_protocol.md) for allocations and pending details.

Candidate B/RQ4 is deferred for now, not completed, and excluded from immediate
implementation and execution scope. Earlier descriptions of the bounded follow-up
are retained as historical design, not active commitments. This scope reduction
prioritizes the course-project workload; no Candidate B outcomes informed it.
Baseline test outcomes were previously inspected; no adaptation experiments
were run as part of this documentation update.
