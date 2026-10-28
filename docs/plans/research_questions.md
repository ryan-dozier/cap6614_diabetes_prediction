# Research questions and hypotheses (R04)

Status: **Approved research design (R04); detailed experimental protocols remain separate.**
Prepared: October 27, 2026.
Owner: Ryan Dozier. Drafting: Ryan Dozier, with AI assistance.
Reviewed and approved by: Ryan Dozier on October 27, 2026.

This is the accepted starting scope. Minor revisions may follow as the research
develops. Date each amendment, explain its reason and affected comparisons, and
record whether relevant results had already been inspected. Record changes before
running the affected experiments; retain the exploratory/prospective distinctions.

## Purpose and scope

Study how synthetic generator differences affect diabetes prediction and how
modern tabular methods transfer to an observed real domain. The principal methods
are TabPFN v3.5, TabICLv2, and FT-Transformer; CatBoost is the conventional reference.
SAINT implementation is deferred. Adding a model is not itself a research contribution.

The domains are Original Synthetic (`source`), Playground S5E12 (`kaggle`), and
NHANES 2015–2016 (`nhanes`). Both source and kaggle are synthetic. The primary
feature space is `strict_common_16`, using the existing fixed splits and participant
assignments. Additional feature sets are separate sensitivity studies.

Local laptop runs establish implementation feasibility and preliminary findings.
Newton HPC is the planned platform for the final repeated experiments, adaptation,
and ablations. Final budgets and configurations must be explicitly recorded;
laptop VRAM is not the final experimental budget. Newton hardware availability
and resource requests have not been verified by this document.

## Evidence already observed

The initial seed-42 matched-n2000 predictions and test metrics have already been
inspected. The existing [FT results review](../results/ft_transformer_predictive_results.md)
and [TabICL results review](../results/tabicl_predictive_results.md) document them.

Consequently:

- RQ1/RQ2 and their proposed hypotheses below are informed by exploratory results.
  They must not be described as hypotheses registered before seeing those results.
- Repeating experiments on HPC with the same test participants measures additional
  training/sampling variability. It does not create a previously unseen test set
  or provide independent confirmation of a hypothesis selected from that test set.
- Adaptation and predictive mitigation have not yet been executed. Their specific
  interventions, budgets, comparisons, and analysis rules must be fixed before
  evaluating them, while acknowledging prior exposure to baseline test outcomes.
- A prospective plan improves transparency; it does not erase earlier test access.
  Record amendments and their reasons before subsequent runs. Do not regenerate
  the fixed split merely to make an independence claim.

## RQ1: Synthetic generator shift

**Question:** For a fixed evaluation population and observed training-row budget,
how does changing the synthetic training domain change predictive performance
across the four model profiles?

**Working hypothesis H1 (exploratory-informed):** Training-domain effects vary
by model and evaluation domain; strong within-domain performance does not imply
a consistently superior model across domains.

**Comparisons:**

- Train on source versus kaggle and evaluate on the same source test participants.
- Repeat that comparison on the same kaggle test participants.
- Compare each advanced method with CatBoost within the same training domain
  and evaluation population.

The primary contrast is
`AUC(model, kaggle training, fixed test) - AUC(model, source training, fixed test)`.
Use actual saved probabilities and paired test identities. Do not compare AUC
on different test populations as if only the training domain had changed.
Changing the training domain does not match latent data-generation mechanisms,
so these contrasts do not establish a causal generator effect.

**Endpoints:** ROC-AUC is primary; average precision, Brier score, and fixed-threshold
classification metrics are secondary. Report signed effects and uncertainty
under E05, without requiring every advanced method to beat CatBoost.

**Evidence supporting H1:** Model/domain contrasts show reproducible heterogeneous
effects under the declared analysis. Similar contrasts or uniformly ordered
results weaken the heterogeneity hypothesis. Failure to detect a difference
does not establish equivalence; any equivalence margin requires prior justification.

## RQ2: Synthetic-to-real ranking and probability quality

**Question:** On the fixed NHANES test population, do models trained on synthetic
data preserve useful ranking while producing biased or unreliable probabilities?

**Working hypothesis H2 (exploratory-informed):** Synthetic-trained models can
retain discrimination on NHANES while systematically overestimating diabetes
probabilities, consistent with the observed prevalence and measurement differences.

**Comparisons:** Evaluate each synthetic-trained profile on the same NHANES
participants. Contrast ranking metrics and probability diagnostics within that
population. Synthetic-test results provide context, not an isolated estimate
of a domain-transfer effect.

**Endpoints:**

- Ranking: primary ROC-AUC; secondary average precision.
- Probability quality: Brier score and log loss.
- Calibration diagnostics: mean predicted probability minus observed prevalence,
  reliability curves with bin support, and the existing ECE definition.
- Classification: F1, precision, and recall at the fixed threshold 0.5.

**Evidence supporting H2:** Positive prediction bias and systematic departures
from reliability accompany retained discrimination under the declared uncertainty
analysis. Near-zero bias or reliable probabilities would weaken the proposed
overprediction mechanism; lack of discrimination would weaken the retained-ranking
part of the hypothesis. Aggregate bias alone cannot establish calibration.

Brier is overall squared probability error, not a pure calibration measure.
Do not tune thresholds or fit a calibrator on test labels. A constant predictor
using observed test prevalence is only a retrospective descriptive reference;
an operational baseline must estimate its probability from permitted training data.

NHANES label definitions, feature harmonization, complete-case selection, and
class prevalence differ from the synthetic domains. These results alone cannot
attribute all failures to synthetic-generation artifacts or establish national
survey-population performance.

## RQ3: Few-shot NHANES adaptation

**Question:** At a fixed real-label budget, how much does target-domain adaptation
improve on synthetic-only predictions, and when does synthetic information help
relative to training on real data alone?

**Proposed hypothesis H3 (untested intervention):** Access to labeled NHANES
training participants improves average real-domain performance over the synthetic-only
baseline; synthetic information can be useful at small real-label budgets, with
its advantage potentially shrinking as more real labels become available.

This is an aggregate hypothesis, not a promise of a monotone improvement for
every budget, model, or random sample.

**Required comparison families:**

1. Synthetic-only, zero-real-label baseline for each training domain and model.
2. Real-only model at each supported positive real-label budget.
3. Synthetic-plus-real adaptation with the same total real-label budget.
4. Calibration-only intervention, if included, reported separately from model/context
   adaptation.

Real-only supervised training is undefined at zero labeled participants; mark it
not applicable rather than inventing a zero-label fitted model.

**Endpoints:** NHANES ROC-AUC and its paired changes are primary. Report Brier,
average precision, log loss, and classification metrics as secondary outcomes.
An adaptation approach may improve probability quality without improving ranking;
state which outcome is supported.

**Evidence supporting H3:** Prespecified positive-budget adaptation contrasts
improve on zero-shot baselines under the declared repeated-sampling analysis.
Compare adapted and real-only methods at identical label budgets. Negative or
inconsistent contrasts are valid findings and weaken the corresponding hypothesis.

**Depends on X01/X02/E05:** Exact participant counts for the proposed 0%, 1%, 5%,
10%, 25%, 50%, and 100% budgets; class coverage; nested sampling; model-appropriate
adaptation; synthetic/real context allocation; and uncertainty rules.
Any NHANES labels used for early stopping, hyperparameter selection, or calibration
count toward the real-label budget. A full NHANES validation split is not free
supervision. Access to unlabeled NHANES features must also be declared as a separate
target-data condition. Test participants never enter adaptation or context.

## RQ4: Artifact-aware predictive mitigation

**Question:** Does an explicitly defined artifact-aware intervention improve
NHANES prediction compared with using all primary features and with equally sized
non-artifact-aware interventions?

**Proposed hypothesis H4 (candidate; requires R05/R06):** A targeted intervention
can improve transfer by reducing reliance on domain-specific signal beyond what
is explained by generic feature removal or reduced context size.

The concrete intervention is not selected here. Feature filtering is a candidate,
not an established novel method. Domain distinguishability is not evidence by
itself that a feature is spurious or should be removed.

**Required comparisons:** Unmodified profile; selected intervention; matched
random-feature-removal or random-context-selection control, as appropriate;
and component ablations under the same relevant budgets. Use features from
`strict_common_16` for the primary comparison.

**Endpoints:** Paired NHANES ROC-AUC change is primary; Brier, average precision,
and synthetic-domain performance are secondary. Report any loss in predictive
information as well as improvements. A reduction in domain-classification AUC
alone does not support predictive mitigation.

**Evidence supporting H4:** A prespecified intervention improves the primary
endpoint relative to both the unchanged baseline and an appropriate matched
control under E05. Improvement on only a secondary endpoint supports a narrower
claim. No improvement or a loss relative to controls is a legitimate negative result.

Rankings and intervention settings must use permitted training/validation evidence.
Audit the provenance of earlier rankings before reusing them. Do not choose
features, removal counts, context policies, or the winning method from test scores.
Target-domain feature or label access must be accounted for as in RQ3.

## Common evaluation commitments

- Preserve the fixed participant splits, identifiers-as-provenance policy, and
  training-only learned preprocessing.
- Match training samples and order within each declared comparison group.
  Keep the existing n2000 group separate from any larger HPC budget.
- Record synthetic-training labels, target-training labels, validation labels,
  target-feature access, external pretraining, and compute separately.
- Matching observed training rows does not isolate the causal effect of pretraining;
  architectures, optimization, ensembles, and selection policies also differ.
- Use 0.5 as the existing classification threshold unless a separate threshold
  intervention and its selection data are prospectively defined.
- E05 must distinguish model randomness, training-subset randomness, and finite
  test-sample uncertainty. Use paired evaluation populations and account for
  multiplicity before making formal claims across many models/budgets.
- Prespecify any practical-effect or noninferiority margins in E05 with rationale.
  Do not choose them from the effects observed after the analysis.
- A completed research question does not require a positive hypothesis result.
  Complete it by executing and reporting the declared comparisons, uncertainty,
  limitations, and negative findings.
- Capture code revision, effective configuration, versions, device, selected samples,
  completion status, and artifacts for the Newton runs. Larger hardware does not
  by itself establish fairness or reproducibility.

## Decisions still needed before additional research experiments

| Task | Required decision |
|---|---|
| R05 | Review and cite directly relevant model, adaptation, shift, and calibration literature |
| R06 | Select and justify one potentially novel contribution and its controls |
| X01/X02 | Freeze exact real-label budgets, validation access, adaptation mechanisms, and baselines |
| E05 | Freeze model/sampling seeds, repetition counts, uncertainty methods, primary contrast families, and any effect margins |
| HPC execution planning | Verify Newton resources and freeze configurations, job commands, and experiment inventory |

This document records the approved research design. Detailed execution
protocols, related-work justification, and the final method choice remain separate tasks.

## Approval and amendment record

| Date | Decision | Basis |
|---|---|---|
| 2026-10-27 | R04 approved by Ryan Dozier | User confirmed the reviewed questions and hypotheses are accepted, with minor future revisions possible. Detailed protocol and method-selection tasks remain open. |
