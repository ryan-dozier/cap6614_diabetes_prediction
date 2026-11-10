# Distribution shift and evaluation interpretation

Draft prepared November 9, 2026; reviewed and approved by Ryan Dozier
for the current planning milestone. Prepared with AI assistance.
This companion to [the related-work draft](related_work_draft.md) supplies
material for the final paper's evaluation rationale and limitations.
Citation keys refer to [references.bib](references.bib). Revisit this wording
after the experimental protocol and results are finalized.

## Distribution shift and its assumptions

We study transfer between two synthetic diabetes datasets and an observational
NHANES cohort without assuming a single mechanism of distribution shift.
Let X denote the harmonized predictors and Y the diabetes outcome.
Covariate shift assumes that the feature distribution P(X) changes while
the conditional outcome distribution P(Y | X) remains unchanged.
This assumption underlies covariate-shift adaptation methods that adjust
the influence of source examples to better reflect the target population
\cite{bickel2009covariateshift}. Label shift instead assumes that outcome
prevalence P(Y) changes while the within-class feature distribution P(X | Y)
remains unchanged. Lipton et al. use this assumption to estimate target class
proportions from unlabeled target predictions and a source confusion matrix,
subject to identifiability conditions
\cite{lipton2018detectingcorrectinglabelshift}.

Neither assumption follows from observing different feature distributions
or diabetes prevalences. In this study, synthetic generation, population
selection, measurement differences, and label construction may all contribute
to differences between domains. We therefore describe the observed changes
as distribution shift rather than asserting pure covariate or label shift.
Domain classification provides a diagnostic of dataset distinguishability,
consistent with the shift-detection framework examined by Rabanser et al.
\cite{rabanser2019failingloudly}. It does not establish the mechanism of shift
or whether the differences harm diabetes prediction. Likewise, a feature that
distinguishes domains is not necessarily an erroneous artifact. Claims that
feature removal mitigates harmful shift require downstream comparisons with
the full feature set and appropriate removal controls.

## Discrimination and outcome prevalence

ROC-AUC and average precision (AP) summarize complementary aspects of
discrimination across decision thresholds. ROC curves compare true-positive
and false-positive rates, whereas precision-recall curves emphasize the
fraction of positive predictions that are correct and the fraction of actual
positives recovered. With an imbalanced outcome, a modest false-positive rate
can still correspond to many false positives relative to true positives.
Davis and Goadrich explain this distinction and show that optimizing ROC area
does not necessarily optimize precision-recall area \cite{DavisPrecisionRecall}.

Precision depends on outcome prevalence, including when class-conditional
prediction behavior is held fixed. Consequently, we emphasize comparisons
between models evaluated on the same held-out target participants and report
the cohort's positive prevalence alongside its metrics. A difference in AP
between a synthetic dataset and NHANES cannot by itself isolate deterioration
caused by transfer. ROC-AUC also does not eliminate the consequences of
different populations or class-conditional feature distributions.

Our implementation computes AP using scikit-learn's average_precision_score.
It summarizes precision across thresholds using increases in recall as
weights, rather than trapezoidal integration of a precision-recall curve
\cite{scikitLearnAveragePrecision}. We therefore use the label "average
precision" consistently when reporting this metric.

## Probability quality and calibration

Discrimination alone does not determine whether predicted probabilities
accurately represent outcome frequencies. A model can rank participants
well while systematically overestimating their probability of diabetes.
We therefore complement ROC-AUC and AP with Brier score and log loss.
These losses belong to the proper-scoring-rule framework: their expected
values are optimized by reporting the true outcome probabilities
\cite{gneiting2007properscoring}. Lower values indicate better probabilistic
prediction on the evaluated cohort.

Brier score and log loss assess overall probability quality rather than
calibration alone. We therefore do not interpret an improvement in either
metric as sufficient evidence of improved calibration. Specific calibration
claims should additionally examine agreement between predicted probabilities
and observed frequencies, using calibration summaries and plots interpreted
with attention to sample size and binning \cite{GuoConfidenceCalibration}.
Comparisons of probability quality, like comparisons of discrimination,
should prioritize the same evaluation participants.

## Scope of uncertainty claims

Evaluation on the same participants permits paired comparisons between
models; correlated ROC-AUC estimates are explicitly addressed by DeLong et al.
\cite{DeLongROCAOC}. Variation due to the finite evaluation sample is distinct
from variation caused by training subsets, random seeds, or context selection.
The latter requires repeated runs and is part of the benchmark variability
discussed by Bouthillier et al. \cite{bouthillier2021variance}.

This discussion does not select a confidence-interval procedure, repetition
count, or survey-weighting strategy. Those choices remain in the designated
evaluation and NHANES documentation tasks. Until survey treatment is specified
and justified, conclusions should be scoped to the analyzed NHANES cohort;
this draft makes no nationally representative performance claim.

## Citation audit notes (working material, not paper prose)

| Claim | Existing citation key | Support and boundary |
| --- | --- | --- |
| Covariate shift preserves P(Y given X) while P(X) changes; adaptation can reweight source influence. | bickel2009covariateshift | Supports the assumption and adaptation framework, not its validity for these datasets. |
| Label shift preserves P(X given Y); confusion-matrix estimation requires identifiability conditions. | lipton2018detectingcorrectinglabelshift | Supports the definition and method; changed prevalence alone is insufficient. |
| Dataset-shift detection and assessing harm are separate questions. | rabanser2019failingloudly | Supports diagnostic framing, not a claim that our domain-ranked features are harmful. |
| ROC and precision-recall views differ, especially under imbalance. | DavisPrecisionRecall | Supports metric rationale; cross-domain interpretation is our application of that rationale. |
| AP uses recall-increment weighting rather than trapezoidal PR integration. | scikitLearnAveragePrecision | Supports the exact implemented metric; the bibliography identifies version 1.9.1 documentation. |
| Proper scoring rules motivate probabilistic evaluation. | gneiting2007properscoring | Supports Brier/log-loss rationale, not a calibration-only interpretation. |
| Calibration concerns agreement between confidence and observed outcomes. | GuoConfidenceCalibration | Supports calibration interpretation; does not establish calibration for our models. |
| ROC-AUC estimates on the same participants are correlated. | DeLongROCAOC | Supports paired-comparison rationale; does not select our eventual interval method. |
| Learning-procedure randomness contributes to benchmark variation. | bouthillier2021variance | Supports repeated-run rationale; does not prescribe our repetition count. |

The possible sources of mismatch in this project and the proposed reporting
boundaries are study-specific interpretations, not findings attributed to
these papers. The NHANES sampling and survey-treatment rationale will require
the appropriate CDC documentation in the final methods section.
