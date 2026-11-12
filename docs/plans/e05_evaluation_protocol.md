# E05: Candidate A evaluation and uncertainty protocol

Prepared: November 12, 2026.
Status: **Analysis approach approved by Ryan Dozier; implementation defaults
specified below. Planning complete; implementation and execution remain open.**
Prepared with AI assistance. Ryan approved the comparisons, reporting policy,
paired participant bootstrap, interpretation limits, and acceptance checks.
The numerical bootstrap defaults below are implementation choices supplied in
this document, not separately user-selected values.

Companion: [X01 adaptation protocol](x01_adaptation_protocol.md).
This protocol applies to Candidate A only. Candidate B is deferred.

## 1. Comparisons and estimands

For each model, positive real-label budget, and synthetic source:
- Primary: mixed versus matched real-only.
- Supporting: mixed versus same-source synthetic-only anchor.

Pair conditions by root seed (42, 43, 44, 45, 46). Each comparison must use the
same test participant IDs and labels. Real-only baselines are reused across
synthetic-source comparisons and anchors across budgets; they are not independent
replicates simply because they occur in multiple contrasts.

The full grid has 24 primary cells: four models x three budgets x two sources.
There are also 24 supporting cells. Reduce those counts consistently if the
shared budget grid changes through the recorded Newton feasibility decision.

ROC-AUC is primary. AP, Brier score, and log loss are secondary.
Define positive improvement as:
- AUC/AP: mixed minus comparator.
- Brier/log loss: comparator minus mixed.

The estimand is the mean within-seed metric difference across five fitted runs.
Do not average probabilities across seeds first: that would evaluate an ensemble
rather than the agreed repeated-run procedure.

Task-trained mixed-versus-anchor comparisons change validation domain/access as
well as fitting composition. Interpret them as pipeline adaptation contrasts,
not isolated fitting-row replacement effects.

## 2. Descriptive reporting

For each condition and metric, save all five seed values and report their mean
and sample standard deviation (ddof=1). For each contrast, save all five paired
improvements and report their mean and sample standard deviation.

Run-to-run standard deviation describes combined sampling and model variability.
It is not a confidence interval or a separation of the two variance sources.
The five runs share the same evaluation cohort.

Retain the existing fixed-threshold classification and calibration diagnostics
as descriptive outputs under X01. No threshold tuning, calibration fitting,
or additional metric-specific hypothesis testing is introduced.

## 3. Paired stratified participant bootstrap

Implementation defaults:
- 2,000 bootstrap replicates.
- Bootstrap seed 20261112, separate from the experimental root seeds.
- NumPy default_rng; record package version and algorithm.
- Pointwise 95% percentile intervals, using quantiles 0.025 and 0.975 with
  explicit linear interpolation.
- Stratify by observed outcome, preserving positive and negative test counts.

Algorithm:
1. Validate predictions, sort by canonical NHANES participant ID, and align every
   run to that same order. The current test cohort has 545 negative and
   101 positive participants; verify these counts against the frozen manifest.
2. Using a single bootstrap RNG, draw 545 negative indices with replacement,
   followed by 101 positive indices with replacement, for each replicate.
   Use the validated class counts rather than silently assuming fixed constants.
3. Use identical bootstrap indices for all models, conditions, metrics, and
   fitted repetitions. Save the index artifact or sufficient reproducible
   metadata plus its hash; never seed resampling independently by condition.
4. Recompute each metric for each fitted run on those resampled participants.
5. Calculate the signed within-seed improvement and average the five improvements.
6. Take the specified percentiles of the 2,000 mean improvements.

The reported point estimate uses the original test cohort, not the mean of
bootstrap estimates. No refitting or resampling of the five seed identities
occurs within this bootstrap.

Preserving both classes avoids one-class AUC replicates and fixes empirical
prevalence for AP/probability metrics. Invalid or nonfinite predictions, missing
classes in the original cohort, or nonfinite replicate metrics are errors:
do not silently omit replicates or adjust denominators. Use the same metric
implementations and probability handling as the predictive framework.

## 4. Interpretation and multiplicity

Intervals describe finite-participant uncertainty conditional on the five
fitted runs and the selected cohort's observed class counts. They do not include
all uncertainty over future training samples, changing prevalence, or new
populations. Report them alongside the five paired effects and run standard
deviation, without combining them into an unsupported total-uncertainty claim.

This is an unweighted, participant-resampling analysis of the selected NHANES
cohort. It does not account for NHANES clusters, strata, and survey weights and
does not support nationally representative inference. Selection and within-survey
dependence remain limitations; the ordinary participant-bootstrap approximation
must be described as such.

Report all prespecified primary cells, not just favorable ones. The intervals
are pointwise, not simultaneous across models, budgets, sources, or metrics.
No p-value testing, family-wide significance claim, or overall statistically
significant winner is planned. Emphasize effect magnitudes, interval widths, and
consistency across conditions. Crossing zero is inconclusive evidence, not
equivalence; excluding zero in a selected cell is not a multiplicity-adjusted
discovery. No practical-effect/equivalence margin is invented after results.

Supporting contrasts and secondary metrics refine interpretation without
replacing the primary question when its results are unfavorable. Negative
findings are valid project outcomes.

## 5. Incomplete runs and validation requirements

Final comparison tables require all five expected repetitions for both sides.
An incomplete cell must be visibly marked incomplete and omitted from final
five-run intervals; keep failures visible in a run-status table. Do not silently
drop seeds, substitute a different seed, or report a four-run result as five-run.
Any exploratory partial summary must be separately labeled with its actual n.

Acceptance checks for the reporting implementation:
- Exactly matching participant IDs/labels, no duplicates, finite valid probabilities.
- Expected model/source/budget/seed and fitting/validation fingerprints.
- Complete five-seed pairing and correct anchor reuse.
- Correct improvement sign for each metric.
- Identical predictions give zero paired effects and zero bootstrap differences.
- Row-order changes do not change results after ID alignment.
- Shared bootstrap indices preserve class counts and produce reproducible output.
- Mean of per-run metric differences is used, not a metric of averaged predictions.
- Missing runs and schema/provenance mismatches produce explicit incomplete/error states.

Save raw per-seed metrics, paired effects, bootstrap mean-effect distributions,
intervals, configuration, index provenance, run inventory, and software/code
versions. Do not overwrite historical reports. Timing includes resampling/reporting
cost separately from model execution.

## 6. Handoff and approval record

- E05 planning: complete; approach approved by Ryan, defaults documented here.
- X02/reporting implementation: implement these requirements and meaningful checks.
- E06: execute repeats and uncertainty analysis; not completed by this document.
- X01: retain its approved design; Newton feasibility/final budget freeze remains.
- Final test evaluation: wait for the feasibility decision and validated execution
  and reporting paths. Previously inspected baseline outcomes remain disclosed.

| Date | Event |
| --- | --- |
| 2026-11-12 | Ryan approved the proposed comparisons, summaries, paired participant-bootstrap approach, scope/multiplicity limits, and validation requirements. |
| 2026-11-12 | Documented implementation defaults: 2,000 stratified resamples, seed 20261112, and pointwise 95% percentile intervals. No experiments or bootstrap analysis executed. |

Changes to this analysis policy must be dated and justified before the affected
research analysis, with any prior result exposure recorded.
