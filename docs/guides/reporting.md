# Predictive comparison reporting

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

The current four-model group contains eight matched-budget runs; E04b merged
through PR #9 on 2026-10-27. Earlier four-run and six-run manifests remain supported.
The sections below retain their distinct schema/profile descriptions.

The legacy artifact-based report uses the explicit, versioned selection in
`configs/reports/predictive_n2000_seed42.json`. It continues to validate the
four matched n=2,000 CatBoost/TabPFN runs and supports both tables-only and
figure-enabled workflows. Its schema-version-1 CSV columns and meanings are
preserved.

The six-run
`configs/reports/predictive_n2000_seed42_with_tabicl.json` manifest selects six
completed runs explicitly (schema version 2). It
compares each TabPFN and TabICL run with the CatBoost baseline in its training
domain—four declared comparisons total—and validates training identity
sequences, evaluation populations and labels, saved profiles, and
prediction-derived metrics. Its metric differences use generic
`candidate_minus_baseline_*` names; it does not add TabICL-versus-TabPFN pairs.
`validation.json` and `manifest.json` record schema versions, input validation
and report-generation status, run/input hashes, declared comparisons, reporting
versions, and available Git provenance.

The schema-version-2 report supports both tables-only and `--with-figures`
workflows. Its figure set contains ROC-AUC, average-precision, and Brier-score
comparisons, six training/evaluation reliability panels, grouped calibration-bin
support, and prediction runtimes (each PNG and PDF). Reliability plots omit
empty bins, while support plots retain their zero counts; common calibration
bin boundaries are validated before plotting. The report distinguishes ranking
performance (ROC-AUC and average precision), fixed-threshold classification
performance (F1), and overall probability quality (Brier score). Runtime
presentation separates timings from device/profile/resource details; memory
units in the report are human-readable while `runtime.csv` retains bytes.

The profiles intentionally differ: CatBoost uses same-domain validation for
early stopping on CPU; TabPFN and TabICL use training context only and execute
on CUDA. TabPFN is v3.5 float32 with four estimators and query batches 256
(Original Synthetic) / 1024 (Playground S5E12). TabICL uses the experimental
TabICLv2 none-normalization, uncached FP32 profile, eight estimators, native
ensemble batch size 1, CPU column-output storage, and external query batch 128.
These matched row budgets do not make the methods compute-matched; pretraining,
validation access, ensemble design, device, and runtime are confounders. The
TabICL none-normalization profile differs from its vendor-default normalization
ensemble.

Both workflows read saved artifacts only: they do not train, load checkpoints,
or run inference. The completed TabICLv2 runs included verbose native console
echo; the later quiet-console logging change was not measured in those runs, so
no speedup is inferred. Bootstrap intervals, multi-seed aggregation, and
full-data reference comparisons are not implemented.

From the project root, choose a fresh output directory for each invocation.
The default is tables-only and does not import Matplotlib:

```bash
python3 -m src.reporting.compare_predictive \
  --manifest configs/reports/predictive_n2000_seed42.json \
  --output-dir results/reports/predictive_n2000_seed42_tables
```

The six-run schema-version-2 tables-only report uses the separate manifest and
a fresh output directory:

```bash
python3 -m src.reporting.compare_predictive \
  --manifest configs/reports/predictive_n2000_seed42_with_tabicl.json \
  --output-dir results/reports/predictive_n2000_seed42_with_tabicl_tables
```

Generate the integrated three-model report with figures (use a new output
directory for each run):

```bash
python3 -m src.reporting.compare_predictive \
  --manifest configs/reports/predictive_n2000_seed42_with_tabicl.json \
  --output-dir results/reports/predictive_n2000_seed42_with_tabicl_figures_final \
  --with-figures
```

The manifest explicitly selects every run; runs are never discovered or chosen
by recency or score. Reports contain `metrics.csv`,
`metric_differences.csv`, `runtime.csv`, and `calibration_bins.csv`, plus
`report.md`, `manifest.json`, and `validation.json`. Figure-enabled reports
include a `figures/` directory with metric, reliability, bin-support, and
runtime plots. The report manifest records figure selection, generated
filenames, Matplotlib version when applicable, consumed artifact hashes,
reporting package versions, and available Git provenance. All required files
must be nonempty and report-relative links must resolve before completion is
recorded; failed generation retains a failure diagnostic.

Output paths resolve from the project root and must be new directories under
`results/reports/`. Existing output paths and paths overlapping input runs are
refused. Input validation and report-generation completion are recorded
separately; failures during validation, plotting, or writing leave a failed
diagnostic and are never reported as completed.

Figures are descriptive: these are single-seed fixed-split observations, not
uncertainty estimates or significance claims. Equal training-row budgets do
not equalize external pretraining, validation access, hardware, or compute.
Average precision and Brier scores depend on the evaluation population and
prevalence; NHANES results describe the held-out sample and protocol, not
national-population estimates. No model selection, threshold tuning, or
calibration fitting is performed.

## Four-model comparison report (E04b)

The [eight-run manifest](../../configs/reports/predictive_n2000_seed42_with_ft_transformer.json)
extends schema version 2 with the frozen task-trained FT-Transformer profile.
Legacy four-run and six-run manifests remain supported.

```bash
python3 -m src.reporting.compare_predictive --manifest configs/reports/predictive_n2000_seed42_with_ft_transformer.json --output-dir results/reports/predictive_n2000_seed42_with_ft_transformer --with-figures
```

This output directory was generated and validated on 2026-10-27. Select a new
directory when repeating the command; existing outputs are never overwritten.
[Report](../../results/reports/predictive_n2000_seed42_with_ft_transformer/report.md) /
[validation](../../results/reports/predictive_n2000_seed42_with_ft_transformer/validation.json).

Eight experiments produce 24 metric rows, 18 candidate-minus-CatBoost differences,
eight runtime rows, 240 calibration rows, and six PNG/PDF pairs. All 21
reference-to-run population checks passed; all six PNGs were inspected.
Forty reporting tests passed, including legacy compatibility.
Reporting uses saved predictions only. FT fitting includes training and full-validation
checkpoint selection, not foundation-model context preparation. Equal training
budgets do not imply equal validation-label access, pretraining, or compute.
Allocator peaks retain their recorded cumulative scope. Conclusions remain single-seed.

Results are ignored local artifacts, not supplied by cloning the repository.
E04b merged through [PR #9](https://github.com/ryan-dozier/cap6614_diabetes_prediction/pull/9),
implementation 78ec6de, manifest commit 02d1290, merge ba9e16a on 2026-10-27.
The 40-test validation and six-figure inspection above are recorded historical checks.
