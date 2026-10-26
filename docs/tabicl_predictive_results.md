# Initial matched TabICL predictive results

Reviewed: 2026-10-26. These are single-seed (42), strict_common_16, n2000 observations. Training identities/order, schemas, prepared-file fingerprints, evaluation identities/labels, thresholded predictions, and four prediction-derived metrics were checked against the matched CatBoost/TabPFN artifacts. No retraining or inference was performed for this review.

TabICL uses the explicit none-normalization, uncached FP32 profile with eight ensemble members and 128-row query chunks. This differs from the vendor-default normalization ensemble. TabPFN uses four ensemble members, float32, batch 256 for Original Synthetic and 1024 for Playground S5E12. Equal row budgets do not equalize pretraining, validation access, ensembles, hardware, or compute.

## ROC-AUC across evaluation domains

| Training domain | Model | Original Synthetic | Playground S5E12 | NHANES |
|---|---|---:|---:|---:|
| Original Synthetic | CatBoost | 0.649753 | 0.665150 | 0.758361 |
| Original Synthetic | TabPFN | 0.649322 | 0.672194 | 0.787864 |
| Original Synthetic | TabICL | 0.650641 | 0.673041 | 0.805995 |
| Playground S5E12 | CatBoost | 0.639871 | 0.679632 | 0.769062 |
| Playground S5E12 | TabPFN | 0.648997 | 0.698164 | 0.780870 |
| Playground S5E12 | TabICL | 0.650522 | 0.687861 | 0.781324 |

Ranking performance (ROC-AUC and average precision), classification performance
at the fixed threshold (F1), and probability quality (Brier score) measure
different aspects of prediction. Brier score describes overall probability
accuracy, not calibration alone.

## NHANES ranking, classification, and probability quality

| Training domain | Model | ROC-AUC | Average precision | F1 at 0.5 | Brier (lower better) |
|---|---|---:|---:|---:|---:|
| Original Synthetic | CatBoost | 0.758361 | 0.342324 | 0.312693 | 0.322067 |
| Original Synthetic | TabPFN | 0.787864 | 0.383181 | 0.351852 | 0.318468 |
| Original Synthetic | TabICL | 0.805995 | 0.412307 | 0.343363 | 0.349823 |
| Playground S5E12 | CatBoost | 0.769062 | 0.374393 | 0.354478 | 0.342607 |
| Playground S5E12 | TabPFN | 0.780870 | 0.375148 | 0.362205 | 0.313163 |
| Playground S5E12 | TabICL | 0.781324 | 0.356710 | 0.349624 | 0.351982 |

Original Synthetic-trained TabICL has higher observed NHANES AUC/AP than the matched CatBoost and TabPFN runs, but a worse Brier score than either. Playground-trained TabICL has almost the same observed NHANES AUC as TabPFN, lower AP, and worse Brier/F1. TabICL is not uniformly better. NHANES has 646 test participants, including 101 positives; small AUC differences should not be treated as established superiority. These aggregate scores do not identify the cause of the probability errors; calibration plots/failure analysis remain useful.

## TabICL execution and resources

| Training domain | Total (s) | Fit/context (s) | Predict Original Synthetic (s) | Predict Playground (s) | Predict NHANES (s) | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Synthetic | 1716.79 | 1.59 | 212.74 | 1475.98 | 10.46 | 348.58 | 478.00 |
| Playground S5E12 | 1696.06 | 1.42 | 209.40 | 1458.83 | 10.59 | 348.58 | 478.00 |

Both TabICL runs completed without detected native recovery or swap. Allocator peaks are not total GPU memory. Recorded total runtimes were approximately 28–29 minutes; full-test uncached inference repeated context work across many chunks. Verbose console echo was enabled in these completed runs. Future quiet-console output must not be presented as a measured speedup without a separate controlled comparison.

## Evidence

- [tabicl_v2_source_strict_common_16_n2000_none_fp32_uncached_seed42](../results/predictive/tabicl_v2_source_strict_common_16_n2000_none_fp32_uncached_seed42/metadata.json): completed `2026-10-26T04:07:31.465085+00:00`; ordered training hash `7d81147cce93cd5c56b2e66484f9f8902ae4ef1784bab270c32016e43f44d170`.
- [tabicl_v2_kaggle_strict_common_16_n2000_none_fp32_uncached_seed42](../results/predictive/tabicl_v2_kaggle_strict_common_16_n2000_none_fp32_uncached_seed42/metadata.json): completed `2026-10-26T04:41:07.295576+00:00`; ordered training hash `612fcfb9dae65a36b591f44065baef3cf629b028435737db4a5480b09469fe3b`.

The integrated E04 comparison report now includes all six selected CatBoost,
TabPFN, and TabICLv2 runs, the four declared candidate-versus-CatBoost
comparisons, calibration support, runtime presentation, and three-model
figures. It is generated from these saved artifacts only; it does not retrain,
load checkpoints, or run inference. See the
[validated report](../results/reports/predictive_n2000_seed42_with_tabicl_figures_final/report.md)
and its [validation record](../results/reports/predictive_n2000_seed42_with_tabicl_figures_final/validation.json).
The figures are single-seed descriptive views, not significance or uncertainty
estimates. Completed TabICLv2 runs included verbose native console echo; the
later quiet-console logging change was not measured, so no speedup is inferred.
