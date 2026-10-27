# FT-Transformer usage and persistence

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

FT-Transformer is a **task-trained neural comparison**, not a pretrained
foundation model. TabPFN/TabICL are pretrained comparisons and CatBoost
remains the conventional reference. Differences in architecture, optimization,
validation-label access, and compute mean this comparison does not isolate
the causal effect of pretraining.

The existing environment manifests pin `rtdl-revisiting-models==0.0.2`.
Torch and the FT implementation are imported lazily when the adapter is used;
unrelated models/config loading do not require the FT runtime. FT training
starts from initialized weights and requires **no pretrained checkpoint
download**. See the [verified implementation/API reference and sources](../plans/ft_transformer_integration_plan.md#19-stage-1a-verified-evidence).

The [Original Synthetic config](../../configs/ft_transformer_source_n2000.json)
and [Playground S5E12 config](../../configs/ft_transformer_kaggle_n2000.json) retain
the successful full-profile feasibility settings:

- `strict_common_16`; exact matched n2000 training identities/order, model
  and sampling seeds 42. Stage 3 verified matches against all three existing
  baselines; no new sampling was needed to prepare these configs.
- Train-fitted numerical standardization; sorted training-only categorical
  vocabularies, with unseen categories aliased to each feature's training mode.
- Three blocks, width 192, eight heads, ReGLU hidden width 256, dropout
  0.2/0.1/0.0; AdamW parameter groups, learning rate 1e-4, weight decay 1e-5.
- Training batch 256; complete fixed same-domain validation every epoch in
  batches of 1024. Restore the highest raw validation ROC-AUC checkpoint;
  exact ties retain the earliest epoch. Validation labels count as additional
  model-selection access, so matched training rows do not imply equal label access.
- Maximum 200 epochs; patience 20, `min_delta=0.001` affects patience only.
  `early_stopping_rounds=null` is separate from CatBoost's stopping setting.
  Explicit float32 and `cuda:0`; no automatic CPU/precision/batch fallback.

**Both commands below were executed by Ryan Dozier; saved artifacts passed
post-run validation (M10).** These are the frozen execution commands, not a
request to rerun. From the project root with `synthetic-tabular-ai` active:

```bash
python3 -m src.experiments.runner \
  --config configs/ft_transformer_source_n2000.json

python3 -m src.experiments.runner \
  --config configs/ft_transformer_kaggle_n2000.json
```

These train on the named domain and evaluate Original Synthetic
(`source`), Playground S5E12 (`kaggle`), and NHANES 2015–2016 (`nhanes`).
Completed output directories (unused at historical Stage 4A validation):

```text
results/predictive/ft_transformer_source_strict_common_16_n2000_float32_seed42/
results/predictive/ft_transformer_kaggle_strict_common_16_n2000_float32_seed42/
```

Every execution requires a fresh directory; choose a distinct output name
for reruns, never reuse or delete completed results. Source JSON retains
the explicit requested settings; the loader canonicalizes the output path,
and adapter-resolved settings appear in `metadata.json` and the saved model.
Unlike smoke mode, the predictive runner has no mode/epoch override.

Outputs include `model.ft_transformer.pt` (weights and preprocessing; see
[shared artifacts](predictive_experiments.md)), full epoch history,
selected/executed epochs and stopping reason in artifact and model metadata,
plus shared predictions, metrics, calibration, training identities/hashes,
prepared-file fingerprints, fit/prediction timings, environment and resource
metadata. Console training progress is concise (first/periodic/final epochs),
not per batch. Resource snapshots retain their operation/reset scope:
later prediction allocator snapshots are not fit peaks or total GPU/process
memory. Detailed operation-specific fit peaks are in the smoke evidence.
Reload requires the same saved device and supported exact runtime versions.
Predictive runs **retrain**: there is no runner checkpoint-reuse/resume mode.

Historical local Stage 3D feasibility observations on the RTX 4060 Laptop GPU
(not predictive-run timings):

| Training domain | Train / full validation | Executed / selected epochs | Stop | Fit / workflow elapsed |
|---|---:|---:|---|---:|
| Original Synthetic | 2,000 / 15,000 | 23 / 3 | Patience | 26.75 / 33.80 s |
| Playground S5E12 | 2,000 / 105,000 | 29 / 13 | Patience | 78.18 / 90.47 s |

Both had fit allocator peaks of 182.82 MiB allocated / 290 MiB reserved.
Same-device save/reload passed, and independent full-validation AUC/BCE
recomputation exactly reproduced each selected checkpoint. These are local
observations for the executed patience-based policy, not 200-epoch timing
guarantees, convergence evidence, or held-out test performance. See
[complete Stage 3 evidence](../results/ft_transformer_stage3_results.md#stage-3d-full-profile-local-feasibility)
and the [integration plan](../plans/ft_transformer_integration_plan.md).
The [predictive results review](../results/ft_transformer_predictive_results.md)
records six FT metric rows, explicit matched-baseline comparisons, full-precision
metric recomputation, identity/provenance checks, and actual predictive timings.
Predictive fits took 16.79 s (source) / 53.60 s (Playground); totals were
31.98 / 66.43 s. Source-trained FT had higher NHANES AUC/AP, while
Playground-trained FT had lower Brier and higher F1@0.5; these measure different
properties and are descriptive single-seed results, not statistical superiority.
E04b adds a separate FT-inclusive four-model report (merged through PR #9); earlier
CatBoost/TabPFN/TabICL reports are preserved.
Historical Stage 4B final validation passed 184 repository tests with no skips,
`pip check`, manifest pin checks, and research/smoke configuration checks.
See the [final review and acceptance record](../plans/ft_transformer_integration_plan.md#21-stage-4b-final-review-and-acceptance)
for the artifact-validation corrections and evidence scope. This is
automated technical acceptance. [PR #7](https://github.com/ryan-dozier/cap6614_diabetes_prediction/pull/7) subsequently merged on 2026-10-27 as [a42668e](https://github.com/ryan-dozier/cap6614_diabetes_prediction/commit/a42668e); human review is not separately documented.
Detailed evidence documents are tracked in Git, but their linked
`results/` outputs and saved checkpoints are ignored local artifacts and
are not supplied by cloning the repository.
