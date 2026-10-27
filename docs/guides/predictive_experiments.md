# Shared predictive experiments and artifacts

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

All four adapters and their initial matched-budget evaluations are complete.
See [model usage guides](../../Readme.md) for checkpoint/device requirements.
Predictive runs retrain; no runner checkpoint-reuse or resume mode is provided.

The predictive framework evaluates robustness under synthetic generator shift
(Original Synthetic and Playground S5E12 are **both synthetic**) and
synthetic-to-real shift to NHANES 2015–2016.
CatBoost is the reference adapter. Local pretrained TabPFN classification is also
supported for completed research runs. TabICL's explicit none/FP32 profile passed
bounded train/validation-only local checks for both frozen n2000 contexts
(see [TabICL usage](tabicl_usage.md)); both initial n2000 predictive runs are complete (M08).
FT-Transformer's task-trained adapter and full-profile local feasibility
are implemented and verified; its research configs passed final acceptance
(see [FT usage](ft_transformer_usage.md)). Both frozen n2000 seed-42 predictive runs are complete and their
saved artifacts validated; see the [FT results review](../results/ft_transformer_predictive_results.md).

The first configuration fits **one** CatBoost model with seed 42 on Original
Synthetic (`source`) `train`, using the same domain's `val` exclusively for
early stopping and best-model selection. That same model predicts Original
Synthetic, Playground S5E12, and NHANES 2015–2016 `test` using the
ordered `strict_common_16` features. Test sets never select models, preprocessing,
thresholds, or calibration. There is **no target-domain validation** or use of
NHANES labels during fitting/model selection in this initial synthetic-to-real
experiment.

After preparing the canonical datasets, activate the project environment and
run from the repository root:

```bash
python3 -m src.experiments.runner --config configs/catboost_source.json
```

The decision rule is `P(diagnosed_diabetes=1) >= 0.5`, with a fixed threshold
shared across all test domains. `average_precision` is our precision-recall
summary, not trapezoidal PR-curve area. Calibration uses ten fixed equal-width
bins (including empty bins); no probability calibrator is fitted. One-class
evaluation sets report ROC-AUC and average precision as JSON `null` with a warning.

`src/paths.py` anchors output paths. Each run creates its configured directory
inside `results/` exclusively and **refuses to reuse an existing directory**.
Choose a new `output_dir` in the JSON configuration for every rerun. Outputs:

- `config.json`: loader-serialized configuration (absolute output path);
  adapter-resolved parameters are recorded separately in model metadata.
- `metadata.json`: running/completed/failed status, UTC timestamps, schema,
  split/class counts, model/preprocessing/compute details, timings, environment
  versions, SHA-256 prepared-file fingerprints, and namespaced split identities.
- `metrics.json`, `metrics.csv`, and `calibration_bins.csv`: per-domain evaluation.
- `predictions_<domain>_test.csv.gz`: authoritative row indices (and NHANES
  `seqn`) as provenance, labels, positive-class probabilities, and predictions.
- `model.cbm`: the fitted CatBoost model in native format.
- `model.tabpfn_fit` and `model.adapter.json`: TabPFN fitted state and adapter
  sidecar; restoration requires the original checkpoint and supported runtime.
- `model.tabicl_fit` and `model.adapter.json`: TabICL fitted state with embedded
  weights/context and adapter sidecar; the supported restore policy requires
  trusted artifacts and the same saved device and dependency versions.
- `model.ft_transformer.pt`: FT-Transformer selected weights, train-fitted
  preprocessing/schema, constructor settings, versions, and complete training
  history/checkpoint metadata; same-device reload only.
- `training_sample.csv.gz`: actual training context, with domain, `train` split,
  authoritative identifiers, and zero-based `sample_order` (also emitted for
  full-data runs).

## CatBoost behavior and checks

Identifiers never enter predictors. Native categorical handling converts
categorical columns to strings on copies; there is no scaling, one-hot encoding,
external target encoding, or imputation. Missing/nonfinite inputs are rejected.
Model parameters use a small allowlist: `iterations`, `depth`, `learning_rate`,
`loss_function` (Logloss), `eval_metric` (AUC or Logloss), `task_type`,
`thread_count`, `verbose`, and `l2_leaf_reg`. Aliases and framework-controlled
seed, early-stopping, best-model, and automatic-file-writing controls are rejected.
The supplied CPU/four-thread configuration is a reproducibility-oriented starting
point, not a claim that GPU is unavailable.

Failures preserve available artifacts and record a failed status before
re-raising the error. Fixed seeds and fingerprints aid reproducibility, but exact
reproducibility across hardware or package versions is not guaranteed.

Small dataset-independent checks (no full training or preparation):

```bash
python3 -m unittest discover -s tests -v
```
