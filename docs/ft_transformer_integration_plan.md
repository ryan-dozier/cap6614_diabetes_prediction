# FT-Transformer integration plan

Status: Stage 2C integration implemented — ready for Stage 3 local feasibility
Owner: Ryan Dozier
Stage 1A/1B design prepared by: AI assistant using Copilot SDK in VS Code
Reviewed by: Not yet human-reviewed
Last updated: 2026-10-26

## 1. Purpose and research role

Add FT-Transformer as a task-trained neural comparison to the existing
CatBoost, TabPFN, and TabICL experiments.

The initial research question is:

> How does a transformer trained on the available task data compare with
> pretrained tabular models under synthetic generator shift and
> synthetic-to-real transfer?

Evaluate differences in ranking performance, probability quality, and
cross-domain transfer under the existing matched-training-row protocol.

FT-Transformer also provides a model whose weights can later be adapted
using limited NHANES training data. Adaptation is a separate implementation
and experimental task.

FT-Transformer is not a pretrained foundation model. CatBoost remains the
conventional reference baseline.

This comparison does not isolate the causal effect of pretraining.
Architecture, capacity, optimization, pretraining exposure, and validation
use also differ between models.

Successful integration does not require FT-Transformer to outperform another
model. A reproducible negative result is a valid experimental outcome.

Adding this baseline does not itself fulfill the project's potentially
novel contribution. Artifact-aware interventions, adaptation, and their
ablations remain separately tracked research tasks.

This baseline contributes to checklist task M09 but does not complete it.
Full predictive experiments belong to M10.

## 2. Scope

### Included

- Adapter compatible with the existing model interface.
- Train-only numerical and categorical preprocessing.
- Supervised binary classification training.
- Same-domain validation-based checkpoint selection.
- Batched probability prediction.
- Model and preprocessing persistence.
- Training history, timing, and reproducibility metadata.
- Focused tests and a bounded train/validation-only smoke workflow.
- Matched-budget research configurations for both synthetic domains.
- Documentation sufficient to reproduce and interpret the model profile.

### Deferred

- Full-dataset training.
- Automated hyperparameter searches.
- NHANES adaptation or fine-tuning.
- SAINT or additional architectures.
- Distributed training.
- Resuming interrupted training.
- General-purpose missing-data support unless required by the input contract.
- Predictive reporting expansion before new runs are available.

Preserve canonical datasets, split assignments, and completed experiment
artifacts.

## 3. Implementation stages

| Stage | Objective | Required outcome |
|---|---|---|
| 1. Design and compatibility | Verify the implementation and resolve experimental choices | Completed decision tables and an implementation-ready plan |
| 2. Adapter and focused tests | Implement preprocessing, training, prediction, and persistence | Tested adapter with minimal framework changes |
| 3. Local feasibility | Verify correct behavior and practical resource use | Recorded train/validation-only evidence for both synthetic domains |
| 4. Research configuration and documentation | Freeze the accepted profile and prepare research runs | Validated configs, updated documentation, and M09 acceptance record |

Review each stage before proceeding.

Stage 1 is documentation and inspection work. It does not install or
upgrade dependencies, implement the adapter, or run research experiments.

## 4. Existing project contracts

Inspect:

- src/models/base.py
- src/models/catboost.py
- src/models/tabpfn.py
- src/models/tabicl.py
- src/experiments/config.py
- src/experiments/runner.py
- src/experiments/sampling.py
- src/experiments/results.py
- src/experiments/metrics.py
- src/data.py
- Relevant predictive framework and adapter tests

Record the actual interface requirements after inspection, including:

- Fit and prediction signatures.
- Validation-data handling.
- Positive-class probability format.
- Save/load contract.
- Configuration validation and model registration.
- Sampling and identity provenance.
- Timing, metadata, and failure reporting.

Preserve:

- Internal domain IDs: source, kaggle, nhanes.
- Existing descriptive display names.
- Identifier exclusion.
- Deterministic matched-budget sampling.
- Existing adapter and configuration compatibility.
- Fresh output directories.
- Existing result lifecycle and failure diagnostics.

Reuse shared utilities where practical. Make shared-interface changes only
when a concrete requirement justifies them.

## 5. Implementation and dependency selection

Adopt the verified `rtdl-revisiting-models==0.0.2` implementation. During
Stage 2, add this exact pin to `requirements-local.txt` and `environment.yml`;
preserve the existing PyTorch pin. No incompatibility was found in the
installed implementation or tiny CPU/CUDA checks.

Record:

| Item | Verified value |
|---|---|
| Implementation/repository | `rtdl_revisiting_models.FTTransformer`; official Yandex Research package |
| Package version or source revision | Installed version `0.0.2`; installed source SHA-256 matches official upstream file at commit `e3ed46cac38568785289d8fa16b8cfa585bde27e`; no release tag was found |
| License | Apache License 2.0 in the installed distribution's `LICENSE` file |
| Official paper/reference | Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021; arXiv:2106.11959 |
| Documented Python/PyTorch requirements | Package metadata: Python >=3.8 and `torch>=1.8,<3`; installed environment is Python 3.11 and PyTorch 2.14.1+cu126 |
| Actual constructor and forward API | Verified from installed source; see Stage 1A evidence below |
| Additional dependencies | Only declared runtime dependency is PyTorch; it is already in the project environment |
| Compatibility with the current environment | Tiny float32 forward/backward/optimizer/evaluation checks passed on CPU and CUDA; this is not a convergence, full reproducibility, resource-feasibility, or research-readiness result |
| Dependency declaration | Stage 2C adds `rtdl-revisiting-models==0.0.2` to both existing environment manifests; preserve `torch==2.14.1+cu126` |
| Upstream training/preprocessing example | Official package README and example notebook at the verified upstream commit; recommendations and their limits are recorded below |

The dependency version and license are verified. The exact project-manifest
pin is now present in both manifests; no package was installed and the
PyTorch pin was not changed.

## 6. Initial experiment protocol

Primary feature set: strict_common_16.

Initial training domains:

- Original Synthetic — source.
- Playground S5E12 — kaggle.

For each training domain:

- Select 2,000 training rows using the existing sampling mechanism.
- Use seed 42 for the initial comparison.
- Match the exact training identities and row order of the corresponding
  existing n2000 baselines.
- Use same-domain validation data for checkpoint selection.
- Evaluate the selected model on the fixed test partitions of all three
  domains through the existing runner.

Training batches may shuffle the selected rows deterministically.
The stored sampling order and identity provenance must still match the
existing experiment protocol.

Do not use NHANES validation or any test results to select the
synthetic-trained baseline's preprocessing, architecture, stopping policy,
or hyperparameters.

Matching training rows does not establish equal computation, equal total
label access, or equal tuning effort. Document these differences.

The full validation split supplies labels for repeated epoch-level model
selection, unlike pretrained comparisons that do not use validation labels.
Equal training-row counts therefore do not imply equal total label access,
compute, or tuning effort.

Additional seeds and larger training budgets require distinct
configurations and result directories.

## 7. Feature schema and preprocessing

### 7.1 Feature schema

Obtain numerical and categorical feature definitions from the existing
data/schema contracts.

Record the exact ordered feature lists. Do not infer categorical status
solely from storage dtype or query-batch values.

Exclude targets and identifiers from predictors.

Define handling of:

- Missing columns.
- Unexpected columns.
- Duplicate columns.
- Reordered columns.
- Invalid values and incompatible types.

Follow existing adapter conventions where appropriate. Fail explicitly
when safe reconciliation is not possible.

### 7.2 Numerical preprocessing

Decision — use per-column `sklearn.preprocessing.StandardScaler` fitted on
the selected training rows only (`with_mean=True`, `with_std=True`), with
float64 fit statistics and float32 transformed model inputs. Reuse the fitted
mean and scale for train, full validation, and every prediction batch.
Persist the ordered means/scales and feature names; do not refit on reload.

This is an engineering choice, not a claim of empirical optimality. The
upstream example offers StandardScaler as its simple option and also shows
a train-fitted QuantileTransformer with small seeded fitting noise. Prefer
standardization here because:

- The research question includes synthetic-to-real and generator shift.
  Standardization preserves unbounded linear extrapolation beyond the
  training range, whereas quantile transforms can saturate out-of-range
  values at endpoint quantiles.
- Several schema-numeric fields are binary indicators. Standardization
  preserves their two ordered levels, while quantile ranks and the example's
  noise used only during fitting complicate ties and the mapping of those
  levels.
- The training-only mean and scale are deterministic, compact, and easy to
  audit/persist. A training-constant column has scale 1: its training values
  transform to zero, while any held-out value transforms to
  `value - training_mean`. Do not force the whole column to zero at
  validation or prediction time; repeated values need no tie-breaking.

This deliberately departs from the example's QuantileTransformer choice:
no synthetic jitter, quantile ranking, clipping, or test-informed
transformation is used. No preprocessing search is authorized.

Keep the complete-case input contract: use the existing base adapter
validation, reject missing and non-finite values explicitly, and do not add
imputation. Reject non-finite transformed float32 arrays rather than
silently continuing. All statistics come only from the exact selected
training rows.

### 7.3 Categorical preprocessing

Use the existing `ModelAdapter._inputs` contract first: categorical values
are copied and converted with `astype(str)`, and missing/non-finite
categorical values are rejected. For each categorical feature independently,
build a vocabulary from selected training rows only by sorting its distinct
normalized strings in ascending Python string order and assigning indices
`0..K-1`. The FT input column order is the categorical subset of the
authoritative feature order; cardinality is exactly `K`.

For an unseen query value, map it to the training mode category's ordinary
index. Determine the mode from selected training counts; break frequency
ties by the lexicographically smallest normalized string. This reuses an
embedding row that received normal training updates and needs no reserved
embedding, model-source modification, extra synthetic-category examples, or
category-dropout intervention. The deliberate tradeoff is that unseen
values become indistinguishable from that feature's training mode; record
this limitation and do not interpret the alias as learned unknown semantics.
The mode embedding continues to update from its genuine training examples;
unknown query rows never update it.

Keep mapping and mode indices in fitted adapter state. Best-checkpoint
selection applies only to model weights; restore the selected weights
without refitting or changing preprocessing. Save mappings/mode indices in
the same model artifact and restore them unchanged on reload.

Record per-feature unseen counts on the full validation split and separately
for each prediction domain. Counts are diagnostics only: validation counts
do not change mappings, and test-domain counts are recorded after model
selection and never feed back into training, tuning, or later test-domain
predictions. No validation/test categories expand vocabularies.

### 7.4 Preprocessing persistence

Save:

- Fitted numerical transformations.
- Categorical mappings.
- Per-feature fallback mode indices and the unknown-category alias policy.
- Feature types and order.
- Required preprocessing version/configuration information.

Reloading must not refit preprocessing or require the original training CSV.

Prediction must not mutate preprocessing state or input data.

## 8. Architecture and optimization

Use supervised binary classification with an appropriate numerically
stable loss: one output logit (`d_out=1`), binary labels as float32, and
`BCEWithLogitsLoss`. Apply sigmoid only at inference to return positive-class
probabilities through the existing adapter contract.

Frozen initial engineering profile (same for source and kaggle):

| Setting | Decision |
|---|---|
| Transformer blocks | 3 |
| Token/embedding dimension | 192 |
| Attention heads | 8 |
| FFN hidden dimension/activation | Multiplier `4/3` (hidden width 256); `ReGLU` |
| Attention / FFN / residual dropout | 0.2 / 0.1 / 0.0 |
| Optimizer and parameter groups | AdamW using `model.make_parameter_groups()`; explicit learning rate and weight decay below |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Training batch size | 256 |
| Maximum epochs | 200 |
| Validation frequency and batch size | Every epoch; full fixed validation split in batches of 1024 |
| Checkpoint selection | Highest finite full-validation ROC-AUC; exact ties keep earliest epoch |
| Patience and minimum improvement | 20 consecutive epochs without ROC-AUC improvement of at least 0.001 |
| Prediction batch size | 256 |
| Epoch numbering | 1-based; history records every completed epoch |
| Precision | float32; no autocast or mixed precision |
| Device | Explicit `cuda:0` in the initial research configs; fail if unavailable, no fallback |
| Class weighting | None |
| Numerical preprocessing | Train-only StandardScaler as specified in §7.2 |
| Unknown categories | Map to the training mode category's learned embedding as specified in §7.3 |
| Random seed | 42 for model initialization and a dedicated deterministic training-row shuffle generator |
| Determinism controls | Seed Python/NumPy via runner and Torch CPU/CUDA in adapter; CUDA TF32 off; cuDNN benchmark off and deterministic on; Torch deterministic-algorithms mode remains off; do not promise bitwise repeatability across devices, libraries, or hardware |

The three-block settings use the verified upstream default kwargs and retain
its attention/FFN/residual dropout defaults. Use the upstream parameter-group
helper, but set learning rate and weight decay explicitly as shown above;
the library's optimizer helper/example settings are guidance, not binding
requirements. The 256 minibatch follows the official example; 200 epochs, patience 20,
0.001 AUC delta, validation batch 1024, prediction batch 256, and explicit
CUDA/float32 are project engineering choices, not empirically optimized
settings. Use this single predefined recipe for both training domains.

Training must:

- Set model-initialization and data-shuffling seeds.
- Use training mode for optimization.
- Use evaluation mode and disabled gradients for validation and prediction.
- Compute validation metrics over the entire declared validation population.
- Restore the selected checkpoint before final prediction and saving.
- Report selected epoch and actual epochs executed.
- Check loss, gradients before the optimizer step, and updated parameters for
  finiteness; fail with useful diagnostics rather than attempting an
  unconfigured recovery on non-finite values or unrecoverable OOM.
- Avoid silent device, precision, architecture, or batch-size changes.

Record deterministic settings and limitations. Exact repeatability across
different hardware or dependency versions is not assumed.

## 9. Validation and checkpoint selection

### 9.1 Validation population

Use the complete fixed same-domain validation split for every validation
epoch, in batches of 1024 without changing membership. Do not implement
automatic validation subsetting. Measure full-validation runtime and memory
in Stage 3; if it proves impractical, revise and document the profile
explicitly before any research evaluation.

### 9.2 Selection policy

At the end of each 1-based epoch (one validation check per epoch, so
patience counts epochs and validation checks), compute sample-weighted
validation BCE and ROC-AUC over concatenated full-population predictions.
Require both label classes in validation and finite logits, loss, and
ROC-AUC. Fail explicitly before training if validation has one class; fail
the run if any required metric becomes undefined or non-finite.

Checkpoint choice and early stopping are separate:

- Save the checkpoint whenever raw finite ROC-AUC is strictly greater than
  the highest previously seen score. An exact tie retains the earliest
  checkpoint; any strict raw increase, however small, becomes the best
  checkpoint.
- Initialize the patience reference to the first epoch's finite ROC-AUC and
  the counter to zero. Thereafter, reset the counter only when ROC-AUC
  improves by at least 0.001 over the last patience-reset reference score,
  and move the reference to that score. Otherwise increment it. Stop after
  20 consecutive completed epochs without such an improvement, or at epoch
  200.
- Restore the highest-ROC-AUC checkpoint before prediction and persistence.
  Record its epoch, the patience reference, actual epochs executed, and
  whether stopping occurred through patience or the epoch cap.

If patience is exhausted on the maximum-epoch validation check, record
`patience` as the stopping reason because that condition is checked first;
the epoch count still records the maximum reached.

The minimum-improvement threshold affects patience only; it cannot cause the
saved checkpoint to differ from the highest-ROC-AUC epoch.

### 9.3 Information boundaries

Training rows supply:

- Weight updates.
- Numerical preprocessing statistics.
- Category vocabularies.

Same-domain validation data supplies:

- Checkpoint selection.
- Early stopping.
- Explicitly declared development decisions.

Test partitions supply final evaluation only and are not inspected during
training, validation selection, or development. Do not load NHANES
validation for a source- or kaggle-trained profile.

Smoke workflows must not load test partitions.

Previously observed test scores must not drive undocumented tuning.
Any further tuning requires a predefined validation-only protocol and
budget.

## 10. Adapter integration

Planned location:

- src/models/ft_transformer.py

The adapter location and registration points were inspected in Stage 1A;
the minimal proposed file map is recorded below.

Keep optional model dependencies lazy so other models and reporting remain
usable without the FT-Transformer dependency installed.

Reuse existing:

- Data loading.
- Sampling and identity checks.
- Metrics.
- Timing.
- Output lifecycle.
- Provenance.
- Failure handling.

Use the runner's existing `fit(X_train, y_train, X_val, y_val)` call and
fresh output-directory lifecycle. Add only the FT model branch to config
validation, lazy adapter construction, model-specific environment metadata,
and validation-based selection-policy text. Preserve the existing base
adapter, sampler, metric, and result contracts.

The runner loads validation before `fit`; the adapter evaluates the entire
passed validation population. During `fit`, save validation unseen counts
with adapter metadata. Add one adapter diagnostic method,
`unseen_category_counts(X)`, that validates/maps a query without inference
and returns per-feature counts. After each FT prediction, the runner calls
it on that domain's already-loaded `X`, stores the result under the domain
in run metadata, and refreshes `metadata["model"]`. These counts never
affect subsequent predictions or model settings. Keep epoch history and
selected epoch in the existing `metadata["model"]` extension point rather
than changing shared result schemas.

## 11. Persistence and future adaptation

Save one adapter artifact, `model.ft_transformer.pt`, in the fresh run
directory. Use `torch.save` for a plain mapping of primitive metadata and
CPU tensor `state_dict` values; restore with
`torch.load(..., map_location=device, weights_only=True)`. Do not pickle
adapter or sklearn objects. Write to a temporary sibling then atomically
replace the final path.

The artifact payload contains format version 1; explicit constructor
settings; selected state dict; ordered feature/numeric/categorical schema;
StandardScaler means/scales; sorted category vocabularies, cardinalities,
fallback mode indices and policy; `P(y=1)` class meaning; preprocessing
version; model/library/PyTorch versions; seed/device/precision; training
history; best epoch, epochs executed and stopping reason; full-validation
size/selection policy; and a fingerprint of the fitted training matrix and
labels. The adjacent run `metadata.json` and `training_sample.csv.gz`
continue to provide the sampler's exact ordered identity hashes and
identity rows.

`load(path, *, device)` requires a complete existing file, supported
artifact format, exact FT package version 0.0.2 and exact recorded PyTorch
version, exact schema/architecture fields, and strict state-dict key/shape
agreement. Read with `map_location="cpu"` and `weights_only=True`, validate
metadata before moving model tensors to the requested device, then construct
and strictly restore the model. Reject a saved device that differs from the
requested device; same-device reload is the initial acceptance criterion.
Cross-device load is deferred and must not be claimed supported. Corrupt,
incomplete, or incompatible artifacts fail explicitly; never refit
preprocessing or return a partial/success-shaped adapter.

The content fingerprint is distinct from sampler identity hashes. Serialize
the schema as sorted-key compact JSON, then hash the row count, ordered raw
normalized numeric values as little-endian float64 C-order bytes, each
categorical column's ordered normalized strings as compact UTF-8 JSON with
length prefixes, and labels as little-endian int64 C-order bytes, using
SHA-256. Include a fingerprint format version in the schema portion.

The saved state dict must also permit a later adaptation implementation to
initialize the model from its learned weights; adaptation preprocessing and
optimizer state are not part of this support.

A future adaptation protocol must decide explicitly whether preprocessing
is frozen or updated using permitted target-domain training data.

Optimizer-state restoration, resumable training, and adaptation itself are
outside this implementation's scope.

## 12. Diagnostics and console output

Record:

- Epoch number.
- Training loss.
- Validation loss.
- Validation selection metric.
- Selected checkpoint and epoch.
- Actual epochs executed.
- Stopping reason.
- Validation population size.
- Fit and prediction durations.
- GPU memory measurements where supported.
- Per-feature validation unseen counts and per-prediction-domain unseen
  counts.

Default console output should show:

- Run/model summary.
- Training start.
- Periodic epoch summaries.
- Stopping and checkpoint-selection summary.
- Evaluation-domain progress.
- Final metrics and output location.

Save detailed epoch history in the versioned model artifact and mirrored
model metadata rather than printing every batch. Keep warnings and failures
visible. Include training loss, validation loss/AUC, epoch durations,
population size, selected epoch, actual epochs and stopping reason.

## 13. Focused verification

### 13.1 Preprocessing

Verify:

- StandardScaler statistics use train rows only; held-out extreme values do
  not change fitted means/scales, and an out-of-range query remains
  unsaturated under the stored linear transform.
- A training-constant numeric column uses scale 1, so its training values
  transform to zero; add a test where its validation value differs from the
  training constant and assert that it transforms to
  `validation_value - training_mean`, not zero. Prediction applies the same
  rule and must not forcibly zero the column. Repeated and binary-indicator
  values remain finite and deterministic.
- Category strings follow `_inputs` normalization; sorted vocabulary indices
  and modal fallback are deterministic, including lexical tie-breaking.
- Validation-only categories map to the selected training mode embedding,
  do not expand cardinality, and increment only their feature count.
- Prediction does not mutate fitted preprocessing state or input frames.
- Schema and feature-order behavior matches the documented contract.
- Unexpected missing or non-finite values fail as declared.

### 13.2 Training

Verify:

- Parameters change during optimization.
- A deterministic, deliberately learnable in-memory fixture has one numeric
  feature with rows `[-2, -1, -2, -1, 1, 2, 1, 2]` and labels
  `[0, 0, 0, 0, 1, 1, 1, 1]`. Fit the declared train-only StandardScaler on
  these eight rows, use no categorical features, and run on CPU after setting
  Python, NumPy, and Torch seeds to 42. Train a one-block `d_block=16`,
  four-head, FFN-width-32 ReGLU model with one output logit and all dropouts
  zero, full-batch size 8, and `torch.optim.AdamW` with `lr=0.01`,
  `weight_decay=0`. Run exactly 500 epochs without validation or
  validation-based early stopping. At the final epoch require training
  BCE-with-logits below 0.05 and 100% training accuracy. Investigate failures
  before changing this criterion. This checks optimization capability under
  a separate test-only profile, not convergence of the research profile;
  controlled tests separately verify research checkpoint selection and
  patience behavior.
- Best-checkpoint restoration is correct even when later epochs worsen.
- Raw-score checkpoint selection, exact ties, sub-0.001 gains, qualifying
  gains, patience reset/count, epoch-cap stop, and undefined/non-finite
  metric failures follow §9 exactly.
- Training history identifies the selected checkpoint correctly.

Use a controlled fixture or lightweight substitute for checkpoint-selection
tests rather than relying on an unstable real training trajectory.

Do not require high training accuracy on noisy diabetes data as a
correctness criterion.

### 13.3 Prediction

Verify:

- Correct output shape and positive-class meaning.
- Finite probabilities in [0, 1].
- One-dimensional shape, preserved row order, and correct partial final
  batch at prediction size 256.
- Same-device repeat, inverse-permutation, and batch-size-equivalence checks
  at `rtol=1e-5`, `atol=1e-6`, using batch sizes 1, 3, and 8 on an 11-row
  fixture to cover partial batches.
- Do not require CPU and CUDA training losses or weights to match; the
  Stage 1A CPU/CUDA checks were separate training-mode compatibility
  executions with dropout, not cross-device equivalence tests.

Define numerical tolerances before accepting results. Investigate failures
before changing a tolerance.

### 13.4 Persistence

Verify:

- Save/reload probability agreement.
- Preservation of preprocessing and schema.
- Reload without refitting.
- Clear errors for missing or incompatible artifacts.
- Same-device reload agreement at `rtol=1e-5`, `atol=1e-6`; cross-device
  loading remains deferred.
- Weight-only loading and rejection of unsupported format/package/device.

### 13.5 Framework compatibility

Verify:

- Existing model configurations remain valid.
- Relevant focused tests pass.
- The repository test suite passes before finalization.
- Matched training identities/order agree with existing baselines.

## 14. Local feasibility workflow

Prefer a small command using the actual adapter and existing utilities.

Chosen bounded smoke entry point:

- src/experiments/smoke_ft_transformer.py

Reuse `load_partition`, `select_training`, schema/metric utilities, the
actual adapter, and a fresh output directory. Load training and same-domain
validation only; never load test or NHANES validation for a synthetic-trained
smoke run. Do not add a general benchmark harness.

### Gate A: Functional correctness

Use an in-memory synthetic fixture for focused unit tests. The smoke command
uses at most 64 selected training rows and 32 deterministically selected
same-domain validation rows, both classes required, three epochs maximum,
and a tiny test architecture. These bounded smoke subsets do not replace
the complete validation population required by the initial research
profile. It exercises:

- Preprocessing.
- Optimization.
- Checkpoint selection.
- Prediction.
- Persistence.
- Diagnostics.

### Gate B: Resource estimate

For each domain, use the exact matched n2000 training sample and the complete
validation split with the frozen architecture. Begin with two measured
epochs. Review the measurements and extend one epoch at a time, only when
needed for a useful resource estimate, up to ten epochs total. Save to a
distinct smoke/resource directory; record both the actual epoch count and
`truncated: true`. This is not a convergence or model-quality result. Measure:

- Training epoch cost.
- Validation cost.
- Peak memory.
- Prediction batching behavior.

Do not prescribe an arbitrary runtime or memory threshold before measuring
the installed hardware. Record elapsed time, per-epoch training/validation
time, peak allocated/reserved GPU memory, and available device memory. After
Gate B, review those measured costs and the remaining resource budget before
launching either full Gate C run; do not start Gate C automatically if the
estimate appears impractical.

### Gate C: Research-profile feasibility

Exercise the intended research configuration for both synthetic domains:

- 2,000 matched training rows.
- Declared validation population.
- Exact frozen architecture and optimization policy, up to 200 epochs and
  patience-based stop.
- Selected-checkpoint persistence.
- Full validation evaluation each epoch in fixed-size batches.

Record stopping behavior, duration, memory, and any failures.

Gate C is mathematically bounded by 200 epochs and full-profile patience;
there is no automatic retry, smaller batch, CPU fallback, or validation
subset. A resource failure is recorded and blocks feasibility until the
profile is explicitly revised before research evaluation.

If a feasibility run fully satisfies the frozen research training protocol,
its checkpoint may be reused for later evaluation only when the runner's
artifact lifecycle supports that reuse with complete provenance.

Otherwise, perform a fresh research run. Avoid adding complex checkpoint
reuse machinery solely to save one small training run.

Passing these gates establishes correctness and local feasibility.
It does not establish predictive superiority or statistical significance.

## 15. Research configurations

Create exactly these initial research profiles after Stage 2 and Stage 3
functional checks pass:

- configs/ft_transformer_source_n2000.json
- configs/ft_transformer_kaggle_n2000.json
- `configs/smoke/ft_transformer_source.json` and
  `configs/smoke/ft_transformer_kaggle.json` for bounded train/validation
  checks with separate output directories

Configurations must identify:

- Model profile.
- Feature set.
- Training domain.
- Training sample budget and seed.
- Architecture.
- Preprocessing.
- Optimization and stopping policy.
- Validation population policy.
- Device and precision.
- Prediction batch size.

Each research `model_params` object uses the profile values in §8:
`n_blocks=3`, `d_block=192`, `attention_n_heads=8`,
`attention_dropout=0.2`, `ffn_d_hidden=null`,
`ffn_d_hidden_multiplier=1.3333333333333333`, `ffn_activation="ReGLU"`,
`ffn_dropout=0.1`, `residual_dropout=0.0`, `learning_rate=0.0001`,
`weight_decay=0.00001`, `train_batch_size=256`, `max_epochs=200`,
`validation_frequency=1`, `validation_batch_size=1024`, `patience=20`,
`min_delta=0.001`, `preprocessing="standard_scaler"`,
`unknown_category_policy="training_mode"`, `precision="float32"`,
`device="cuda:0"`, and `class_weighting="none"`. Set top-level
`prediction_batch_size=256`, `seed=42`, `sampling_seed=42`,
`train_sample_size=2000`, same-domain `validation_domain`, and
`early_stopping_rounds=null`. The model-specific config validator must
reject unknown keys, unsupported values, and inconsistent architecture or
training settings.

Freeze the accepted profile before full test evaluation.

Meaningful configuration changes after results exist require a distinct
profile/configuration and output directory.

Reporting integration follows completion and validation of M10 runs.

## 16. Acceptance criteria for M09

- [ ] Implementation, version/revision, license, and API verified.
- [ ] Implementation reference added to the bibliography or reference record.
- [ ] Research role and comparison limitations documented.
- [ ] Architecture and optimization settings resolved.
- [ ] Numerical preprocessing and unknown-category behavior resolved.
- [ ] Validation population and development budget recorded.
- [ ] Adapter integrated without regressions to existing models.
- [ ] Train-only preprocessing checks pass.
- [ ] Training and best-checkpoint restoration checks pass.
- [ ] Prediction ordering and batching checks pass.
- [ ] Save/reload checks pass.
- [ ] Both domains pass the declared feasibility gates.
- [ ] Exact matched training identities/order verified.
- [ ] Research configurations frozen.
- [ ] Relevant and full repository tests pass.
- [ ] README and checklist reflect verified status.
- [ ] Full predictive runs remain separately tracked under M10.
- [ ] Adaptation, artifact ablations, and uncertainty analysis remain tracked.

## 17. Review and evidence record

| Stage | Date | Performed by | Reviewed by | Evidence |
|---|---|---|---|---|
| Stage 1A inspection | 2026-10-26 | AI assistant using Copilot SDK in VS Code | Not yet human-reviewed | Installed source/API and project contracts inspected; synthetic CPU/CUDA check passed; see Stage 1A evidence below |
| Stage 1B design | 2026-10-26 | AI assistant using Copilot SDK in VS Code | Not yet human-reviewed | Concrete implementation profile and acceptance criteria recorded in §§6–15 and 20 |
| Stage 2A preprocessing/prediction foundation | 2026-10-26 | AI assistant using Copilot SDK in VS Code | Not yet human-reviewed | Added preprocessing/prediction adapter foundation and focused tests; 14 FT adapter tests and 21 existing predictive-framework tests passed in `synthetic-tabular-ai`; training, persistence, and runner/config integration remain deferred |
| Training and persistence (Stage 2B) | 2026-10-26 | AI assistant using Copilot SDK in VS Code | Not yet human-reviewed | 30 FT adapter tests passed; 500-epoch CPU overfit diagnostic reached BCE 0.00013014 and 100% accuracy in 5.145 s; CPU and CUDA fit/predict/save/load checks passed; CPU save/reload agreement and existing predictive-framework (21) and TabICL-adapter (17) tests passed. No project data or research profile was run. Stage 2C runner/config integration remains deferred. |
| Stage 2B review corrections | 2026-10-26 | AI assistant using Copilot SDK in VS Code | Not yet human-reviewed | Prediction now scopes/restores declared backend settings without reseeding, including loaded adapters; loading rejects NaN/Inf state tensors; constructor canonicalizes categorical order to feature order for artifact consistency. FT adapter suite (33), predictive-framework suite (21), and TabICL-adapter suite (17) passed. No research profile or project data was run. |
| Runner/config integration (Stage 2C) | 2026-10-26 | AI assistant using Copilot SDK in VS Code | Not yet human-reviewed | Added shared dependency-light FT parameter defaults/validation, config registration, lazy runner construction, task-trained validation policy, package metadata, per-domain unseen diagnostics, and exact dependency pins. Focused suites passed: FT runner/config 8, FT adapter 33, predictive framework 21, TabICL adapter 17. Full repository unittest suite: 170 passed in 145.187 s, no skips. `pip check` reported no broken requirements; environment YAML and all requirement lines parsed, and both manifests matched the FT pin while preserving the PyTorch pin. No project data/research run. Stage 3 feasibility remains. |
| Local feasibility | — | — | — | — |
| Final configuration and documentation | — | — | — | — |

Record actual contributors and reviewers when work is completed.

For each experiment or validation run, record the command, configuration,
output directory, outcome, and relevant limitations.

## 18. Stage 1 completion gate

Stage 1B has resolved the implementation/dependency choice, integration
surface, numerical and categorical preprocessing, initial architecture and
training profile, validation/checkpoint policy, artifact format, numerical
tolerances, and bounded feasibility workflow. Stage 2 can begin after
implementation review. Full-validation runtime/memory, profile convergence,
and domain-specific feasibility remain Stage 3 measurements, not design
blockers to starting implementation. If those measurements show the frozen
profile impractical, revise it explicitly before research evaluation.

## 19. Stage 1A verified evidence

Stage 1A inspected the installed implementation and repository contracts and
ran only an in-memory synthetic compatibility diagnostic. No training,
validation, or test CSV was loaded; no research experiment, package
installation, or dependency change was performed. The prepared-data schema
metadata was read to establish the ordered feature schema.

### 19.1 Installed implementation and dependency facts

| Fact | Verified evidence |
|---|---|
| Distribution/version | `rtdl_revisiting_models` / `rtdl-revisiting-models` 0.0.2 |
| Installed module | `/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/lib/python3.11/site-packages/rtdl_revisiting_models.py` |
| Installed module SHA-256 | `ca7cef3b44cd54b1aeb51f2cc733bef01fbfe3eff00b5c2bd28ca4b4db6b01b2` |
| Matched upstream source | SHA-256 identical to `package/rtdl_revisiting_models.py` at official commit `e3ed46cac38568785289d8fa16b8cfa585bde27e` |
| Upstream release tag | No tags were returned by the official repository's tags endpoint; the installed wheel metadata does not itself record a VCS commit |
| Declared runtime dependencies | `torch>=1.8,<3`; no other `Requires-Dist` entries |
| Python requirement | `>=3.8` |
| License evidence | Installed `rtdl_revisiting_models-0.0.2.dist-info/LICENSE` contains Apache License, Version 2.0; wheel metadata also has the Apache Software License classifier but no `License` field |
| Project manifest status at Stage 1A | `environment.yml` and `requirements-local.txt` pinned PyTorch but did not declare `rtdl-revisiting-models`; Stage 1B chooses to add the exact 0.0.2 pin to both in Stage 2 |
| Runtime used | `/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python`, Python 3.11, PyTorch 2.14.1+cu126, CUDA available with one device |

The official repository identifies this as the officially recommended
implementation for the paper. The local installed source was verified
directly; the upstream commit is included as a source fingerprint match,
not inferred from the package version alone.

### 19.2 Installed API and tensor contract

Installed signatures:

```python
FTTransformer(
    *,
    n_cont_features: int,
    cat_cardinalities: List[int],
    _is_default: bool = False,
    **backbone_kwargs,
) -> None

FTTransformer.forward(
    self,
    x_cont: Optional[torch.Tensor],
    x_cat: Optional[torch.Tensor],
) -> torch.Tensor
```

Verified behavior from the installed 0.0.2 source:

- Pass `d_out=1` as a backbone keyword for one raw output logit per row.
  The output shape is `(batch_size, 1)`; the module does not apply sigmoid.
  The project's binary-label mapping makes that logit the score for label 1,
  and the adapter must convert it to a one-dimensional `P(y=1)` vector.
- `x_cont` has shape `(batch_size, n_cont_features)` and is linearly
  tokenized using one learned weight and bias vector per numerical column.
  `x_cat` has shape `(batch_size, n_cat_features)` and is tokenized using one
  separate `nn.Embedding` table per categorical column plus a learned
  per-feature bias. These are not one concatenated embedding table and the
  package does not use global category-index offsets.
- Each categorical cardinality must be positive and the number of supplied
  cardinalities must equal the categorical tensor's final dimension.
  Feature `i` must be encoded with indices `0..cat_cardinalities[i]-1`.
  Values outside that range reach the underlying embedding lookup and fail;
  no unknown-category index or fallback is implemented by the library.
- The example uses a `torch.int64` categorical tensor. Categorical values
  must be valid integer embedding indices. Numerical inputs must use a
  floating dtype compatible with the model parameters.
- Numerical-only input is supported with `cat_cardinalities=[]` and
  `x_cat=None`. Categorical-only input is supported with
  `n_cont_features=0` and `x_cont=None`. At least one feature kind must be
  present and its corresponding tensor must be supplied. Omitting a tensor
  for a configured feature type is rejected.
- Categorical embedding weights, their feature biases, numerical embedding
  weights/biases, and the CLS vector are initialized uniformly in
  `[-1/sqrt(d_block), +1/sqrt(d_block)]`. Other module parameters use their
  underlying PyTorch module initialization. Dropout is provided in attention,
  feed-forward, and residual branches and follows ordinary `Module.train()`
  / `Module.eval()` behavior.
- Model and inputs must be on compatible devices. The package does not
  provide project preprocessing, category mapping, or high-level input
  validation; several invalid shapes, dtypes, devices, and indices fail in
  lower-level PyTorch operations.
- The module is a normal `torch.nn.Module`. The package exposes no
  model-specific save/load or preprocessing-persistence format. The adapter
  must persist its state and all fitted preprocessing/schema information.

Configuration and optimizer helpers:

- `FTTransformer.get_default_kwargs(n_blocks=3)` returns default backbone
  kwargs. With three blocks, verified defaults are `d_block=192`,
  `attention_n_heads=8`, attention dropout `0.2`, FFN hidden multiplier
  `4/3`, FFN dropout `0.1`, and residual dropout `0.0`.
- `model.make_parameter_groups()` separates parameters excluded from weight
  decay (feature/CLS embeddings, normalization parameters, and biases) from
  the remaining parameters. It can be passed to a PyTorch optimizer.
- `model.make_default_optimizer()` creates AdamW with `lr=1e-4` and
  `weight_decay=1e-5`. It warns when the model was not created using the
  package's default configuration. These are helpers, not mandatory
  optimizer settings.

### 19.3 Official preprocessing and training guidance

The [official package README](https://github.com/yandex-research/rtdl-revisiting-models/blob/e3ed46cac38568785289d8fa16b8cfa585bde27e/package/README.md)
and
[official end-to-end example notebook](https://github.com/yandex-research/rtdl-revisiting-models/blob/e3ed46cac38568785289d8fa16b8cfa585bde27e/package/example.ipynb)
were inspected at commit `e3ed46cac38568785289d8fa16b8cfa585bde27e`.
The README explicitly says the choice of preprocessing depends on the task
and model. It gives StandardScaler as a simple train-fitted alternative and
shows a more elaborate train-fitted QuantileTransformer recipe:

- Add seeded Gaussian noise with standard deviation `1e-5` to the training
  numerical array only for fitting the transformer.
- Set `n_quantiles=max(min(n_train // 30, 1000), 10)`,
  `output_distribution="normal"`, and `subsample=10**9`.
- Fit using the training numerical rows, then transform train, validation,
  and test using that fitted transformer. The added noise is not used when
  transforming the partitions.

This is upstream example guidance, not an FT-Transformer library
requirement. Stage 1B selects the simpler train-fitted StandardScaler for the
reasons recorded in §7.2; the example's synthetic noise and quantile tie
behavior are not adopted.

The notebook demonstrates the package's default architecture and
`make_default_optimizer()`; it also shows AdamW using
`make_parameter_groups()` with `lr=1e-4`, `weight_decay=1e-5`. It uses
`binary_cross_entropy_with_logits` for a binary one-logit task, shuffled
minibatches, training mode for updates, evaluation mode under no-grad, and
full validation evaluation each epoch. Example settings include batch size
256, patience 16, and a very high epoch cap. The notebook says its
hyperparameters are not tuned; none of these values are binding project
requirements.

The notebook also evaluates its test partition every epoch and records the
best validation score without restoring saved best weights. Both behaviors
are unsuitable as project acceptance behavior: this project must not consult
test data during training/selection and must restore the selected checkpoint.

### 19.4 Existing project integration contracts

| Contract | Existing behavior and FT-Transformer implication |
|---|---|
| Adapter lifecycle | `ModelAdapter` requires `fit(X_train, y_train, X_val, y_val)`, `predict_proba(X)`, `save(Path)`, and `metadata()`; `artifact_name` names the saved model. Fit receives validation even though pretrained adapters intentionally ignore it. |
| Input and probability contract | The base class requires a nonempty DataFrame with exact ordered feature names, rejects missing/non-finite values, copies inputs, converts categorical values to strings, and checks numeric types. `predict_proba` must return one float64 value per input row, the positive-class probability, in row order and within `[0,1]`. For FT-Transformer, sigmoid must be applied after raw one-logit inference. |
| Validation data | `runner.run` loads the complete same-domain validation split with `load_partition` and passes `val.X`, `val.y` to `fit`; it does not sample validation. The FT adapter can evaluate the full validation population at each declared validation point without changes to runner data loading. |
| Feature types/order | `data.load_feature_schema(config.feature_set)` derives ordered features and categorical/numeric lists from prepared schema metadata. For `strict_common_16`, the ordered features are `age`, `alcohol_consumption_per_week`, `bmi`, `cardiovascular_history`, `cholesterol_total`, `diastolic_bp`, `ethnicity`, `family_history_diabetes`, `gender`, `hdl_cholesterol`, `heart_rate`, `hypertension_history`, `physical_activity_minutes_per_week`, `sleep_hours_per_day`, `smoking_status`, `systolic_bp`. Categorical order: `ethnicity`, `gender`, `smoking_status`. Numeric order: `age`, `alcohol_consumption_per_week`, `bmi`, `cardiovascular_history`, `cholesterol_total`, `diastolic_bp`, `family_history_diabetes`, `hdl_cholesterol`, `heart_rate`, `hypertension_history`, `physical_activity_minutes_per_week`, `sleep_hours_per_day`, `systolic_bp`. Emit numerical and categorical tensors in these respective orders; these lists were obtained from schema metadata only, no row data was loaded. |
| Matched identities/order | Runner loads the requested training split, then `select_training` sorts by identity, uses the existing seed-controlled proportional class allocation and without-replacement sampling, and returns selected rows in ascending identity order. The seed-42 n2000 configs therefore can reuse the exact existing selection path. `sampling_metadata` records identity and ordered-identity SHA-256; `training_sample.csv.gz` records each selected row's order and identity. |
| Model/config registration | Stage 1A found `ExperimentConfig` allowed only `catboost`, `tabpfn`, and `tabicl`, with model-specific allowlists. Stage 2C registers `ft_transformer` using a dependency-light shared validator for the frozen architecture/training fields in §15. Keep top-level `early_stopping_rounds=null`; patience/max epochs/minimum improvement control FT training. |
| Lazy optional imports | Runner imports TabPFN and TabICL adapters only within their model branches. Those adapters defer third-party imports to runtime helper methods. FT-Transformer and torch imports should likewise remain in the FT branch/runtime so unrelated models and reporting do not require this optional adapter. |
| Timing | Runner synchronizes around `fit` and each test-domain prediction, recording `fit_duration_seconds` and per-domain `prediction_duration_seconds`. Fit timing includes all training and any validation/checkpoint passes performed inside `fit`; it does not separately time per-epoch validation. Model saving occurs after the fit timer. Stage 3 must separately measure training and full-validation costs if those are needed for interpretation. |
| Artifacts and status | Runner creates a fresh output directory with `exist_ok=False`, writes running/config metadata before work, saves the adapter artifact after fit, writes completed status only at success, and records failure type/message then re-raises. Existing results cannot be overwritten by reusing a run directory. There is no shared model loader in `ModelAdapter`; persistence is adapter-specific. |
| Metadata extension | `adapter.metadata()` is stored under `metadata["model"]`; runner also exposes preprocessing and compute settings at top level. Store training history, best epoch, stopping reason, validation population, requested/effective settings, and artifact details in this nested metadata. After each FT domain prediction, refresh nested FT metadata and record that domain's unseen counts in run metadata. Keep values JSON-finite (`write_json` rejects NaN/Infinity). |
| Reusable tests/utilities | `tests/test_predictive_framework.py` covers config validation, adapter contract, run lifecycle, failure status, and overwrite prevention. `tests/test_tabpfn_sampling.py` covers deterministic matched sampling/provenance. `tests/test_tabicl_adapter.py` supplies patterns for lazy runtime checks, preprocessing and persistence. Existing smoke workflows under `src/experiments/` and `tests/` provide examples; they are not a generic FT training adapter. |

Stage 2B label alignment is explicit: a pandas `Series` label index must
exactly equal its predictor frame's index and order; array-like labels align
positionally and are never reordered. Both training and validation must
contain labels 0 and 1.

Stage 2C completed the FT branch in config validation, lazy adapter
construction, model-specific environment metadata, and task-trained
validation-selection policy in the runner. The adapter supplies fitted
metadata; the runner adds per-domain unseen-category diagnostics outside the
timed prediction call and refreshes model metadata after saving and
prediction. `src/models/base.py`, sampling, results writing, metrics, and
shared result schemas remain unchanged. Future reporting manifests and
display registries are explicitly deferred until results exist, consistent
with the existing M10/reporting boundary.

### 19.5 Synthetic compatibility diagnostic

The following short diagnostic was run in the existing
`synthetic-tabular-ai` environment. It uses no project data, downloads,
compilation, mixed precision, or persistent test file:

```python
import torch
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer

seed = 42
x_cont = torch.tensor([
    [-1.0, 0.5], [-0.5, -1.0], [0.0, 0.25], [0.5, 1.0],
    [1.0, -0.5], [1.5, 0.75], [-1.5, -0.25], [0.25, 1.5],
], dtype=torch.float32)
x_cat = torch.tensor([
    [0, 0], [1, 1], [2, 2], [0, 3],
    [1, 0], [2, 1], [0, 2], [1, 3],
], dtype=torch.int64)
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)

for device_name in ("cpu", "cuda:0"):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        continue
    torch.manual_seed(seed)
    if device_name.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model = FTTransformer(
        n_cont_features=2, cat_cardinalities=[3, 4], d_out=1,
        n_blocks=1, d_block=16, attention_n_heads=4,
        attention_dropout=0.1, ffn_d_hidden=8,
        ffn_d_hidden_multiplier=None, ffn_dropout=0.1,
        ffn_activation="ReGLU", residual_dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.make_parameter_groups(), lr=1e-3, weight_decay=1e-5,
    )
    x_num, x_cat_device, labels = (
        x_cont.to(device), x_cat.to(device), y.to(device)
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x_num, x_cat_device)
    assert logits.shape == (8, 1)
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )
    model.eval()
    with torch.no_grad():
        probabilities = model(x_num, x_cat_device).squeeze(-1).sigmoid()
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
```

Observed results (one local run; not a performance benchmark):

| Device | Result |
|---|---|
| CPU | PASS; logits `(8, 1)`, finite BCE-with-logits loss `0.71325243`, 24 finite gradient tensors, at least one parameter changed, finite probability range `[0.57473230, 0.59373069]` |
| CUDA `cuda:0` | PASS; logits `(8, 1)`, finite BCE-with-logits loss `0.70754665`, 24 finite gradient tensors, at least one parameter changed, finite probability range `[0.57456654, 0.59364313]` |

This confirms only that this small configuration executes a forward pass,
backward pass, optimizer step, and evaluation probability conversion on
these devices in the installed environment. It does not establish
convergence, full reproducibility, resource feasibility, or research
readiness.

### 19.6 Stage 2 implementation file map and deferred work

| Location | Stage 2C outcome or explicitly deferred work |
|---|---|
| `src/models/ft_transformer_config.py` | Added dependency-light shared FT defaults/validation so config and adapter use one parameter contract without importing Torch |
| `src/experiments/config.py` | Registers `ft_transformer`, validates supported parameters, preserves `early_stopping_rounds=null`, and keeps same-domain validation |
| `src/experiments/runner.py` | Adds lazy adapter construction, task-trained validation-selection/environment metadata, adapter-provided fitted metadata, and per-domain unseen counts outside prediction timing |
| `src/models/ft_transformer.py` | Existing Stage 2A/2B adapter selected by the runner; metadata now supplies validation and compute settings expected by the framework |
| `requirements-local.txt`, `environment.yml` | Added exact `rtdl-revisiting-models==0.0.2` pin; preserved `torch==2.14.1+cu126` |
| `tests/test_ft_transformer_adapter.py` | Retains preprocessing, training, checkpoint, prediction, persistence, and review-regression tests |
| `tests/test_ft_transformer_runner.py` | Added config validation/serialization, blocked optional-import checks, and synthetic real-adapter success/failure lifecycle tests |
| `tests/test_predictive_framework.py`, `tests/test_tabicl_adapter.py` | Unmodified; both run as framework/pretrained regression coverage |
| Research configs and `src/experiments/smoke_ft_transformer.py` | Not added in Stage 2C; remain future Stage 3/4 work and no smoke command was introduced |

No `src/models/base.py`, shared metrics, sampling, artifact lifecycle, or
result schema change is currently justified. When M10 results are prepared
for reporting, separately inspect the hard-coded model/profile lists in
`src/reporting/validation.py`, `src/reporting/tables.py`, and
`src/reporting/figures.py`; reporting support is not a Stage 1A or adapter
acceptance requirement.

### 19.7 Evidence status and remaining gates

The installed-library/API facts and tiny CPU/CUDA compatibility
measurements in §19 are Stage 1A observations. The frozen policies in
§§6–15 and §20 are Stage 1B design decisions; the adapter and synthetic
compatibility/runner checks now verify their covered behavior, but selecting
the profile does not establish research convergence or feasibility.

Stage 2A, Stage 2B (including review corrections), and Stage 2C have
implementation and focused-test evidence recorded above. The complete
repository unittest suite also passed. No actual project data split or
research-profile run was loaded or launched. Stage 3 full-validation cost,
resource estimates, maximum-profile behavior, and feasibility on both
synthetic training domains remain unmeasured. Research configs, final
documentation/checklist work, and M09 acceptance remain incomplete; do not
mark M09 complete based on these integration checks.

### 19.8 Sources and review evidence

Official sources inspected:

- [Package README at verified upstream commit](https://github.com/yandex-research/rtdl-revisiting-models/blob/e3ed46cac38568785289d8fa16b8cfa585bde27e/package/README.md)
- [Installed module source at verified upstream commit](https://github.com/yandex-research/rtdl-revisiting-models/blob/e3ed46cac38568785289d8fa16b8cfa585bde27e/package/rtdl_revisiting_models.py)
- [Official end-to-end example notebook at verified upstream commit](https://github.com/yandex-research/rtdl-revisiting-models/blob/e3ed46cac38568785289d8fa16b8cfa585bde27e/package/example.ipynb)
- [PyPI release metadata for 0.0.2](https://pypi.org/pypi/rtdl-revisiting-models/0.0.2/json), including declared requirements and upload metadata
- [Official paper, arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- [Upstream commit record](https://github.com/yandex-research/rtdl-revisiting-models/commit/e3ed46cac38568785289d8fa16b8cfa585bde27e)

Project evidence inspected: `src/models/base.py`, the CatBoost, TabPFN,
and TabICL adapters, experiment config/runner/sampling/results/metrics,
data schema implementation and schema metadata, matched n2000 configs, and
predictive framework/adapter/sampling/smoke tests. No repository data split
was loaded. Stage 1A evidence and Stage 1B design were prepared by the AI
assistant recorded above; no human review has occurred.

## 20. Stage 1B decision summary and implementation readiness

| Decision | Frozen initial choice |
|---|---|
| Implementation/dependency | `rtdl-revisiting-models==0.0.2`; exact pin is present in both existing manifests; preserve the PyTorch pin |
| Numeric transform | Train-only StandardScaler, preserve linear extrapolation; constant scale 1; float32 model tensors |
| Categorical vocabulary/unknowns | Sorted train-only string vocabulary; unseen values alias the training mode embedding, with lexicographic mode tie-break; no untrained reserved embedding |
| Model/training profile | Three blocks, width 192, eight heads, ReGLU hidden width 256; dropout 0.2/0.1/0.0; AdamW parameter groups, `lr=1e-4`, `weight_decay=1e-5`; train batch 256, max 200 epochs |
| Selection | Full same-domain validation each epoch, batch 1024; highest ROC-AUC checkpoint, earliest exact tie; patience 20 epochs and `min_delta=0.001`; validation BCE finite-check |
| Runtime/output | float32, `cuda:0`, no fallback; seed 42; prediction batch 256; same recipe for both domains |
| Persistence | Atomic single-file `model.ft_transformer.pt`, versioned tensor state plus primitive preprocessing/training metadata; strict package/PyTorch/schema/state checks; same-device reload only |
| Framework scope | FT-specific config validation, lazy runner factory/environment/selection handling and per-domain unseen counts are implemented; preserve base adapter, sampler, metrics, and result lifecycle |
| Stage 3 bounds | Smoke: 64 train / 32 validation rows / 3 epochs; resource estimate: matched n2000 plus full validation / 2 initial epochs, extend only as needed to at most 10; research-profile feasibility: both domains, full validation, maximum 200 epochs |

Stage 2C integration and its synthetic checks are complete with no known
dependency, API, or project-contract blocker to Stage 3. The remaining
unknowns are empirical acceptance results, not unresolved design choices:
full-validation time and memory, full-profile training duration/stop
behavior, and feasibility on both domains. Measure them in Stage 3 using
the workflow in §14. Do not respond to a failure with an automatic fallback;
revise the declared profile before any research evaluation if the chosen
settings prove impractical.

This document is ready for Stage 3 local-feasibility review, but has not
received human review. M09 remains incomplete.