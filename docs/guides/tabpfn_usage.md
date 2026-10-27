# TabPFN usage, numerical checks, and persistence

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

The completed matched comparisons use Original Synthetic batch 256 and
Playground batch 1024. Both use float32. Preserve these historical profiles
when reproducing the saved comparison group.

The adapter is verified against installed **TabPFN 9.2.0**, **PyTorch
2.14.1+cu126**, and Python 3.11. It explicitly uses **TabPFN v3.5**, checkpoint
`tabpfn-v3.5-20260909.safetensors` from `Prior-Labs/tabpfn_3_5`, not the older
v2/v2.5 defaults. Other TabPFN package versions/checkpoint names fail explicitly
until verified. No fine-tuning, AutoTabPFN, tuning, remote API, or target-domain
adaptation is performed.

**Local prerequisite:** manually obtain the licensed checkpoint and complete any
PriorLabs/Hugging Face authorization/license acceptance required to obtain it.
The adapter does not download weights or store credentials. It requires the file
to exist in the installed package's cache (`~/.cache/tabpfn` here, respecting
`TABPFN_MODEL_CACHE_DIR`), or at a local `model_params.model_path` you explicitly
provide. Relative checkpoint paths resolve from the project root. The checkpoint
path and SHA-256 are recorded; the verified file is present on this development
machine. CUDA is checked when the adapter is instantiated; a requested unavailable
CUDA device is an error, never an automatic CPU fallback.

The new configurations use `strict_common_16`, model seed 42, **2,000 stratified
training rows**, sampling seed 42, and the complete fixed test splits (Original
Synthetic 15,000; Playground S5E12 105,000; NHANES 2015–2016 646). Validation and test sets are not sampled in
these experiments. A 2,000-row budget is an initial experiment choice, **not** a
package limit or a guarantee of fitting into 8 GB VRAM.

Sampling uses only `train`: sort by authoritative `row_index` (NHANES `seqn`),
allocate proportional binary class counts with half-up rounding constrained to
retain both classes, sample within each class without replacement with NumPy
`default_rng(sampling_seed)`, and return selected rows in ascending identity
order. This makes rows **and order** identical across models for equal domain,
feature set, sample size, and sampling seed, even if gateway row order or the
model seed changes. Budgets below two, above available rows, or lacking both
classes fail clearly rather than changing the budget. `train_sample_size: null`
(default) preserves the original full training data and ordering; sampled
configurations require an explicit `sampling_seed`. Full training partitions
remain the basis for overlap checks even when only a subset is fitted.
Metadata separately records the original partition, actual context, selected
class counts, algorithm, and ordered/unordered identity fingerprints.

**Validation policies differ intentionally:** matched CatBoost retains the original
hyperparameters and full same-domain validation set for early stopping. TabPFN
fits its pretrained predictor on the selected training context only; validation
X/y are not inspected by the adapter, concatenated into context, used for encoding,
or used for early stopping/calibration/selection. Its `early_stopping_rounds` must
be omitted or null. Matching observed training rows does **not** match total
training compute: TabPFN was pretrained externally and can perform substantial
work during `predict_proba`. Fit/context-preparation time alone is not a
cross-model training-cost comparison. No settings were selected using observed
NHANES test scores.

The original diagnostic configurations for the RTX 4060 Laptop/8 GB use explicit
`device: "cuda:0"`, `n_estimators: 4`, `fit_mode: "fit_preprocessors"`,
`inference_precision: "autocast"`, and `memory_saving_mode: true`.
For new n2000 research runs use explicit `float32` precision and prediction
batch size 1024, supported by the [validation benchmarks](tabpfn_benchmark.md).
Allowed TabPFN model parameters are exactly those five plus a local `model_path`.
The adapter controls the random state, categorical indices, and
`ignore_pretraining_limits: false`; aliases and unsupported knobs are rejected.
The verified API also permits `low_memory`/`fit_with_cache`, explicit `cpu`,
`auto`/`float32` precision, and `auto`/boolean memory saving. These are explicit
configuration changes, not automatic fallback policies.

Categoricals are string-normalized on copies, with explicit categorical indices.
The installed TabPFN implementation fits its internal ordinal encoder on the
training context and reuses it across domains; unseen categories map to `-1`.
There is no independently fitted domain encoder, external one-hot encoding,
scaling, or imputation. TabPFN's checkpoint-specific internal preprocessing
(including internal scaling/transforms) is recorded in metadata. Unexpected
missing/nonfinite canonical inputs are rejected before the package receives them.

`prediction_batch_size` bounds sequential query chunks, including a final
partial chunk, without refitting or changing the training context. It does not
reduce test-set size. Original configurations use 256; the explicit batch-1024
configurations use 1024. Omitted/null batching preserves the existing unbatched
behavior. Installed v3.5 code uses training-only scaling/ECDF statistics and
train-key attention; fitted preprocessors transform queries rather than fitting
on them. This is **not a claim of numerical chunking equivalence**: batch shape
can affect kernels, precision, and memory heuristics. The optional real-model
smoke check below measures unchunked/chunked differences on validation queries.
Measured agreement results for the tested float32 settings are documented below.

CUDA operations are synchronized around fit/prediction timing. Metadata records
resolved devices, ensemble count, inference/fit settings, package versions, and
process PyTorch allocator peaks since the adapter's reset (not total VRAM usage).
On OOM, the run is marked failed with sample/batch/ensemble/device context.
**Recovery is manual:** use a new output directory and explicitly smaller
batch/context/ensemble settings. There is no silent reduction of rows, ensemble,
precision, or retry, and failed/completed directories remain protected.

TabPFN persistence uses the installed `save_fit_state`/`load_from_fit_state` API:
`model.tabpfn_fit` contains the fitted preprocessing/context state, and
`model.adapter.json` records schema, initialization parameters, and checkpoint
dependency. This is **not self-contained**: weights are deliberately omitted by
TabPFN. Keep the exact checkpoint at its recorded resolved path, its hash
unchanged, and the same supported package version.
Load only trusted fitted-state archives, since the native format includes joblib
state. Restore through:

```python
from pathlib import Path
from src.models.tabpfn import TabPFNAdapter

adapter = TabPFNAdapter.load(
    Path("results/predictive/<run>/model.tabpfn_fit"), device="cuda:0"
)
```

Use the explicit float32 batch-1024 configurations in the [benchmark guide](tabpfn_benchmark.md)
for n2000 TabPFN experiments. Original batch-256 and autocast configurations are
retained for reproducibility and diagnostics; the Original Synthetic autocast smoke test
exceeded the strict chunk-equivalence tolerance, while float32 passed. Existing
full-data CatBoost configurations and completed results are unchanged.

Optional **local real-model smoke check**, not a research experiment:

```bash
python3 -m src.experiments.smoke_tabpfn \
  --config configs/tabpfn_source_n2000.json \
  --output-dir results/smoke/tabpfn_local \
  --sample-size 64 --query-size 33 --batch-size 16
```

This uses Original Synthetic (`source`) train/validation only, never loads test partitions, performs no
metric-based selection, and does not download weights. It measures chunked versus
unchunked probabilities, saves/restores the fitted state, checks reload predictions,
and writes differences plus status to `smoke.json` separately from research
outputs. Tolerance is `rtol=1e-5, atol=1e-6`, not bitwise identity. A difference
beyond tolerance is reported as failure for inspection, not automatically worked
around. Use a fresh smoke output directory when repeating. The small check does
not prove that a 2,000-row/full-test run fits in GPU memory.

To diagnose precision-related chunk differences without modifying research
configurations, repeat in a new directory with the explicit smoke-only override:

```bash
python3 -m src.experiments.smoke_tabpfn \
  --config configs/tabpfn_source_n2000.json \
  --output-dir results/smoke/tabpfn_local_float32 \
  --sample-size 64 --query-size 33 --batch-size 16 \
  --inference-precision float32
```

This leaves the tolerance unchanged and records the effective precision and
batch size, maximum/mean probability differences, and threshold disagreements
in the smoke report. A passing small float32 smoke check does not establish
autocast equivalence or change the precision used by research configurations.

Resource-feasibility check (2026-10-23): the exact Original Synthetic
2,000-row context (800 negative / 1,200 positive), first 257 deterministic
Original Synthetic validation queries, and 256+1 prediction chunks passed with explicit float32 on the RTX
4060 Laptop GPU. Four estimators, seed 42, v3.5 checkpoint, and
`fit_preprocessors` were unchanged. The report is
`results/smoke/tabpfn_source_n2000_float32/smoke.json`.
Context preparation took 2.04 s; initial chunked prediction took 4.95 s with
peak process PyTorch allocated/reserved memory of 1.84/2.41 GiB.
Save/reload chunked predictions matched exactly. The separate unchunked
257-query comparison also succeeded, with maximum difference `5.96e-8`,
within the unchanged `rtol=1e-5, atol=1e-6` tolerance. No test data was used.
Timings include synchronized operations but have different warm-up/cache states;
they are not a benchmark comparing chunked and unchunked speed.

The smoke command now records timing and CUDA allocator peaks per operation,
valid probabilities, sampling/query fingerprints, and query identities.
It runs the intended chunked path and save/reload first. An OOM in the separate
unchunked comparison is explicitly recorded without classifying the successful
intended path as failed or retrying with changed settings; other failures stop
the check and preserve available diagnostics.

The Original Synthetic-trained float32 batch-256 TabPFN research run completed
successfully, evaluating all three complete test splits in approximately
16 minutes. Its matched 2,000-row CatBoost reference also completed with
identical training identities and order. The completed Playground S5E12 n2000
CatBoost and float32 TabPFN batch-1024 results are recorded in
`results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/` and
`results/predictive/tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024/`.
The completed Original Synthetic batch-256 predictive run remains valid; its
batch-1024 rerun is optional and solely for runtime comparison. Batch-256 and
autocast configurations are preserved.
