# Synthetic Tabular Data Under Distribution Shift

This project studies how diabetes-prediction models trained on synthetic tabular data generalize across synthetic generators and to real survey data. **Original Synthetic: the 100,000-row Diabetes Health Indicators Dataset published on Kaggle by Mohan Krishna Thalla.** This is a descriptive project label, not an independently verified claim about original authorship or the complete generation history. Original Synthetic and Playground S5E12 are both synthetic; NHANES 2015–2016 is real survey data. CatBoost is the reference model, and pretrained TabPFN provides the current tabular foundation-model comparison.

| Internal ID | Display name | Dataset |
|---|---|---|
| `source` | Original Synthetic | Diabetes Health Indicators Dataset, 100,000 rows |
| `kaggle` | Playground S5E12 | Kaggle Playground Series Season 5 Episode 12 |
| `nhanes` | NHANES 2015–2016 | National Health and Nutrition Examination Survey |

Both synthetic datasets are hosted on Kaggle; the internal ID `kaggle` specifically means Playground S5E12. The downloaded Original Synthetic `diabetes_dataset.csv` exactly matches the local CSV (SHA-256 below).

## Current Status

Status as of 2026-10-25:

| Component | Status |
|---|---|
| Preparation, harmonization, and fixed splits | Completed |
| Distribution-shift analysis, domain classification, and feature ablation | Completed |
| Full-data Original Synthetic and Playground S5E12 CatBoost evaluations | Completed |
| Matched Original Synthetic n2000 CatBoost and float32 TabPFN evaluations | Completed |
| Original Synthetic and Playground S5E12 validation-only TabPFN benchmarks | Completed; batch 1024 selected among tested settings |
| Matched Playground S5E12 n2000 CatBoost and float32 TabPFN batch-1024 evaluations | Completed |
| TabICL and FT-Transformer | Not implemented |

Matched n2000 CatBoost and float32 TabPFN batch-1024 results for both synthetic domains are complete. The Playground S5E12 results are in `results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/` and `results/predictive/tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024/`. The Original Synthetic batch-256 predictive results are complete and remain valid; rerunning them at batch 1024 is optional and only for a runtime comparison. Benchmark measurements are computational results, not additional predictive evaluations.

## Environment Setup

This project uses a Conda environment named `synthetic-tabular-ai` with Python 3.11. The recommended setup method is to recreate the environment from the provided `environment.yml` file.

### Using Conda

From the project root, create the environment with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate synthetic-tabular-ai
```

Verify that the correct Python interpreter is active:

```bash
which python3
python3 --version
```

The Python version should be 3.11.x.

The environment file includes both Conda-managed packages and the pip-installed Python dependencies required by the project, including PyTorch, TabPFN, CatBoost, pandas, SciPy, scikit-learn, and Matplotlib. You should not normally need to run `pip install -r requirements-local.txt` after creating the environment from `environment.yml`.

### Pip Requirements

`requirements-local.txt` contains a snapshot of the pip packages installed in the development environment. It can be used as an alternative or as a reference when reproducing the Python package configuration.

For a pip-based installation:

```bash
python3 -m pip install -r requirements-local.txt
```

The development environment includes a CUDA-enabled build of PyTorch and NVIDIA CUDA dependencies. Systems without a compatible NVIDIA GPU or CUDA environment may require a different PyTorch installation.

### Verify the Environment

After activating the environment, the main dependencies can be checked with:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import tabpfn; print('TabPFN:', tabpfn.__version__)"
python3 -c "import catboost; print('CatBoost:', catboost.__version__)"
```

If using an NVIDIA GPU, PyTorch GPU access can also be checked with:

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

To update the environment snapshot after installing additional packages:

```bash
conda env export --no-builds | grep -v '^prefix:' > environment.yml
python3 -m pip freeze > requirements-local.txt
```

## Initial Project Setup

After creating and activating the Conda environment, the project can be initialized using the setup script in `scripts/`.

First, activate the project environment:

```bash
conda activate synthetic-tabular-ai
```

Then run:

```bash
./scripts/initial_setup.sh
```

If the script is not executable, run:

```bash
chmod +x scripts/initial_setup.sh
./scripts/initial_setup.sh
```

The setup script performs the following checks and initialization steps:

1. Verifies that the active Conda environment is `synthetic-tabular-ai`.
2. Verifies that Python 3.11 is being used.
3. Checks that the required raw datasets are present:
   - `data/raw/source/diabetes_dataset.csv`
   - `data/raw/kaggle_s5e12/train.csv`
4. Downloads and inspects the NHANES 2015-2016 components, then runs `scripts/prepare_nhanes.py` to create the harmonized NHANES datasets and fixed splits.
5. Runs `scripts/prepare_data.py` to:
   - validate the raw datasets;
   - create the canonical shared-feature datasets;
   - generate the fixed train/validation/test splits; and
   - write preparation metadata.
6. Runs `src/data.py` as a smoke test to verify that the prepared datasets, schema, and fixed splits can be loaded correctly.

A successful setup ends with:

```text
========================================
Setup complete.
========================================
```

The setup script does **not** run the research experiments such as the domain classifier, distribution-shift analysis, or feature-ablation experiments. Those should be run separately as needed.

### Acquire the Raw Data

Before running setup, manually place the original, unmodified CSV files at:

| Dataset | Required local file | Acquisition |
|---|---|---|
| Original Synthetic (`source`): Diabetes Health Indicators Dataset (Mohan Krishna Thalla) | `data/raw/source/diabetes_dataset.csv` | Download and extract `diabetes_dataset.csv` from [the verified Kaggle dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset). Check the file hash below. |
| Playground S5E12 (`kaggle`) | `data/raw/kaggle_s5e12/train.csv` | Download `train.csv` from the competition's Data page, completing any required access steps. |

**Verified Original Synthetic provenance (2026-10-25):** the `diabetes_dataset.csv` retrieved
from Kaggle dataset `mohankrishnathalla/diabetes-health-indicators-dataset` is
byte-for-byte identical to the project's local Original Synthetic file. Both
have SHA-256:

```text
72ced87cdffdbc2dd594aa214a11c3561498200980def2958e2610413eb29bef
```

This identifies the exact 100,000-row, 31-column synthetic CSV used here. An
explicit Kaggle version number was not captured; the hash identifies the verified
file contents. To verify a downloaded copy from the project root on Linux/WSL:

```bash
sha256sum data/raw/source/diabetes_dataset.csv
```

The internal ID `source` is displayed as “Original Synthetic.” The display name
does not independently verify authorship or the complete generation history.
Comparing these two synthetic datasets studies synthetic-to-synthetic
distribution shift; NHANES 2015–2016 is the real survey-data evaluation domain.

The related Hugging Face `MaxPrestige/Synthetic-Diabetes-Dataset` has a different
published schema and is not the verified file reference for this project. File
identity with the Kaggle download does not independently establish the complete
generation history or the relationship between all public dataset copies.

Setup does not download these CSVs. Preparation records raw-file hashes in `data/processed/metadata.json`. The official Kaggle competition test file is not the project's held-out evaluation split: fixed train/validation/test partitions are created from the labeled `train.csv`.

Setup downloads NHANES 2015–2016 components through `scripts/download_nhanes_2015_2016.sh`, then harmonizes them and creates fixed splits. Existing downloaded files are skipped.

Original Synthetic/Playground S5E12 preparation retains their 24 shared predictors for two-domain analyses. Primary predictive experiments across all three domains use `strict_common_16`. The harmonized `common_20` and `common_22` feature sets are available for sensitivity studies; their existence does not imply those experiments are completed.

### Expected Setup Order

For a new checkout of the repository:

```bash
conda env create -f environment.yml
conda activate synthetic-tabular-ai
./scripts/initial_setup.sh
```

If the Conda environment already exists, only the final two commands are required:

```bash
conda activate synthetic-tabular-ai
./scripts/initial_setup.sh
```

## Scripts and Workflow

Data preparation lives in `scripts/`, diagnostics in `scripts/diagnostics/`, and distribution-shift analyses in `src/analysis/`. The shared data loader lives in `src/data.py`.

`src/paths.py` is the single source for the project root, data directories, raw dataset locations, and result directories. Update those definitions when changing the directory layout; individual modules derive their output filenames from these shared locations.

Run Python entry points as modules (`python3 -m ...`) from the project root, as shown below. This keeps package imports working without modifying `sys.path` or setting `PYTHONPATH`. Paths to data and results are anchored to `src/paths.py`, not the current working directory. The shell setup and download scripts locate the project root themselves and read dataset locations from the same module; the download script therefore also requires Python.

A typical workflow is:

```text
inspect_datasets.py
        ↓
prepare_data.py
        ↓
data.py
        ↓
domain_classifier.py
        ↓
analyze_distribution_shift.py
        ↓
domain_feature_ablation.py
```

### Shared Predictive Experiments

The predictive framework evaluates robustness under synthetic generator shift
(Original Synthetic and Playground S5E12 are **both synthetic**) and
synthetic-to-real shift to NHANES 2015–2016.
CatBoost is the reference adapter. Local pretrained TabPFN classification is also
supported; TabICL and FT-Transformer are not implemented yet.

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

- `config.json`: fully resolved configuration.
- `metadata.json`: running/completed/failed status, UTC timestamps, schema,
  split/class counts, model/preprocessing/compute details, timings, environment
  versions, SHA-256 prepared-file fingerprints, and namespaced split identities.
- `metrics.json`, `metrics.csv`, and `calibration_bins.csv`: per-domain evaluation.
- `predictions_<domain>_test.csv.gz`: authoritative row indices (and NHANES
  `seqn`) as provenance, labels, positive-class probabilities, and predictions.
- `model.cbm`: the single fitted CatBoost model in native format (TabPFN uses
  the fitted-state artifacts described below).
- `training_sample.csv.gz`: actual training context, with domain, `train` split,
  authoritative identifiers, and zero-based `sample_order` (also emitted for
  full-data runs).

#### Fixed-seed predictive comparison report

The first artifact-based comparison report uses the explicit, versioned
selection in `configs/reports/predictive_n2000_seed42.json`. It validates the
four matched n=2,000 CatBoost/TabPFN runs, their saved identities, labels,
fingerprints, metrics, calibration bins, and evaluation populations across all
four runs, then writes CSV and Markdown tables. `validation.json` distinguishes
input-validation from report-generation status; `manifest.json` records consumed
artifact hashes, reporting package versions, and available Git revision/dirty
state. It reads saved artifacts only: it does not train, load checkpoints, or
run inference. Bootstrap intervals, multi-seed aggregation, and full-data
reference comparisons are not implemented.

From the project root, choose a fresh output directory for each invocation.
The default is tables-only and does not import Matplotlib:

```bash
python3 -m src.reporting.compare_predictive \
  --manifest configs/reports/predictive_n2000_seed42.json \
  --output-dir results/reports/predictive_n2000_seed42_tables
```

Add `--with-figures` to generate descriptive performance, reliability/bin
support, and prediction-runtime charts, each as PNG and PDF:

```bash
python3 -m src.reporting.compare_predictive \
  --manifest configs/reports/predictive_n2000_seed42.json \
  --output-dir results/reports/predictive_n2000_seed42_figures \
  --with-figures
```

The manifest explicitly selects every run; runs are never discovered or chosen
by recency or score. Reports contain `metrics.csv`,
`metric_differences.csv`, `runtime.csv`, and `calibration_bins.csv`, plus
`report.md`, `manifest.json`, and `validation.json`. Figure-enabled reports
include a `figures/` directory with metric, reliability, bin-support, and
runtime plots. The report manifest records figure selection, generated
filenames, Matplotlib version when applicable, consumed artifact hashes,
reporting package versions, and available Git provenance.

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

#### Local TabPFN and matched training budgets

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
Upcoming n2000 research runs use explicit `float32` precision and prediction
batch size 1024, supported by the validation benchmarks below.
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

Use the explicit float32 batch-1024 configurations in the benchmark section below
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

### Validation-only TabPFN inference benchmark

`src.experiments.benchmark_tabpfn` measures prediction throughput and numerical
agreement across batch sizes. It is **not a predictive research result**: it
loads only configured train and same-domain validation partitions through
`src.data`, never requests a test partition, calculates no predictive metrics,
and does not change research configurations or their saved outputs.

Completed Original Synthetic and Playground S5E12 benchmarks each used 2,000
training rows, 2,049 same-domain validation queries, explicit float32, batches
256/512/1024, and three
measured repetitions after one full-query warm-up per case on the RTX 4060 Laptop
GPU. Saved reports are under
`results/benchmarks/tabpfn_source_n2000_float32/` and
`results/benchmarks/tabpfn_kaggle_n2000_float32/`.

| Training/query dataset | Batch 256 median (s) | Batch 1024 median (s) | Measured speedup |
|---|---:|---:|---:|
| Original Synthetic | 21.8232 | 8.3923 | 2.6004x |
| Playground S5E12 | 17.7591 | 6.8744 | 2.5834x |

Median throughput increased from 93.8909 to 244.1513 rows/s for Original
Synthetic and from 115.3775 to 298.0624 rows/s for Playground S5E12. Batch 512
medians were 12.1719 s
(168.3391 rows/s) and 10.6220 s (192.9017 rows/s), respectively.
**Batch 1024 was the fastest numerically eligible tested setting in both domains.**
All measured repetitions passed the unchanged `rtol=1e-5, atol=1e-6` criterion,
with maximum reference probability differences of `1.1920928955078125e-7`
(Original Synthetic) and `1.4901161193847656e-7` (Playground S5E12), and zero threshold disagreements.
For batch 1024, both domains recorded peak allocated memory of 2,346,739,200
bytes and peak reserved memory of 2,982,150,144 bytes (approximately 2.78 GiB
reserved). These are process PyTorch allocator peaks, not total GPU usage.

Based on this computational evidence, use **batch 1024 for upcoming n2000
float32 research runs**, without changing the context, seed, checkpoint,
ensemble, precision, fit mode, threshold, or evaluation settings. These are
inference benchmarks, **not new predictive results**. Fixed case order, laptop
thermal/cache effects, and same-domain validation queries limit interpretation:
neither the same speedup on complete cross-domain evaluation workloads nor
equivalence under different settings or hardware is guaranteed.

The explicit batch-1024 configurations are available. The Playground S5E12
run has completed; the Original Synthetic batch-1024 rerun is optional:

```bash
python3 -m src.experiments.runner --config configs/tabpfn_source_n2000_float32_batch1024.json
```

The completed Playground S5E12 batch-1024 TabPFN output is
`results/predictive/tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024/`;
its matched n2000 CatBoost output is
`results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/`.
Do not rerun either completed configuration into its existing protected output.

With `synthetic-tabular-ai` activated, the completed Original Synthetic benchmark was run
from the project root as follows (use a fresh output name to repeat it):

```bash
python3 -m src.experiments.benchmark_tabpfn \
  --config configs/tabpfn_source_n2000_float32.json \
  --output-dir results/benchmarks/tabpfn_source_n2000_float32 \
  --query-size 2049 \
  --batch-sizes 256 512 1024 \
  --repetitions 3
```

The query size, batch sizes, and repetition count shown are the defaults.
The command requires explicit float32, an explicit CUDA device, a configured
training sample budget and sampling seed, and batch size 256 among unique
positive batch sizes. The local checkpoint must already be available; there
are no downloads, CPU fallbacks, or automatic changes to model settings.
Use a **new directory under `results/benchmarks/`** each time; existing directories
are refused, including interrupted attempts. Research output directories are
never reused.

The context is selected with the existing deterministic training sampler.
Validation queries are sorted by authoritative identity (`row_index` for
Original Synthetic/Playground S5E12, `seqn` for NHANES 2015–2016), then the first `query_size` rows are used.
Selection does not use validation labels. Full train/validation identities must
be disjoint. Training context, query order, feature order, checkpoint, model
seed, ensemble count, precision, fit mode, and memory-saving settings are held
fixed across cases.

Batch 256 always executes first, followed by the other sizes in requested order.
Each case creates one fresh adapter, prepares its context once, executes one
full-query warm-up, and then performs the requested measured repetitions
without refitting. The shared batching implementation preserves query order
and the last partial chunk: 2,049 queries yield 8+1, 4+1, and 2+1 chunks for
batches 256, 512, and 1024 respectively. The adapter is released and unused CUDA
allocations are cleared between cases, not between measured repetitions.

Prediction timing uses a monotonic timer with CUDA synchronization before and
after prediction and peak counters reset before each repetition. It excludes
loading, construction/checkpoint hashing, context preparation, warm-up,
serialization, comparisons, and console output. Context preparation and
warm-up have separate timings. Memory peaks describe the **process's PyTorch
allocator, not total GPU usage**. Fixed case order, cache state, and laptop
thermals may affect timings; small timing differences are not statistically
established speedups.

Every measured vector is checked for exact row count, one dimension, finite
values, and probabilities in `[0, 1]`. The first measured batch-256 vector is
the fixed reference for **all** measured repetitions, including later
batch-256 repetitions. Reports include maximum/mean absolute differences,
`np.allclose(rtol=1e-5, atol=1e-6)`, and threshold disagreements using
`probability >= config.threshold`. Probabilities are not rounded. Matching
classifications alone do not establish probability equivalence; the tolerance
cannot be relaxed through CLI options.

Outputs, saved incrementally with atomic JSON/CSV replacement:

- `benchmark.json`: UTC timestamps, arguments, resolved configuration,
  package/device/checkpoint metadata, sampling/query fingerprints, feature
  order, operation failures, measurements, agreement and eligibility.
- `benchmark_summary.csv`: one row per requested size, including failed or
  unattempted cases; context/warm-up times, median throughput/duration,
  minimum/maximum duration, maximum measured allocator peaks, and speedup
  relative to batch 256's median duration.
- `benchmark_repetitions.csv`: one row per attempted measured repetition,
  with its status, available timing/memory, agreement, and probability path.
- `training_sample.csv.gz` and `validation_queries.csv.gz`: selected row
  provenance. Query provenance contains identifiers and query order, not
  features or labels.
- `probabilities_batch<SIZE>_repetition<N>.npy`: each measured probability
  vector, saved without pickle; load with `np.load(path, allow_pickle=False)`.

Only a case with all requested measured repetitions completed and passing
agreement is numerically eligible. The report identifies the fastest eligible
measured case, but never rewrites a research config or launches an experiment.
OOM reports identify the operation, batch size, repetition (if applicable),
and error, preserving completed measurements without altered-setting retries.
Reference OOM or disagreement stops the benchmark. Candidate OOM may continue
after successful cleanup; candidate disagreement retains its measurements,
excludes the case, and continues. Unexpected errors and KeyboardInterrupt
preserve available diagnostics and propagate; reporting/cleanup failures do
not replace the original error.

Top-level statuses and exit behavior:

- `running`: in progress, or the process ended before final diagnostics.
- `completed`: every requested case completed and passed; exit 0.
- `completed_with_issues`: reference passed but a candidate had OOM or failed
  agreement; **exit 1**, even if another candidate passed.
- `failed`: reference or unexpected execution failure; nonzero exit.

Per-case execution status and numerical eligibility are recorded separately,
with case statuses `completed`, `out_of_memory`, `agreement_failed`, `failed`,
or `not_run` (and `running` while active).

### `scripts/diagnostics/inspect_datasets.py`

Performs an initial inspection of the raw Original Synthetic and Playground S5E12 datasets.

The script:

- reports dataset shapes and target distributions;
- identifies features shared between the two datasets;
- identifies Original Synthetic-only and Playground S5E12-only columns;
- checks missing values and data types;
- separates categorical and numeric features;
- verifies that categorical features use compatible category values;
- compares numeric means and standard deviations; and
- reports standardized mean differences as an initial estimate of distribution shift.

Run with:

```bash
python3 -m scripts.diagnostics.inspect_datasets
```

This script is primarily a diagnostic tool and does not modify the datasets.

---

### `scripts/prepare_data.py`

Creates the canonical datasets used by the experiments.

The project uses the 24 predictors shared by the Original Synthetic and Playground S5E12 datasets. Original Synthetic-only variables and the Playground S5E12 `id` column are excluded from the normal cross-domain feature space.

The script:

- validates the raw dataset schemas;
- enforces the frozen shared-feature schema;
- validates numeric values and categorical vocabularies;
- checks for missing values and duplicate rows;
- creates canonical Original Synthetic and Playground S5E12 datasets;
- creates deterministic stratified train/validation/test splits using seed `42`;
- preserves original row identity through split-assignment files; and
- records dataset metadata and raw-file hashes for reproducibility.

The fixed split proportions are:

```text
70% training
15% validation
15% test
```

Run with:

```bash
python3 -m scripts.prepare_data
```

Generated data is written under:

```text
data/processed/
data/splits/
```

This script should be run before any experiment scripts.

---

### `src/data.py`

Provides the common data-loading interface used by the experiment scripts.

Rather than having each experiment read CSV files independently, `data.py` loads and validates the canonical datasets and their fixed split assignments.

Important functions include:

```python
load_metadata()
load_schema()
load_common(domain)
load_split_assignments(domain)
load_split(domain, split)
load_xy(domain, split)
split_summary(domain, split)
```

The module intentionally performs **no learned preprocessing**. It does not:

```text
encode categorical variables
scale numeric variables
impute values
select features
rebalance classes
```

Any preprocessing required by a particular model should be fitted using that experiment's training data only.

The module can also be executed directly as a smoke test:

```bash
python3 -m src.data
```

This verifies the prepared schema, fixed splits, and feature/target loading.

---

### `src/analysis/domain_classifier.py`

Measures how easily the Original Synthetic and Playground S5E12 distributions can be distinguished.

The script constructs a balanced binary domain-classification problem:

```text
domain = 0 -> Original Synthetic (`source`)
domain = 1 -> Playground S5E12 (`kaggle`)
```

For each train, validation, and test partition, Playground S5E12 examples are deterministically downsampled to match the number of Original Synthetic examples.

Only the 24 shared predictors are provided to the classifier. The following are explicitly excluded from model input:

```text
diagnosed_diabetes
id
row_index
```

A CatBoost classifier is trained using native handling of categorical features and validation-based early stopping.

The experiment reports metrics including:

```text
ROC-AUC
accuracy
log loss
precision
recall
F1
```

It also calculates feature importance to identify predictors that contain the strongest domain-specific signal.

Run with:

```bash
python3 -m src.analysis.domain_classifier
```

Outputs are written to:

```text
results/domain_classifier/
├── domain_classifier.cbm
├── metrics.json
├── feature_importance.csv
├── test_predictions.csv
└── sampling_provenance.csv.gz
```

`sampling_provenance.csv.gz` records the exact Original Synthetic and Playground S5E12 rows used in the balanced domain experiment so later experiments can reproduce the same samples.

---

### `src/analysis/analyze_distribution_shift.py`

Performs a more detailed statistical analysis of the differences between the Original Synthetic and Playground S5E12 distributions.

For numeric features, the analysis includes:

```text
standardized mean difference
standard-deviation ratio
variance distortion
Kolmogorov-Smirnov statistic
Wasserstein distance
normalized Wasserstein distance
distribution quantiles
```

Binary numeric indicators also receive explicit prevalence comparisons.

For categorical features, the analysis includes:

```text
category frequencies
total variation distance
Jensen-Shannon distance/divergence
category-level frequency differences
```

The statistical shift measures are combined with the feature importance produced by `domain_classifier.py`. Spearman rank correlations are also calculated to examine how closely learned domain importance agrees with individual statistical measures of distribution shift.

Run with:

```bash
python3 -m src.analysis.analyze_distribution_shift
```

Outputs are written to:

```text
results/distribution_shift/
├── numeric_shift.csv
├── categorical_shift.csv
├── categorical_frequencies.csv
├── combined_feature_shift.csv
├── metric_correlations.csv
├── metadata.json
└── figures/
```

Statistical calculations use the complete prepared datasets. Plotting may use deterministic subsampling for efficiency.

---

### `src/analysis/domain_feature_ablation.py`

Measures how domain separability changes when the most domain-discriminative features are progressively removed.

The experiment uses three fixed feature-removal rankings:

```text
domain_importance
ks_statistic
variance_distortion
```

For each ranking, the top:

```text
1
3
5
10
```

features are removed and the CatBoost domain classifier is retrained.

To avoid test-set leakage:

- the domain-importance ranking comes from the original domain classifier;
- KS rankings are calculated using only the reconstructed domain-training split; and
- variance-distortion rankings are calculated using only the reconstructed domain-training split.

The exact balanced samples from the original domain-classifier experiment are reconstructed from `sampling_provenance.csv.gz`, allowing feature sets to be compared without introducing additional sampling variation.

The script first reproduces the original 24-feature classifier and verifies that its test ROC-AUC matches the saved baseline before continuing.

Run with:

```bash
python3 -m src.analysis.domain_feature_ablation
```

Outputs are written to:

```text
results/domain_feature_ablation/
├── ablation_results.csv
├── removed_features.csv
├── training_ranking_metrics.csv
├── metadata.json
└── figures/
    ├── auc_vs_removed_features.png
    └── accuracy_vs_removed_features.png
```

The primary question addressed by this experiment is whether a smaller set of artifact-heavy features accounts for most of the distinguishability between the Original Synthetic and Playground S5E12 datasets.

## Running the Current Analysis Pipeline

With the Conda environment activated and the manually acquired Original Synthetic/Playground S5E12 raw datasets in their expected locations, run the following from the project root. The NHANES downloader skips existing files.

Both Original Synthetic/Playground S5E12 and NHANES preparation must finish before running `src.data`: its smoke test validates all three domains and requires NHANES feature metadata and split assignments.

```bash
python3 -m scripts.diagnostics.inspect_datasets
./scripts/download_nhanes_2015_2016.sh
python3 -m scripts.diagnostics.inspect_nhanes
python3 -m scripts.prepare_nhanes
python3 -m scripts.prepare_data
python3 -m src.data
python3 -m src.analysis.domain_classifier
python3 -m src.analysis.analyze_distribution_shift
python3 -m src.analysis.domain_feature_ablation
python3 -m src.analysis.compare_harmonized_domains
```

Alternatively, `./scripts/initial_setup.sh` performs downloading, NHANES inspection, preparation for all three domains, and the loader smoke test. After it completes, run the four `src.analysis` commands above.

`inspect_datasets.py` and `data.py` are primarily diagnostic checks.

The required dependency chain for the current experiments is:

```text
prepare_data.py
    ↓
domain_classifier.py
    ↓
analyze_distribution_shift.py

prepare_data.py
    ↓
domain_classifier.py
    ↓
domain_feature_ablation.py
```