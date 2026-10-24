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
(Source and Kaggle are **both synthetic**) and synthetic-to-real shift to NHANES.
CatBoost is the reference adapter. Local pretrained TabPFN classification is also
supported; TabICL and FT-Transformer are not implemented yet.

The first configuration fits **one** CatBoost model with seed 42 on Source
`train`, using Source `val` exclusively for early stopping and best-model
selection. That same model predicts Source, Kaggle, and NHANES `test` using the
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
training rows**, sampling seed 42, and the complete fixed test splits (Source
15,000; Kaggle 105,000; NHANES 646). Validation and test sets are not sampled in
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

The TabPFN defaults for the RTX 4060 Laptop/8 GB starting point are explicit
`device: "cuda:0"`, `n_estimators: 4`, `fit_mode: "fit_preprocessors"`,
`inference_precision: "autocast"`, and `memory_saving_mode: true`.
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

`prediction_batch_size: 256` bounds sequential query chunks, including a final
partial chunk, without refitting or changing the training context. It does not
reduce test-set size. Omitted/null batching preserves the existing unbatched
behavior. Installed v3.5 code uses training-only scaling/ECDF statistics and
train-key attention; fitted preprocessors transform queries rather than fitting
on them. This is **not a claim of numerical chunking equivalence**: batch shape
can affect kernels, precision, and memory heuristics. The optional real-model
smoke check below measures unchunked/chunked differences on validation queries.
No real-model chunk-equivalence result has been asserted during implementation.

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

Run the four experiments independently from the project root. Use the float32
TabPFN configurations for the primary comparison. The original autocast
configurations are retained for diagnostics; the Source autocast smoke test
exceeded the strict chunk-equivalence tolerance, while float32 passed.

```bash
python3 -m src.experiments.runner --config configs/tabpfn_source_n2000_float32.json
python3 -m src.experiments.runner --config configs/tabpfn_kaggle_n2000_float32.json
python3 -m src.experiments.runner --config configs/catboost_source_n2000.json
python3 -m src.experiments.runner --config configs/catboost_kaggle_n2000.json
```

The original full-data CatBoost configurations and completed results are unchanged.
Each new configuration has its own descriptive output directory.

Optional **local real-model smoke check**, not a research experiment:

```bash
python3 -m src.experiments.smoke_tabpfn \
  --config configs/tabpfn_source_n2000.json \
  --output-dir results/smoke/tabpfn_local \
  --sample-size 64 --query-size 33 --batch-size 16
```

This uses Source train/validation only, never loads test partitions, performs no
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

Resource-feasibility check (2026-10-23): the exact Source 2,000-row context
(800 negative / 1,200 positive), first 257 deterministic Source validation
queries, and 256+1 prediction chunks passed with explicit float32 on the RTX
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

Separate float32 research configurations are available; the original autocast
configurations are preserved:

```bash
python3 -m src.experiments.runner --config configs/tabpfn_source_n2000_float32.json
python3 -m src.experiments.runner --config configs/tabpfn_kaggle_n2000_float32.json
```

The Source-trained float32 research run completed successfully, evaluating all
three complete test splits in approximately 16 minutes. Its matched 2,000-row
CatBoost reference also completed, with identical training identities and order.
The Kaggle-trained float32 research run remains pending; Source feasibility
does not establish Kaggle-context feasibility.

### `scripts/diagnostics/inspect_datasets.py`

Performs an initial inspection of the raw source and Kaggle datasets.

The script:

- reports dataset shapes and target distributions;
- identifies features shared between the two datasets;
- identifies source-only and Kaggle-only columns;
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

The project uses the 24 predictors shared by the source and Kaggle datasets. Source-only variables and the Kaggle `id` column are excluded from the normal cross-domain feature space.

The script:

- validates the raw dataset schemas;
- enforces the frozen shared-feature schema;
- validates numeric values and categorical vocabularies;
- checks for missing values and duplicate rows;
- creates canonical source and Kaggle datasets;
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

Measures how easily the source and Kaggle-generated distributions can be distinguished.

The script constructs a balanced binary domain-classification problem:

```text
domain = 0 -> source dataset
domain = 1 -> Kaggle-generated dataset
```

For each train, validation, and test partition, Kaggle examples are deterministically downsampled to match the number of source examples.

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

`sampling_provenance.csv.gz` records the exact source and Kaggle rows used in the balanced domain experiment so later experiments can reproduce the same samples.

---

### `src/analysis/analyze_distribution_shift.py`

Performs a more detailed statistical analysis of the differences between the source and Kaggle distributions.

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

The primary question addressed by this experiment is whether a smaller set of artifact-heavy features accounts for most of the distinguishability between the source and Kaggle-generated datasets.

## Running the Current Analysis Pipeline

With the Conda environment activated and the manually acquired Source/Kaggle raw datasets in their expected locations, run the following from the project root. The NHANES downloader skips existing files.

Both Source/Kaggle and NHANES preparation must finish before running `src.data`: its smoke test validates all three domains and requires NHANES feature metadata and split assignments.

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