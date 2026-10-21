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