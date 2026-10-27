# Data preparation, provenance, and analysis

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

For the usual path, run `bash scripts/initial_setup.sh`: it downloads NHANES,
prepares all three domains, and runs the loader smoke test.
Synthetic raw CSVs are included in Git; NHANES and generated data are not.
Repeating preparation can rewrite generated data and metadata.

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

## Raw data and provenance

A normal clone includes both required synthetic CSVs. The acquisition links below are
provenance and recovery references if a required file is missing:

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

The synthetic CSVs are tracked in Git; setup checks their presence rather than downloading them. Preparation records raw-file hashes in `data/processed/metadata.json`. The official Kaggle competition test file is not the project's held-out evaluation split: fixed train/validation/test partitions are created from the labeled `train.csv`.

Setup downloads NHANES 2015–2016 components through `scripts/download_nhanes_2015_2016.sh`, then harmonizes them and creates fixed splits. Existing downloaded files are skipped.

Original Synthetic/Playground S5E12 preparation retains their 24 shared predictors for two-domain analyses. Primary predictive experiments across all three domains use `strict_common_16`. The harmonized `common_20` and `common_22` feature sets are available for sensitivity studies; their existence does not imply those experiments are completed.

## `scripts/diagnostics/inspect_datasets.py`

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

## `scripts/prepare_data.py`

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

## `src/data.py`

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

## `src/analysis/domain_classifier.py`

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

## `src/analysis/analyze_distribution_shift.py`

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

## `src/analysis/domain_feature_ablation.py`

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

With the Conda environment activated and the bundled Original Synthetic/Playground S5E12 raw datasets present, run the following from the project root. The NHANES downloader skips existing files.

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
