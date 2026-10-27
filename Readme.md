# Synthetic Tabular Data Under Distribution Shift

This project studies how diabetes-prediction models trained on synthetic data
generalize to another synthetic dataset and to real NHANES survey data.
It compares CatBoost with pretrained TabPFN and TabICL models and a task-trained
FT-Transformer.

**New here?** Follow [Quick start](#quick-start), then
[Run your first experiment](#run-your-first-experiment).
For project progress and contributions, see the
[project checklist](PROJECT_CHECKLIST.md).

Last reviewed: October 27, 2026.

## Before you start

The documented development environment uses **Linux/WSL2, Conda, and Python 3.11**.
Run the commands below in a Bash terminal. You also need Git and `curl`, plus
internet access to install dependencies and download NHANES.

The supplied environment is a snapshot of the Linux/WSL CUDA development system.
Its recreation on a clean machine has not yet been independently verified.
It includes Linux-specific packages and `torch==2.14.1+cu126`; neither dependency
snapshot records a custom package-index URL. Installation may need the matching
package source. Treat an unavailable package as an environment-reproduction issue;
do not silently substitute versions when reproducing research results.

Data preparation and the supplied CatBoost configuration run on CPU.
The supplied TabPFN, TabICL, and FT-Transformer research configurations request
an NVIDIA CUDA GPU. GPU experiments were tested on an RTX 4060 Laptop GPU with
8 GB VRAM; this is the tested machine, not a universal minimum requirement.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/ryan-dozier/cap6614_diabetes_prediction.git
cd cap6614_diabetes_prediction
```

If you already have a checkout, open its root directory instead. Your local
directory name may differ. Run all subsequent commands from that root.

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate synthetic-tabular-ai

python3 --version
python3 -m pip check
```

Python should report **3.11.x**. If this environment already exists, activate it;
skip the creation command. The setup script checks both the environment name
and Python version.

`environment.yml` includes the project's Python dependencies. You do not need
to install `requirements-local.txt` afterward. That file is an additional
development-environment snapshot, not a separately verified portable installation
recipe. See [environment details](docs/guides/environment_setup.md) for reference.

### 3. Prepare the datasets

**The two synthetic raw datasets are already included in a normal Git clone.**
No separate Kaggle download or Kaggle account is needed for the files included
in this checkout.

```bash
bash scripts/initial_setup.sh
```

The script:

1. Checks the active Conda environment, Python version, and bundled synthetic CSVs.
2. Downloads the NHANES 2015–2016 components, skipping files already present.
3. Inspects and harmonizes NHANES and creates its fixed data splits.
4. Prepares the two synthetic datasets and their fixed data splits.
5. Runs the shared data-loader smoke test.

A successful run ends with:

```text
========================================
Setup complete.
========================================
```

This completes data preparation. Research experiments are run separately below.
Setup does not acquire pretrained model checkpoints. Repeating setup reruns
preparation and can rewrite generated data and metadata; it is not a resume
mechanism for experiments.

### What a clone includes

| Resource | Included in Git? | How it becomes available |
|---|---|---|
| Original Synthetic raw CSV | Yes | Present under `data/raw/source/` |
| Playground S5E12 raw CSVs | Yes | Present under `data/raw/kaggle_s5e12/` |
| NHANES raw components | No | Initial setup downloads them |
| Prepared datasets and split assignments | No | Initial setup generates them |
| Experiment configurations and code | Yes | Present after cloning |
| Pretrained model checkpoints | No | Follow model-specific setup |
| Saved predictions, trained models, reports, and figures | No | Run the relevant experiments and reporting commands |

The repository also contains BRFSS raw files. They are not required by the
current three-domain setup and predictive workflow.

## Run your first experiment

Start with the **CPU CatBoost matched-budget reference**. It requires no pretrained
checkpoint and uses 2,000 training rows:

```bash
python3 -m src.experiments.runner --config configs/catboost_source_n2000.json
```

This trains one model on Original Synthetic, uses its complete same-domain
validation split for early stopping, and evaluates that model on the fixed test
splits of all three domains. It uses the primary `strict_common_16` feature set,
seed 42, and a classification threshold of 0.5.

The terminal prints ROC-AUC, average precision, F1, Brier score, and an output path.
Average precision is the precision–recall summary used here, not trapezoidal PR area.
Find the saved artifacts at:

```text
results/predictive/catboost_source_strict_common_16_n2000_seed42/
```

| Artifact | Purpose |
|---|---|
| `config.json` | Saved experiment settings |
| `metadata.json` | Completion status, provenance, timing, and environment |
| `metrics.json` / `metrics.csv` | Evaluation results |
| `predictions_*_test.csv.gz` | Per-row predictions for each test domain |
| `calibration_bins.csv` | Reliability-bin statistics |
| `training_sample.csv.gz` | Selected training identities and order |
| Model-specific artifact | Trained model or fitted adapter state |

Check that `metadata.json` reports `"status": "completed"` before using a run in
a comparison.

**Experiment directories are protected against reuse.** If the configured output
directory already exists, inspect it. To run another experiment, copy the JSON
configuration and give it a distinct `experiment_name` and `output_dir`.
Preserve completed or failed results for inspection. The runner accepts
`--config`; it does not provide a command-line output-directory override. Predictive
commands retrain; there is no checkpoint-reuse or resume mode.

## Choose the next workflow

| Goal | Where to start | Additional requirements |
|---|---|---|
| Run the Playground-trained CatBoost reference | `configs/catboost_kaggle_n2000.json` | Prepared data |
| Run FT-Transformer | `configs/ft_transformer_source_n2000.json` and `configs/ft_transformer_kaggle_n2000.json` | CUDA GPU; installed `rtdl-revisiting-models==0.0.2`; no pretrained weights |
| Run TabPFN | [TabPFN setup and experiments](docs/guides/tabpfn_usage.md) | Verified package, local licensed checkpoint, and CUDA GPU |
| Run TabICL | [Pinned checkpoint setup](docs/guides/tabicl_usage.md#pinned-checkpoint-setup) | Download and verify the pinned checkpoint; CUDA GPU |
| Analyze distribution shift | [Analysis commands](docs/guides/data_and_analysis.md#running-the-current-analysis-pipeline) | Prepared data |
| Compare all four models | [Reporting](#generate-comparison-reports) | All eight selected experiment outputs |

Run GPU experiments sequentially on the local development machine. The supplied
research configurations use explicit devices and precision settings; they do not
automatically switch to CPU if CUDA is unavailable.

TabPFN checkpoint access may require authorization/license acceptance.
The TabICL instructions provide a pinned public checkpoint download and hash
verification. Neither experiment adapter downloads weights during a research run.
FT-Transformer starts from initialized weights. Setup for one model does not
complete checkpoint setup for the others.

## Generate comparison reports

Reporting reads **saved experiment outputs**. Running initial setup alone is
insufficient, because results and checkpoints are not included in the repository.

The four-model manifest explicitly selects two training domains for each of
CatBoost, TabPFN, TabICL, and FT-Transformer. All eight completed run directories
must be present at the manifest's configured paths:

- [Eight-run manifest](configs/reports/predictive_n2000_seed42_with_ft_transformer.json)
- [Earlier six-run manifest](configs/reports/predictive_n2000_seed42_with_tabicl.json)
- [Earlier four-run manifest](configs/reports/predictive_n2000_seed42.json)

Once the selected runs are available:

```bash
python3 -m src.reporting.compare_predictive \
  --manifest configs/reports/predictive_n2000_seed42_with_ft_transformer.json \
  --output-dir results/reports/four_model_comparison_local \
  --with-figures
```

Use a new output directory each time. Reporting validates the saved configurations,
training samples, evaluation identities/labels, and prediction-derived metrics
before completing the report. It does not train models or perform inference.

Successful four-model reporting produces `report.md`, validation/provenance
records, 24 metric rows, 18 candidate-minus-CatBoost difference rows, eight runtime
rows, 240 calibration-bin rows, and six PNG/PDF figure pairs.

Links elsewhere in the project to `results/` refer to ignored local artifacts.
They will not resolve in a fresh clone until those outputs are generated or
obtained separately.

## Data and experiment conventions

| Internal ID | Display name | Role and location |
|---|---|---|
| `source` | Original Synthetic | 100,000-row synthetic Diabetes Health Indicators Dataset; `data/raw/source/diabetes_dataset.csv` |
| `kaggle` | Playground S5E12 | Synthetic competition data; `data/raw/kaggle_s5e12/train.csv` |
| `nhanes` | NHANES 2015–2016 | Real survey data; downloaded into `data/raw/nhanes_2015_2016/` |

Both synthetic datasets are hosted on Kaggle; the internal ID `kaggle` means
Playground S5E12 specifically. The first comparison studies synthetic-to-synthetic
distribution shift; NHANES supplies the synthetic-to-real evaluation domain.

The Original Synthetic CSV was verified against the
[Mohan Krishna Thalla Kaggle dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset).
The descriptive project label does not establish the complete generation history.
For provenance and replacement-file acquisition, see
[the existing data reference](docs/guides/data_and_analysis.md#raw-data-and-provenance). Those acquisition instructions are for restoring missing raw CSVs; a normal clone
already includes them.

Verify Original Synthetic on Linux/WSL with:

```bash
sha256sum data/raw/source/diabetes_dataset.csv
```

Expected SHA-256:

```text
72ced87cdffdbc2dd594aa214a11c3561498200980def2958e2610413eb29bef
```

The labeled Playground `train.csv` is divided into the project's fixed
train/validation/test splits. The competition's unlabeled `test.csv` is not the
held-out evaluation population used in these experiments.

Primary three-domain experiments use `strict_common_16`. Preparation also exposes
24 shared synthetic-domain predictors for analysis and the harmonized
`common_20` and `common_22` feature sets for sensitivity studies. Available feature
sets do not imply completed experiments.

Matched n2000 configurations select the same training identities and order within
each training domain. Splits are deterministic and stratified, with seed 42 and proportions 70% training,
15% validation, and 15% test. The shared loader performs no learned preprocessing;
model-specific transforms fit on training data only. Identifiers are excluded from
predictors. Validation/test partitions are kept separate. CatBoost and
FT-Transformer use full same-domain validation labels for selection; TabPFN and
TabICL use training context only. Matching training rows does not match total
label access, external pretraining, or compute.

Results are descriptive single-seed findings. Average precision depends on
prevalence; Brier measures overall probability error and is not a pure calibration
measure. Current NHANES results describe the held-out survey sample, not
survey-weighted national-population estimates.

## Current research status

As of **October 27, 2026**, local preparation, distribution-shift analyses, and
the initial matched-budget predictive comparisons are complete for all four models.
The eight-run comparison-report workflow is also implemented, validated, and
merged.

| Milestone | Evidence |
|---|---|
| FT-Transformer implementation and feasibility (M09) | [PR #7](https://github.com/ryan-dozier/cap6614_diabetes_prediction/pull/7) |
| FT-Transformer predictive comparisons (M10) | [Results review](docs/results/ft_transformer_predictive_results.md); [PR #8](https://github.com/ryan-dozier/cap6614_diabetes_prediction/pull/8) |
| Four-model automated reporting (E04b) | [PR #9](https://github.com/ryan-dozier/cap6614_diabetes_prediction/pull/9) |

Remaining research work includes the uncertainty/repeat-run protocol, related work,
the central contribution/adaptation protocol, and subsequent experiments and
ablations. See [PROJECT_CHECKLIST.md](PROJECT_CHECKLIST.md) for task ownership,
full commit/PR history, contribution records, and acceptance criteria.

## Repository map and technical references

```text
configs/       Experiment, smoke-test, and comparison-report configurations
scripts/       Downloading, preparation, and dataset diagnostics
src/           Data loading, model adapters, experiments, analysis, reporting
tests/         Automated checks
docs/          Integration plans and saved-results reviews
data/raw/      Bundled synthetic CSVs and downloaded NHANES components
data/processed/  Generated harmonized datasets and metadata
data/splits/   Generated fixed split assignments and partitions
results/       Generated experiments, analyses, and reports
checkpoints/   Locally acquired pretrained weights
```

Run Python entry points as modules (`python3 -m ...`) from the repository root.
Shared paths are defined in `src/paths.py`.

- [Documentation index](docs/README.md)
- [Environment setup and maintenance](docs/guides/environment_setup.md)
- [Data preparation and analysis](docs/guides/data_and_analysis.md)
- [Shared predictive runner and artifacts](docs/guides/predictive_experiments.md)
- [FT-Transformer usage](docs/guides/ft_transformer_usage.md)
- [TabPFN usage and persistence](docs/guides/tabpfn_usage.md)
- [TabICL setup and persistence](docs/guides/tabicl_usage.md)
- [TabPFN benchmarking](docs/guides/tabpfn_benchmark.md)
- [Comparison reporting](docs/guides/reporting.md)
- [FT-Transformer integration plan](docs/plans/ft_transformer_integration_plan.md)
- [TabICL integration plan](docs/plans/tabicl_integration_plan.md)
- [FT-Transformer predictive findings](docs/results/ft_transformer_predictive_results.md)
- [TabICL predictive findings](docs/results/tabicl_predictive_results.md)

## Common setup questions

| Problem | What to check |
|---|---|
| Setup reports an incorrect environment | Run `conda activate synthetic-tabular-ai` and confirm Python 3.11 |
| A synthetic raw CSV is missing | Confirm this is a complete checkout; use the provenance links to restore the required file |
| NHANES download fails | Check internet access and `curl`; inspect the failing download before rerunning setup |
| A previously interrupted NHANES file is skipped | The downloader skips existing files without integrity checking; inspect or reacquire the affected component before preparation |
| CUDA is unavailable | Check `torch.cuda.is_available()`; use the CPU CatBoost configuration for the first experiment |
| A pretrained checkpoint is missing | Complete the model-specific checkpoint instructions before running TabPFN/TabICL |
| An output directory already exists | Choose a distinct experiment configuration/output or a new report output directory |
| Reporting cannot find selected runs | Generate all runs named in that manifest; data preparation does not create experiment results |

To check GPU visibility:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

To run automated checks independently of the onboarding workflow:

```bash
python3 -m unittest discover -s tests -v
```

Some real-model smoke workflows require additional checkpoints and hardware;
their instructions and evidence are in the model-specific documentation.
