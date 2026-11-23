# D08 — Clean-environment reproduction on Newton

## Scope and contributor

Performed by Arthur Lookshin, with AI assistance, on November 14–15, 2026.
Starting revision: d259c5b5e7c0ba239b431f9cc7e359fcfca3fb0e.
Branch: d08-clean-environment.

Verified installation, data preparation, automated tests, and small
FT-Transformer functional workflows on CPU and GPU. These checks do not
establish full-scale experiment feasibility or predictive superiority.

## Execution environment

- Cluster: UCF Newton
- Slurm account: course_cap6614
- Partition: normal
- Anaconda module: anaconda/anaconda-2024.10
- Conda: 24.11.3
- Python: 3.11.17
- PyTorch: 2.14.1+cu126
- CPU allocation: job 1207429, node evc2
- GPU allocation: job 1207755, node evc1
- Requested resources per allocation: four CPU resources and 16 GB RAM
- GPU: Tesla V100-PCIE-16GB
- NVIDIA driver: 580.105.08
- PyTorch CUDA build: 12.6

## Installation and package-source finding

The first clean installation using the supplied environment.yml failed:

    ERROR: No matching distribution found for torch==2.14.1+cu126

The exact pinned build was available from the official PyTorch CUDA 12.6
package index. A second installation used a separate, fresh environment
location and completed with exit code 0 without changing package pins:

```bash
module load anaconda/anaconda-2024.10

export CONDA_ENVS_PATH="$HOME/projects/d08-newton/conda-envs-with-pytorch-index"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 \
conda env create -f environment.yml
```

Activation and verification:

```bash
conda activate "$HOME/projects/d08-newton/conda-envs-with-pytorch-index/synthetic-tabular-ai"
python3 --version
python3 -m pip check
python3 -c "import torch; print(torch.__version__)"
```

Observed results:

```text
Python 3.11.17
No broken requirements found.
2.14.1+cu126
```

The failed first environment was preserved separately. The additional package
source is required by this tested installation procedure; the repository's
environment manifest was not changed.

## Setup-script portability fix

Conda reported CONDA_DEFAULT_ENV as the full environment path rather than
only synthetic-tabular-ai. The original exact-name comparison would reject
this working environment.

scripts/initial_setup.sh now compares the final path component, accepting
either the short name or a full path ending in synthetic-tabular-ai:

```bash
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
if [ "${ACTIVE_ENV##*/}" != "$EXPECTED_ENV" ]; then
```

The script passed bash -n after this change. This remains a name check,
not independent verification of every installed package.

## Data preparation and automated tests

From the repository root, with the environment active:

```bash
bash scripts/initial_setup.sh
python3 -m unittest discover -s tests -v
```

Data preparation completed with exit code 0 and reported:

```text
STRICT_COMMON_16 feature order: OK
Setup complete.
```

The automated suite completed with exit code 0:

```text
Ran 189 tests in 56.135s
OK (skipped=1)
```

This represents 188 passing tests and one skipped test.
test_bounded_cuda_fit_and_resource_hooks was skipped because CUDA was
unavailable in the CPU-only allocation. The full suite was not rerun in
the GPU allocation; separate actual GPU checks are recorded below.

## CPU functional smoke

Thread settings:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

Command:

```bash
python3 -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_source.json \
  --output-dir results/smoke/d08_cpu_20261115T051508Z \
  --mode functional \
  --device cpu
```

Observed: 64 training rows, 32 validation rows, three epochs, exit code 0.
The saved smoke.json reported:

```text
Status: completed
Save/reload: passed
```

## GPU checks and functional smoke

Inside GPU allocation 1207755, PyTorch reported:

```text
CUDA available: True
GPU: Tesla V100-PCIE-16GB
```

A 32-by-32 matrix multiplication executed on cuda:0 and returned the
expected value 32.0.

With the same project environment and thread settings, the model check was:

```bash
python3 -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_source.json \
  --output-dir results/smoke/d08_gpu_1207755 \
  --mode functional \
  --device cuda:0
```

Observed: 64 training rows, 32 validation rows, three epochs, exit code 0.
The saved smoke.json reported:

```text
Status: completed
Save/reload: passed
```

These functional checks use training/validation data rather than held-out
test evaluation. They do not constitute the larger research experiments.

## Checkpoint requirements and verification limits

Installing packages and preparing data do not supply every model's
pretrained checkpoint.

- CatBoost: no pretrained checkpoint required; no separate predictive run
  was performed for this reproduction.
- FT-Transformer: no pretrained checkpoint required. Actual CPU and GPU
  functional execution and save/reload were verified.
- TabPFN: requires the project-specified local pretrained checkpoint and
  any required access/license authorization. See
  [TabPFN usage](tabpfn_usage.md).
- TabICL: requires the project-pinned local pretrained checkpoint, including
  its documented revision and integrity verification. See
  [TabICL usage](tabicl_usage.md#pinned-checkpoint-setup).

TabPFN and TabICL real-model inference/checkpoint loading were not executed
as part of these checks. Their acquisition instructions are referenced,
not independently validated here.

Comparison-report generation from historical saved experiment outputs was
not performed. The reproduced small workflow was the CPU/GPU functional
smoke workflow. No claim is made that all models or reporting paths have
been independently reproduced.

## Evidence retained on Newton

The following files are local, ignored run evidence. They are not included
automatically in a clone or this PR. Key observed results are reproduced
above so this guide does not depend solely on inaccessible local files.

Logs under results/d08/logs/:

- starting-state.txt
- environment-create.txt
- torch-index-check.txt
- environment-create-with-pytorch-index.txt
- pip-check.txt
- torch-import.txt
- initial-setup.txt
- tests.txt
- ft-cpu-smoke.txt
- gpu-check.txt
- ft-gpu-smoke.txt

Saved smoke records:

- results/smoke/d08_cpu_20261115T051508Z/smoke.json
- results/smoke/d08_gpu_1207755/smoke.json

Existing output directories must not be reused for reruns. Use new smoke
output names and preserve prior evidence.

## Review status

Execution validated as described above. No large-scale runs were performed.
