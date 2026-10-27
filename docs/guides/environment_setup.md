# Environment setup and maintenance

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

The manifests are Linux/WSL CUDA development snapshots, not verified cross-platform installers.
They pin `torch==2.14.1+cu126` without recording a custom package-index URL.
Clean-machine recreation remains to be verified; keep package-source requirements explicit.

This project uses a Conda environment named `synthetic-tabular-ai` with Python 3.11. The recommended setup method is to recreate the environment from the provided `environment.yml` file.

## Using Conda

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

The environment file includes both Conda-managed packages and the pip-installed Python dependencies required by the project, including PyTorch, TabPFN, TabICL 2.2.0, `rtdl-revisiting-models==0.0.2` (FT-Transformer), CatBoost, pandas, SciPy, scikit-learn, and Matplotlib. You should not normally need to run `pip install -r requirements-local.txt` after creating the environment from `environment.yml`.

## Pip Requirements

`requirements-local.txt` contains a snapshot of the pip packages installed in the development environment. It records the development package configuration; a clean pip-only installation has not been independently verified.

For a pip-based installation:

```bash
python3 -m pip install -r requirements-local.txt
```

The development environment includes a CUDA-enabled build of PyTorch and NVIDIA CUDA dependencies. Systems without a compatible NVIDIA GPU or CUDA environment may require a different PyTorch installation.

## Verify the Environment

After activating the environment, the main dependencies can be checked with:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import tabpfn; print('TabPFN:', tabpfn.__version__)"
python3 -c "from importlib.metadata import version; print('TabICL:', version('tabicl'))"
python3 -c "from importlib.metadata import version; print('FT implementation:', version('rtdl-revisiting-models'))"
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
