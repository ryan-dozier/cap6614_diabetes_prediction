#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPECTED_ENV="synthetic-tabular-ai"

SOURCE_DATA="data/raw/source/diabetes_dataset.csv"
KAGGLE_DATA="data/raw/kaggle_s5e12/train.csv"

echo "========================================"
echo "Diabetes Prediction Project Setup"
echo "========================================"

# -----------------------------------------------------------------------------
# Verify Conda environment
# -----------------------------------------------------------------------------

echo
echo "Checking environment..."

if [ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]; then
    echo "ERROR: Incorrect Conda environment."
    echo
    echo "Current environment:"
    echo "  ${CONDA_DEFAULT_ENV:-none}"
    echo
    echo "Expected environment:"
    echo "  $EXPECTED_ENV"
    echo
    echo "Activate it with:"
    echo "  conda activate $EXPECTED_ENV"
    exit 1
fi

echo "Conda environment: $CONDA_DEFAULT_ENV"

# -----------------------------------------------------------------------------
# Verify Python
# -----------------------------------------------------------------------------

echo
echo "Python:"
python3 --version

PYTHON_MAJOR_MINOR="$(
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"

if [ "$PYTHON_MAJOR_MINOR" != "3.11" ]; then
    echo "ERROR: Expected Python 3.11, found Python $PYTHON_MAJOR_MINOR."
    exit 1
fi

# -----------------------------------------------------------------------------
# Verify manually acquired Source / Kaggle raw data
# -----------------------------------------------------------------------------

echo
echo "Checking required Source/Kaggle raw datasets..."

if [ ! -f "$SOURCE_DATA" ]; then
    echo "ERROR: Missing source dataset:"
    echo "  $SOURCE_DATA"
    exit 1
fi

if [ ! -f "$KAGGLE_DATA" ]; then
    echo "ERROR: Missing Kaggle training dataset:"
    echo "  $KAGGLE_DATA"
    exit 1
fi

echo "Source/Kaggle raw datasets found."

# -----------------------------------------------------------------------------
# Acquire and prepare NHANES first
# -----------------------------------------------------------------------------

echo
echo "Downloading NHANES 2015-2016 components..."
./scripts/download_nhanes_2015_2016.sh

echo
echo "Inspecting NHANES components..."
python3 src/inspect_nhanes.py

echo
echo "Preparing harmonized NHANES datasets and fixed splits..."
python3 src/prepare_nhanes.py

# -----------------------------------------------------------------------------
# Prepare Source / Kaggle canonical data
# -----------------------------------------------------------------------------

echo
echo "Preparing Source/Kaggle canonical datasets and fixed splits..."
python3 src/prepare_data.py

# -----------------------------------------------------------------------------
# Validate unified data-loading interface
# -----------------------------------------------------------------------------

echo
echo "Running three-domain data-loader smoke test..."
python3 src/data.py

echo
echo "========================================"
echo "Setup complete."
echo "========================================"