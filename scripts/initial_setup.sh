#!/usr/bin/env bash

set -euo pipefail

# The script lives in scripts/, so the repository root is one directory above.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "========================================"
echo "Diabetes Prediction Project Setup"
echo "========================================"

# ------------------------------------------------------------
# Verify Conda environment
# ------------------------------------------------------------

EXPECTED_ENV="synthetic-tabular-ai"

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

# ------------------------------------------------------------
# Verify Python
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Verify required raw datasets
# ------------------------------------------------------------

SOURCE_DATA="data/raw/source/diabetes_dataset.csv"
KAGGLE_DATA="data/raw/kaggle_s5e12/train.csv"

echo
echo "Checking raw datasets..."

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

echo "Raw datasets found."

# ------------------------------------------------------------
# Prepare canonical datasets and fixed splits
# ------------------------------------------------------------

echo
echo "Preparing datasets..."
python3 src/prepare_data.py

# ------------------------------------------------------------
# Validate prepared data
# ------------------------------------------------------------

echo
echo "Running data-loader smoke test..."
python3 src/data.py

echo
echo "========================================"
echo "Setup complete."
echo "========================================"