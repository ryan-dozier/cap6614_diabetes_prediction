#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 4 ]] || {
    echo "Expected model, plan, Slurm job ID and four CPUs." >&2
    exit 2
}
PROJECT_DIR="${PROJECT_DIR:-/home/ry482284/cap6616/cap6614_diabetes_prediction}"
PYTHON="${PYTHON:-/home/ry482284/.conda/envs/synthetic-tabular-ai/bin/python}"
JOB_ROOT="${JOB_ROOT:-$PROJECT_DIR/results/adaptation/x01_v1/execution/jobs}"
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VECLIB_MAXIMUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
cd "$PROJECT_DIR"
args=("$PYTHON" -u -m src.experiments.adaptation_launch execute
    --model "$1" --plan "$2" --job-dir "$JOB_ROOT/${1}_${SLURM_JOB_ID}")
exec "${args[@]}"
