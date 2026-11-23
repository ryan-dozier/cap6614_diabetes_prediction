#!/usr/bin/env bash
#SBATCH --account=course_cap6614
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ry482284/cap6616/cap6614_diabetes_prediction}"
[[ $# == 1 ]] || { echo "Usage: sbatch scripts/x03_newton_cpu.sbatch PLAN.json" >&2; exit 2; }
exec bash "$PROJECT_DIR/scripts/x03_newton_common.sh" catboost "$1"
