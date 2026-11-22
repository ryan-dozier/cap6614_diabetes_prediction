#!/usr/bin/env bash
#SBATCH --account=course_cap6614
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ry482284/cap6616/cap6614_diabetes_prediction}"
[[ $# == 2 ]] || { echo "Usage: sbatch scripts/x03_newton_gpu.sbatch MODEL PLAN.json" >&2; exit 2; }
case "$1" in
    ft_transformer|tabpfn|tabicl) ;;
    *) echo "GPU jobs require ft_transformer, tabpfn or tabicl." >&2; exit 2 ;;
esac
exec bash "$PROJECT_DIR/scripts/x03_newton_common.sh" "$1" "$2"
