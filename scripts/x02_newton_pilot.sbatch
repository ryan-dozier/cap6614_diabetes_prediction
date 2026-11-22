#!/usr/bin/env bash
#SBATCH --account=course_cap6614
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ry482284/cap6616/cap6614_diabetes_prediction}"
PYTHON="${PYTHON:-/home/ry482284/.conda/envs/synthetic-tabular-ai/bin/python}"
MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_DIR/results/adaptation/x01_v1/manifest}"
TABPFN_CHECKPOINT="${TABPFN_CHECKPOINT:-/home/ry482284/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors}"
TABICL_CHECKPOINT="${TABICL_CHECKPOINT:-$PROJECT_DIR/checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt}"
wave="${1:-}"

case "$wave" in
    first|full-training|timing-catboost|timing-ft_transformer|timing-tabpfn|timing-tabicl) ;;
    *) echo "Usage: sbatch scripts/x02_newton_pilot.sbatch first|full-training|timing-MODEL" >&2; exit 2 ;;
esac
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-}" != 4 ]]; then
    echo "Expected one wave, a Slurm job ID, and the approved four-CPU profile." >&2
    exit 2
fi

cd "$PROJECT_DIR"
PROJECT_DIR="$(pwd -P)"
root="$PROJECT_DIR/results/adaptation/x01_v1/development/stage2e_${wave}_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$root")"
mkdir "$root"
mkdir "$root/logs" "$root/runs"
exec > >(tee "$root/job.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    echo "Wave $wave job $SLURM_JOB_ID exit status: $status"
    if [[ -x "$PYTHON" && -f "$PROJECT_DIR/scripts/x02_pilot_summary.py" ]]; then
        if "$PYTHON" "$PROJECT_DIR/scripts/x02_pilot_summary.py" "$root" > "$root/summary.json.tmp"; then
            mv "$root/summary.json.tmp" "$root/summary.json"
        else
            echo "ERROR: summary generation failed; inspect retained logs." >&2
            if (( status == 0 )); then status=1; fi
        fi
    fi
    echo "Evidence: $root"
    exit "$status"
}
trap finish EXIT

if [[ "$wave" == first ]]; then
    runs=(
        catboost_mixed_source_B250_S1800_seed42
        ft_transformer_mixed_kaggle_B250_S1800_seed42
        tabpfn_real_only_none_B250_S0_seed42
        tabicl_mixed_source_B250_S1800_seed42
    )
    query_size=64
elif [[ "$wave" == full-training ]]; then
    runs=(
        catboost_mixed_kaggle_B1000_S1200_seed42
        catboost_synthetic_only_kaggle_B0_S2000_seed42
        ft_transformer_mixed_kaggle_B1000_S1200_seed42
        ft_transformer_synthetic_only_kaggle_B0_S2000_seed42
    )
    query_size=446
else
    model="${wave#timing-}"
    runs=(
        "${model}_real_only_none_B1000_S0_seed42"
        "${model}_mixed_source_B1000_S1200_seed42"
        "${model}_mixed_kaggle_B1000_S1200_seed42"
        "${model}_mixed_source_B500_S1600_seed42"
        "${model}_synthetic_only_source_B0_S2000_seed42"
        "${model}_synthetic_only_kaggle_B0_S2000_seed42"
    )
    query_size=446
fi
printf '%s\n' "${runs[@]}" > "$root/plan.txt"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VECLIB_MAXIMUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

echo "Wave=$wave job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "Project=$PROJECT_DIR Python=$PYTHON manifest=$MANIFEST_DIR"
echo "Checkpoints: TabPFN=$TABPFN_CHECKPOINT TabICL=$TABICL_CHECKPOINT"
echo "Slurm account=${SLURM_JOB_ACCOUNT:-unknown} partition=${SLURM_JOB_PARTITION:-unknown} CPUs=$SLURM_CPUS_PER_TASK"
echo "Threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS VECLIB=$VECLIB_MAXIMUM_THREADS"
echo "Offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "Git revision: $(git rev-parse HEAD)"
git status --short --untracked-files=all
git diff --stat
git diff --cached --stat
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv

[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 1; }
for file in \
    "$MANIFEST_DIR/assignments.json" "$MANIFEST_DIR/metadata.json" \
    "$PROJECT_DIR/data/processed/metadata.json" "$PROJECT_DIR/results/nhanes_preparation/metadata.json" \
    "$PROJECT_DIR/data/processed/nhanes_common_16.csv" "$PROJECT_DIR/data/processed/source_common.csv" \
    "$PROJECT_DIR/data/processed/kaggle_common.csv" \
    "$PROJECT_DIR/data/splits/nhanes_split_assignments.csv" \
    "$PROJECT_DIR/data/splits/source_split_assignments.csv" \
    "$PROJECT_DIR/data/splits/kaggle_split_assignments.csv" \
    "$PROJECT_DIR/configs/catboost_source_n2000.json" \
    "$PROJECT_DIR/configs/ft_transformer_source_n2000.json" \
    "$PROJECT_DIR/configs/tabpfn_source_n2000_float32_batch1024.json" \
    "$PROJECT_DIR/configs/tabicl_v2_source_n2000_none_fp32.json" \
    "$PROJECT_DIR/src/experiments/adaptation_manifest.py" \
    "$PROJECT_DIR/src/experiments/adaptation_matrix.py" \
    "$PROJECT_DIR/src/experiments/adaptation_runner.py" \
    "$TABPFN_CHECKPOINT" "$TABICL_CHECKPOINT" \
    "$PROJECT_DIR/scripts/x02_pilot_summary.py"; do
    [[ -f "$file" && -r "$file" ]] || { echo "ERROR: missing/unreadable file: $file" >&2; exit 1; }
done
[[ -x /usr/bin/time ]] || { echo "ERROR: /usr/bin/time is required." >&2; exit 1; }

"$PYTHON" - <<'PY'
import importlib.metadata
import platform
import torch

print("Python:", platform.python_version())
for name in ("numpy", "pandas", "scikit-learn", "catboost", "torch", "rtdl-revisiting-models", "tabpfn", "tabicl"):
    print(f"{name}: {importlib.metadata.version(name)}")
print("PyTorch CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this GPU allocation.")
print("CUDA device:", torch.cuda.get_device_name(0))
PY

printf '%s  %s\n' \
    fea3d3c14d7d32fb2ee371981b2e2c907cc9b1083ac56f4dfd5fdedaaacfb59d "$MANIFEST_DIR/assignments.json" \
    ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3 "$TABPFN_CHECKPOINT" \
    bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0 "$TABICL_CHECKPOINT" | sha256sum --check
"$PYTHON" -m src.experiments.adaptation_manifest validate --output-dir "$MANIFEST_DIR"

for run in "${runs[@]}"; do
    args=("$PYTHON" -u -m src.experiments.adaptation_runner execute
        --manifest-dir "$MANIFEST_DIR" --run-id "$run" --mode development
        --output-root "$root/runs" --query-size "$query_size")
    case "$run" in
        catboost_*|ft_transformer_*)
            if [[ "$wave" != full-training ]]; then
                args+=(--training-cap 5)
            fi
            ;;
        tabpfn_*) args+=(--checkpoint "$TABPFN_CHECKPOINT") ;;
        tabicl_*) args+=(--checkpoint "$TABICL_CHECKPOINT") ;;
    esac
    printf 'Running:'
    printf ' %q' "${args[@]}"
    printf '\n'
    printf '%q ' "${args[@]}" > "$root/logs/$run.command"
    printf '\n' >> "$root/logs/$run.command"
    if /usr/bin/time -v -o "$root/logs/$run.time" "${args[@]}" > "$root/logs/$run.log" 2>&1; then
        :
    else
        status=$?
        echo "ERROR: $run failed with status $status; see $root/logs/$run.log" >&2
        exit "$status"
    fi
    echo "Completed: $run"
done
echo "Wave $wave completed all ${#runs[@]} runs."
