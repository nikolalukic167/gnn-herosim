#!/bin/bash
#SBATCH --job-name=herosim-train-ram
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/train-ram-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/train-ram-%j.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
CACHE_DIR="/share/nikola.lukic/herosim/cache/graphs_cache_gnn_datasets_4tasks_overnight_260422_embedded"
ENV_NAME="${HEROSIM_ENV_NAME:-myenv}"
TORCH_THREADS="${HEROSIM_TORCH_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
BATCH_SIZE="${HEROSIM_TRAIN_BATCH_SIZE:-64}"

export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"
export OPENBLAS_NUM_THREADS="${TORCH_THREADS}"
export PYTHONUNBUFFERED=1

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/models"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project root: ${PROJECT_ROOT}"
echo "Cache: ${CACHE_DIR}"
echo "Environment: ${ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Torch threads: ${TORCH_THREADS}"
echo "Batch size: ${BATCH_SIZE}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"

python3 -c 'import importlib.util, sys; required = ["numpy", "torch", "torch_geometric", "sklearn", "tqdm", "wandb"]; missing = [name for name in required if importlib.util.find_spec(name) is None]; print("Missing Python package(s): " + ", ".join(missing), file=sys.stderr) if missing else None; sys.exit(1 if missing else 0)'

python3 src/notebooks/train_ram.py \
  --project-root "${PROJECT_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-dataloader-workers 0 \
  --torch-threads "${TORCH_THREADS}" \
  --hard-negative-fraction 0.5
