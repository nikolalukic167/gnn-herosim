#!/bin/bash
#SBATCH --job-name=herosim-prep-ram
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/prep-graphs-ram-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/prep-graphs-ram-%j.err
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
DATA_DIR="/share/nikola.lukic/herosim/data/gnn_datasets_4tasks_overnight_260422"
CACHE_DIR="/share/nikola.lukic/herosim/cache/graphs_cache_gnn_datasets_4tasks_overnight_260422_embedded"
ENV_NAME="${HEROSIM_ENV_NAME:-myenv}"
RTT_WORKERS="${HEROSIM_RTT_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
RTT_BATCH_SIZE="${HEROSIM_RTT_BATCH_SIZE:-128}"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${CACHE_DIR}"

scratch_dir="/scratch/${USER}_${SLURM_JOB_ID}"
mkdir -p "${scratch_dir}"
chmod 700 "${scratch_dir}"
export HEROSIM_PREPARE_SCRATCH="${scratch_dir}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project root: ${PROJECT_ROOT}"
echo "Input data: ${DATA_DIR}"
echo "Final cache: ${CACHE_DIR}"
echo "Scratch staging: ${HEROSIM_PREPARE_SCRATCH}"
echo "Environment: ${ENV_NAME}"
echo "RTT workers: ${RTT_WORKERS}"
echo "RTT batch size: ${RTT_BATCH_SIZE}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

python3 -c 'import importlib.util, sys; required = ["numpy", "pandas", "torch", "torch_geometric", "tqdm"]; missing = [name for name in required if importlib.util.find_spec(name) is None]; print("Missing Python package(s): " + ", ".join(missing), file=sys.stderr) if missing else None; sys.exit(1 if missing else 0)'

python3 "${PROJECT_ROOT}/src/notebooks/prepare_graphs_ram.py" \
  --project-root "${PROJECT_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --base-dirs "${DATA_DIR}" \
  --ram-staging \
  --rtt-workers "${RTT_WORKERS}" \
  --rtt-batch-size "${RTT_BATCH_SIZE}"

echo "Graph cache ready at: ${CACHE_DIR}"
