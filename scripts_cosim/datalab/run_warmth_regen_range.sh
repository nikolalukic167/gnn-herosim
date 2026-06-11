#!/usr/bin/env bash
# Run warmth regen for an explicit contiguous index range (standalone SLURM job).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
WARMTH_PHYSICS="${WARMTH_PHYSICS:-node_disk_v2}"
GRID="${GRID:-warmth_v2}"
START_FROM="${START_FROM:?START_FROM required}"
MAX_DATASETS="${MAX_DATASETS:?MAX_DATASETS required}"
SHARD_LABEL="${SHARD_LABEL:-range}"
WORKERS="${WORKERS:-}"

export PYTHONUNBUFFERED=1
export GNN_CAPTURE_DATASET_STATE="${GNN_CAPTURE_DATASET_STATE:-0}"
export COSIM_SUPPRESS_SIM_PRINTS="${COSIM_SUPPRESS_SIM_PRINTS:-1}"

mkdir -p logs "simulation_data/${OUTPUT_SUBDIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

scratch_dir="/scratch/${USER}_${SLURM_JOB_ID:-local}"
mkdir -p "${scratch_dir}"
chmod 700 "${scratch_dir}"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && -z "${WORKERS}" ]]; then
  WORKERS=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1 ))
fi
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
(( WORKERS >= 1 )) || WORKERS=1

end=$(( START_FROM + MAX_DATASETS - 1 ))
progress_name="progress_${OUTPUT_SUBDIR}_${SHARD_LABEL}.txt"
LOG="logs/warmth_regen_${SHARD_LABEL}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Warmth regen ${SHARD_LABEL} ==="
echo "Node: ${SLURMD_NODENAME:-unknown} · Job: ${SLURM_JOB_ID:-local}"
echo "Range: ds_$(printf '%05d' "${START_FROM}") .. ds_$(printf '%05d' "${end}") (${MAX_DATASETS} indices)"
echo "Grid: ${GRID}"
echo "Workers: ${WORKERS}"

python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
  --quiet \
  --grid "${GRID}" \
  --max-datasets "${MAX_DATASETS}" \
  --start-from "${START_FROM}" \
  --resume \
  --warmth-physics "${WARMTH_PHYSICS}" \
  --output-subdir "${OUTPUT_SUBDIR}" \
  --progress-log-name "${progress_name}" \
  --workers "${WORKERS}" \
  2>&1 | tee "${LOG}"

if [[ -d "${scratch_dir}" && "${scratch_dir}" == /scratch/* ]]; then
  rm -rf "${scratch_dir}" || true
fi
