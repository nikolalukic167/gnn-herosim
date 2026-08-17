#!/usr/bin/env bash
# One SLURM array task: --repair --force for a contiguous ds_* index range.
# Re-runs optimal placement with SIM_FORCE_FULL_STATS=1 to backfill disk_snapshot
# and scheduling-time SSC fields (B1 feature plumbing).
#
# Does NOT create placements/placements.jsonl. Datasets missing JSONL still need BF regen.
# memory/placements_jsonl_required.md
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
REMAINING_START="${REMAINING_START:-0}"
TOTAL_DATASETS="${TOTAL_DATASETS:-500}"
NUM_SHARDS="${NUM_SHARDS:-4}"

export PYTHONUNBUFFERED=1
export COSIM_SUPPRESS_SIM_PRINTS="${COSIM_SUPPRESS_SIM_PRINTS:-1}"

mkdir -p logs "simulation_data/${OUTPUT_SUBDIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

scratch_dir="/scratch/${USER}_${SLURM_JOB_ID:-local}"
if [[ ! -d "${scratch_dir}" ]]; then
  mkdir -p "${scratch_dir}"
  chmod 700 "${scratch_dir}"
fi
export SLURM_SCRATCH="${scratch_dir}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID required (submit via warmth_repair*.sbatch)" >&2
  exit 1
fi

remaining=$((TOTAL_DATASETS - REMAINING_START))
per_shard=$(( (remaining + NUM_SHARDS - 1) / NUM_SHARDS ))
start=$(( REMAINING_START + SLURM_ARRAY_TASK_ID * per_shard ))
if (( start >= TOTAL_DATASETS )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID}: nothing to do (start ${start} >= ${TOTAL_DATASETS})"
  exit 0
fi
count="${per_shard}"
if (( start + count > TOTAL_DATASETS )); then
  count=$(( TOTAL_DATASETS - start ))
fi

base_dir="simulation_data/${OUTPUT_SUBDIR}"
progress_name="repair_${OUTPUT_SUBDIR}_array${SLURM_ARRAY_TASK_ID}.txt"
LOG="logs/warmth_repair_${OUTPUT_SUBDIR}_array${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Warmth repair shard (array ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}) ==="
echo "Node: ${SLURMD_NODENAME:-unknown} · Job: ${SLURM_JOB_ID:-local} · Scratch: ${scratch_dir}"
echo "Base dir: ${base_dir}"
echo "Range: ds_$(printf '%05d' "${start}") .. ds_$(printf '%05d' "$((start + count - 1))") (${count} indices)"
echo "Log: ${LOG}"

python3 -u scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir "${base_dir}" \
  --repair \
  --force \
  --start-from "${start}" \
  --max-datasets "${count}" \
  2>&1 | tee "${LOG}"

echo "=== Done repair array ${SLURM_ARRAY_TASK_ID} ==="
tail -1 "${LOG}" 2>/dev/null || true
echo "Progress log: logs/${progress_name}"

if [[ -d "${scratch_dir}" && "${scratch_dir}" == /scratch/* ]]; then
  rm -rf "${scratch_dir}" || true
fi
