#!/usr/bin/env bash
# SLURM array shard: append missing non-unique (full cartesian) placements to an
# existing unique-replica co-sim corpus (warmth_v2 / sparse_warmth_v2).
#
# Uses generate_non_unique_placements_fast.py — does NOT replace placements.jsonl;
# merges cartesian \ unique into the existing file per ds_*.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:?OUTPUT_SUBDIR required}"
REMAINING_START="${REMAINING_START:-0}"
TOTAL_DATASETS="${TOTAL_DATASETS:?TOTAL_DATASETS required}"
NUM_SHARDS="${NUM_SHARDS:-4}"
WORKERS="${WORKERS:-31}"

export PYTHONUNBUFFERED=1
export COSIM_SUPPRESS_SIM_PRINTS="${COSIM_SUPPRESS_SIM_PRINTS:-1}"
export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"

mkdir -p logs "simulation_data/${OUTPUT_SUBDIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

scratch_dir="/scratch/${USER}_${SLURM_JOB_ID:-local}"
mkdir -p "${scratch_dir}"
chmod 700 "${scratch_dir}"
export SLURM_SCRATCH="${scratch_dir}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID required (submit via warmth_non_unique_*.sbatch)" >&2
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
temp_dir="simulation_data/temp_non_unique_${OUTPUT_SUBDIR}_j${SLURM_JOB_ID}_a${SLURM_ARRAY_TASK_ID}"
LOG="logs/non_unique_${OUTPUT_SUBDIR}_array${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Non-unique placement backfill (array ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}) ==="
echo "Node: ${SLURMD_NODENAME:-unknown} · Job: ${SLURM_JOB_ID:-local}"
echo "Corpus: ${base_dir}"
echo "Range: ds_$(printf '%05d' "${start}") .. ds_$(printf '%05d' "$((start + count - 1))") (${count} indices)"
echo "Workers: ${WORKERS} · temp: ${temp_dir}"
echo "Log: ${LOG}"

python3 -u scripts_cosim/generate_non_unique_placements_fast.py \
  --datasets-dir "${base_dir}" \
  --start-from "${start}" \
  --max-datasets "${count}" \
  --workers "${WORKERS}" \
  --temp-dir "${temp_dir}" \
  --quiet \
  2>&1 | tee "${LOG}"

echo "=== Done non-unique array ${SLURM_ARRAY_TASK_ID} ==="
tail -5 "${LOG}" 2>/dev/null || true

rm -rf "${temp_dir}" || true
if [[ -d "${scratch_dir}" && "${scratch_dir}" == /scratch/* ]]; then
  rm -rf "${scratch_dir}" || true
fi
