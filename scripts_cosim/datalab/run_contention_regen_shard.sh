#!/usr/bin/env bash
# One SLURM array task: contention co-sim regen (v2 finisher or v3 full grid).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:?OUTPUT_SUBDIR required}"
GRID="${GRID:?GRID required}"
WARMTH_PHYSICS="${WARMTH_PHYSICS:-node_disk_v2}"
REMAINING_START="${REMAINING_START:-0}"
TOTAL_DATASETS="${TOTAL_DATASETS:?TOTAL_DATASETS required}"
NUM_SHARDS="${NUM_SHARDS:-10}"
WORKERS="${WORKERS:-}"
ONLY_MISSING_JSONL="${ONLY_MISSING_JSONL:-0}"
ALLOW_NON_UNIQUE="${ALLOW_NON_UNIQUE:-1}"

export PYTHONUNBUFFERED=1
export GNN_CAPTURE_DATASET_STATE="${GNN_CAPTURE_DATASET_STATE:-0}"
export COSIM_SUPPRESS_SIM_PRINTS="${COSIM_SUPPRESS_SIM_PRINTS:-1}"

mkdir -p logs "simulation_data/${OUTPUT_SUBDIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && -z "${WORKERS}" ]]; then
  if [[ "${SLURM_CPUS_PER_TASK}" -gt 1 ]]; then
    WORKERS=$((SLURM_CPUS_PER_TASK - 1))
  else
    WORKERS=1
  fi
fi
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
if [[ "${WORKERS}" -lt 1 ]]; then
  WORKERS=1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID required" >&2
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

progress_name="progress_${OUTPUT_SUBDIR}_array${SLURM_ARRAY_TASK_ID}.txt"
LOG="logs/contention_regen_array${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Contention regen shard (array ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}) ==="
echo "Output: simulation_data/${OUTPUT_SUBDIR} · Grid: ${GRID}"
echo "Range: ds_$(printf '%05d' "${start}") .. ds_$(printf '%05d' "$((start + count - 1))") (${count} indices)"
echo "Workers: ${WORKERS} · only_missing_jsonl=${ONLY_MISSING_JSONL} · allow_non_unique=${ALLOW_NON_UNIQUE}"
echo "Log: ${LOG}"

extra_args=()
if [[ "${ONLY_MISSING_JSONL}" == "1" ]]; then
  extra_args+=(--only-missing-jsonl)
fi
if [[ "${ALLOW_NON_UNIQUE}" == "1" ]]; then
  extra_args+=(--allow-non-unique-replicas)
fi

python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
  --quiet \
  --grid "${GRID}" \
  --max-datasets "${count}" \
  --start-from "${start}" \
  --resume \
  --warmth-physics "${WARMTH_PHYSICS}" \
  --output-subdir "${OUTPUT_SUBDIR}" \
  --progress-log-name "${progress_name}" \
  --workers "${WORKERS}" \
  "${extra_args[@]}" \
  2>&1 | tee "${LOG}"

echo "=== Done array ${SLURM_ARRAY_TASK_ID} ==="
jsonl=0
for d in simulation_data/"${OUTPUT_SUBDIR}"/ds_*; do
  if [[ -s "$d/placements/placements.jsonl" ]]; then
    jsonl=$((jsonl + 1))
  fi
done
echo "datasets with placements.jsonl: ${jsonl}"
exit 0
