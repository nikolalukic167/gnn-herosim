#!/usr/bin/env bash
# One SLURM array task: co-sim warmth regen for a contiguous index range.
# Uses SLURM_ARRAY_TASK_ID to pick start/count; auto-sets workers from SLURM_CPUS_PER_TASK.
#
# REQUIRED OUTPUT per ds_*: placements/placements.jsonl (not optional — RTT-hash training).
# --resume skips only best.json + non-empty JSONL. Repair does not recreate JSONL.
# memory/placements_jsonl_required.md
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
WARMTH_PHYSICS="${WARMTH_PHYSICS:-node_disk_v2}"
GRID="${GRID:-warmth_v2}"
REMAINING_START="${REMAINING_START:-235}"
TOTAL_DATASETS="${TOTAL_DATASETS:-500}"
NUM_SHARDS="${NUM_SHARDS:-4}"
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

# Job-specific scratch (cluster etiquette: IO-heavy work off /home).
# Some nodes lack /scratch — fall back to $TMPDIR or repo-local scratch.
scratch_dir=""
for candidate in \
  "/scratch/${USER}_${SLURM_JOB_ID:-local}" \
  "${TMPDIR:-/tmp}/herosim_${USER}_${SLURM_JOB_ID:-local}" \
  "${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-local}"; do
  if mkdir -p "${candidate}" 2>/dev/null; then
    chmod 700 "${candidate}" 2>/dev/null || true
    scratch_dir="${candidate}"
    break
  fi
done
if [[ -z "${scratch_dir}" ]]; then
  echo "ERROR: could not create scratch directory (tried /scratch, TMPDIR, ${PROJECT_ROOT}/.scratch)" >&2
  exit 1
fi
export SLURM_SCRATCH="${scratch_dir}"

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
  echo "ERROR: SLURM_ARRAY_TASK_ID required (submit via warmth_regen_remaining.sbatch)" >&2
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
LOG="logs/warmth_regen_array${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Warmth regen shard (array ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}) ==="
echo "Node: ${SLURMD_NODENAME:-unknown} · Job: ${SLURM_JOB_ID:-local} · Scratch: ${scratch_dir}"
echo "Output: simulation_data/${OUTPUT_SUBDIR}"
echo "Grid: ${GRID}"
echo "Range: ds_$(printf '%05d' "${start}") .. ds_$(printf '%05d' "$((start + count - 1))") (${count} indices, --resume skips done)"
echo "Workers: ${WORKERS} (cpus-per-task=${SLURM_CPUS_PER_TASK:-n/a})"
echo "Log: ${LOG}"

ONLY_MISSING_JSONL="${ONLY_MISSING_JSONL:-0}"
extra_args=()
if [[ "${ONLY_MISSING_JSONL}" == "1" ]]; then
  extra_args+=(--only-missing-jsonl)
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
find "simulation_data/${OUTPUT_SUBDIR}" -name best.json 2>/dev/null | wc -l
tail -3 "logs/${progress_name}" 2>/dev/null || true

# Best-effort scratch cleanup (etiquette)
if [[ -d "${scratch_dir}" && "${scratch_dir}" == /scratch/* ]]; then
  rm -rf "${scratch_dir}" || true
fi
