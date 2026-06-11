#!/usr/bin/env bash
# Co-sim warmth pilot: second half (ds_00025..ds_00049) with node_disk_v2.
# Expects completed ds_00025..ds_00028 rsynced from mitrix when using --resume.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
START_FROM="${START_FROM:-25}"
MAX_DATASETS="${MAX_DATASETS:-25}"
WARMTH_PHYSICS="${WARMTH_PHYSICS:-node_disk_v2}"
WORKERS="${WORKERS:-}"

export PYTHONUNBUFFERED=1
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

LOG="logs/warmth_regen_pilot50_datalab_$(date +%Y%m%d_%H%M%S).log"
echo "=== Warmth pilot second half ==="
echo "Output: simulation_data/${OUTPUT_SUBDIR}"
echo "Range: ds_$(printf '%05d' "${START_FROM}") .. (max ${MAX_DATASETS} from start)"
echo "Workers: ${WORKERS}"
echo "Log: ${LOG}"

python3 scripts_cosim/generate_gnn_datasets_fast.py \
  --max-datasets "${MAX_DATASETS}" \
  --start-from "${START_FROM}" \
  --resume \
  --warmth-physics "${WARMTH_PHYSICS}" \
  --output-subdir "${OUTPUT_SUBDIR}" \
  --workers "${WORKERS}" \
  2>&1 | tee "${LOG}"

echo "=== Done. Progress tail ==="
tail -5 "logs/progress_${OUTPUT_SUBDIR}.txt" 2>/dev/null || true
ls "simulation_data/${OUTPUT_SUBDIR}"/ds_* 2>/dev/null | wc -l
