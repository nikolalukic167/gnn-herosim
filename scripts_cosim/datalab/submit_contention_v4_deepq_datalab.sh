#!/usr/bin/env bash
# Rsync the contention_v4_deepq co-sim sources to datalab and submit the CPU-amd array.
#
# Pilot first (measures coupled fraction on a small sample), then the full grid:
#   MODE=pilot bash scripts_cosim/datalab/submit_contention_v4_deepq_datalab.sh
#   MODE=full  bash scripts_cosim/datalab/submit_contention_v4_deepq_datalab.sh
#
# placements/placements.jsonl is the training label source and is generated on datalab; it
# must be rsynced back before any recache. See docs/notes/placements_jsonl_required.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REMOTE_HOST="${REMOTE_HOST:-datalab}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
MODE="${MODE:-pilot}"

case "$MODE" in
  pilot)
    ARRAY="${ARRAY:-0-2}"
    TOTAL_DATASETS="${TOTAL_DATASETS:-27}"
    NUM_SHARDS="${NUM_SHARDS:-3}"
    OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_contention_v4_pilot}"
    ;;
  full)
    ARRAY="${ARRAY:-0-9}"
    TOTAL_DATASETS="${TOTAL_DATASETS:-900}"
    NUM_SHARDS="${NUM_SHARDS:-10}"
    OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_contention_v4_deepq}"
    ;;
  *)
    echo "ERROR: MODE must be 'pilot' or 'full', got '${MODE}'" >&2
    exit 1
    ;;
esac

# CRLF heal before rsync — Windows-saved .sh/.sbatch fail on the compute nodes.
sed -i 's/\r$//' \
  scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch \
  scripts_cosim/datalab/run_contention_regen_shard.sh \
  scripts_cosim/datalab/submit_contention_v4_deepq_datalab.sh

echo "=== rsync contention_v4_deepq sources -> ${REMOTE_HOST}:${REMOTE_ROOT} ==="
rsync -az \
  scripts_cosim/generate_gnn_datasets_fast.py \
  scripts_cosim/separability_diagnostic.py \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts_cosim/"

rsync -az \
  scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch \
  scripts_cosim/datalab/run_contention_regen_shard.sh \
  scripts_cosim/datalab/submit_contention_v4_deepq_datalab.sh \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts_cosim/datalab/"

# The live sim physics lives in src/ — the generator imports it, so keep it in sync.
rsync -az --delete \
  src/placement/ \
  "${REMOTE_HOST}:${REMOTE_ROOT}/src/placement/"

echo "=== verify grid is registered on remote ==="
ssh "${REMOTE_HOST}" "cd ${REMOTE_ROOT} && grep -q 'contention_v4_deepq' scripts_cosim/generate_gnn_datasets_fast.py && echo 'grid present' || { echo 'ERROR: grid missing on remote' >&2; exit 1; }"

echo "=== sbatch --test-only ==="
ssh "${REMOTE_HOST}" "cd ${REMOTE_ROOT} && sed -i 's/\r\$//' scripts_cosim/datalab/*.sbatch scripts_cosim/datalab/*.sh && sbatch --test-only --array=${ARRAY} --export=ALL,TOTAL_DATASETS=${TOTAL_DATASETS},NUM_SHARDS=${NUM_SHARDS},OUTPUT_SUBDIR=${OUTPUT_SUBDIR} scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch"

echo "=== submit (mode=${MODE}, array=${ARRAY}, total=${TOTAL_DATASETS}, out=${OUTPUT_SUBDIR}) ==="
ssh "${REMOTE_HOST}" "cd ${REMOTE_ROOT} && sbatch --array=${ARRAY} --export=ALL,TOTAL_DATASETS=${TOTAL_DATASETS},NUM_SHARDS=${NUM_SHARDS},OUTPUT_SUBDIR=${OUTPUT_SUBDIR} scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch"

cat <<EOF

Monitor:
  ssh ${REMOTE_HOST} 'squeue -u nikola.lukic'
  ssh ${REMOTE_HOST} 'cd ${REMOTE_ROOT} && find simulation_data/${OUTPUT_SUBDIR} -path "*/placements/placements.jsonl" -size +0 | wc -l'
  ssh ${REMOTE_HOST} 'cd ${REMOTE_ROOT} && tail -5 logs/progress_${OUTPUT_SUBDIR}_array0.txt'

Coupled fraction (the acceptance gate; contention_v2 was 7.1%):
  ssh ${REMOTE_HOST} 'cd ${REMOTE_ROOT} && micromamba run -n gnn python3 scripts_cosim/separability_diagnostic.py simulation_data/${OUTPUT_SUBDIR}'

Pull labels back (placements.jsonl is mandatory):
  rsync -az '${REMOTE_HOST}:${REMOTE_ROOT}/simulation_data/${OUTPUT_SUBDIR}/' 'simulation_data/${OUTPUT_SUBDIR}/'
EOF
