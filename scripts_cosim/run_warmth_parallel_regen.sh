#!/usr/bin/env bash
# Parallel 4-shard warmth co-sim regen (fast path: no inline SSC capture).
# Post-hoc enrichment: refresh_optimal_full_stats.py --repair --force
#
# MANDATORY: each ds_* must keep placements/placements.jsonl (full placement–RTT sweep).
# Repair/recache does NOT replace JSONL. --resume skips only when best.json AND JSONL exist.
# See memory/placements_jsonl_required.md
#
# Note: generate_gnn_datasets_fast.py has 500 combo grid (1×5×20×5); dir name keeps "1060" history.
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL="${TOTAL_DATASETS:-500}"
SHARDS="${NUM_SHARDS:-4}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
WARMTH="${WARMTH_PHYSICS:-node_disk_v2}"
WORKERS_PER_SHARD="${WORKERS_PER_SHARD:-7}"

per_shard=$(( (TOTAL + SHARDS - 1) / SHARDS ))
mkdir -p logs

echo "=== Launching ${SHARDS} parallel regen shards (${TOTAL} datasets, ~${per_shard}/shard, ${WORKERS_PER_SHARD} workers each) ==="

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  start=$(( shard * per_shard ))
  if (( start >= TOTAL )); then
    continue
  fi
  count=$(( per_shard ))
  if (( start + count > TOTAL )); then
    count=$(( TOTAL - start ))
  fi

  log="logs/warmth_regen_shard${shard}_${start}_$((start + count - 1)).log"
  progress="progress_${OUTPUT_SUBDIR}_shard${shard}.txt"

  echo "  shard ${shard}: ds_$(printf '%05d' "${start}")..ds_$(printf '%05d' "$((start + count - 1))") -> ${log}"

  (
    export GNN_CAPTURE_DATASET_STATE=0
    export COSIM_SUPPRESS_SIM_PRINTS=1
    pipenv run python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
      --quiet \
      --warmth-physics "${WARMTH}" \
      --resume \
      --output-subdir "${OUTPUT_SUBDIR}" \
      --progress-log-name "${progress}" \
      --start-from "${start}" \
      --max-datasets "${count}" \
      --workers "${WORKERS_PER_SHARD}"
  ) > "${log}" 2>&1 &
  pids+=($!)
done

echo "PIDs: ${pids[*]}"
echo "${pids[@]}" > logs/warmth_parallel_regen.pids
echo "Monitor: tail -f logs/warmth_regen_shard*.log"
echo "Progress: wc -l simulation_data/${OUTPUT_SUBDIR}/ds_*/best.json"

wait "${pids[@]}"
echo "=== All shards complete ==="
