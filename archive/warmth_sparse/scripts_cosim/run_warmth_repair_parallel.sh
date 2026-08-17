#!/usr/bin/env bash
# Local/mitrix: parallel 4-shard --repair --force for warmth_v2 + sparse_warmth_v2.
set -euo pipefail
cd "$(dirname "$0")/.."

SHARDS="${NUM_SHARDS:-4}"
export COSIM_SUPPRESS_SIM_PRINTS=1

repair_dir() {
  local subdir="$1"
  local total="$2"
  local per_shard=$(( (total + SHARDS - 1) / SHARDS ))
  local base="simulation_data/${subdir}"
  mkdir -p logs

  echo "=== Repair ${subdir} (${total} ds, ${SHARDS} shards) ==="
  pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    local start=$(( shard * per_shard ))
    if (( start >= total )); then
      continue
    fi
    local count=$per_shard
    if (( start + count > total )); then
      count=$(( total - start ))
    fi
    local log="logs/warmth_repair_${subdir}_shard${shard}.log"
    echo "  shard ${shard}: ds_$(printf '%05d' "${start}")..ds_$(printf '%05d' "$((start + count - 1))") -> ${log}"
    (
      pipenv run python3 -u scripts_cosim/refresh_optimal_full_stats.py \
        --base-dir "${base}" \
        --repair \
        --force \
        --start-from "${start}" \
        --max-datasets "${count}"
    ) > "${log}" 2>&1 &
    pids+=($!)
  done
  echo "PIDs (${subdir}): ${pids[*]}"
  wait "${pids[@]}"
}

repair_dir "gnn_datasets_4tasks_1060_warmth_v2" "${WARMTH_TOTAL:-500}"
repair_dir "gnn_datasets_4tasks_sparse_warmth_v2" "${SPARSE_TOTAL:-351}"

echo "=== All repair shards complete ==="
