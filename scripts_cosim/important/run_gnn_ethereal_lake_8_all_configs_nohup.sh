#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/gnn_ethereal_lake_8_20260608}"
TS="$(basename "$OUT_ROOT" | grep -oE '[0-9]{8}(_[0-9]{6})?' | tail -1 || date +%Y%m%d_%H%M%S)"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/gnn_ethereal_lake_8_sweep_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
MODEL_PATH="models/ethereal-lake-8.pt"
TIMEOUT=3600
SEED=42

export GNN_MODEL_PATH="${MODEL_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*"
}

log "=== GNN ethereal-lake-8 sweep start ==="
log "Model: ${MODEL_PATH}"
log "Output: ${OUT_ROOT}"
log "Progress log: ${PROGRESS_LOG}"

run_one() {
  local name="$1"
  local config="$2"
  local output="$3"
  if [[ -f "$output" ]]; then
    log "${name} SKIP (exists)"
    echo "${name} SKIP ${output}" >> "$PROGRESS_LOG"
    return 0
  fi
  log "Running ${name}"
  local start elapsed
  start=$(date +%s)
  if pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"; then
    elapsed=$(( $(date +%s) - start ))
    log "${name} SUCCESS ${elapsed}s"
    echo "${name} SUCCESS ${elapsed}s" >> "$PROGRESS_LOG"
  else
    elapsed=$(( $(date +%s) - start ))
    log "${name} FAILED ${elapsed}s"
    echo "${name} FAILED ${elapsed}s" >> "$PROGRESS_LOG"
  fi
}

run_one "default_20_20_p50" "$BASE_CONFIG" "${RES_DIR}/default_20_20_p50.json"

if [[ ! -d "$SWEEP_CFG_SRC" ]]; then
  log "ERROR: sweep config source missing: ${SWEEP_CFG_SRC}"
  exit 1
fi
cp "$SWEEP_CFG_SRC"/*.json "$CFG_DIR"/

for cfg in "$CFG_DIR"/??_*.json; do
  base_name="$(basename "$cfg" .json)"
  run_one "$base_name" "$cfg" "${RES_DIR}/${base_name}.json"
done

log "=== GNN ethereal-lake-8 sweep complete ==="
