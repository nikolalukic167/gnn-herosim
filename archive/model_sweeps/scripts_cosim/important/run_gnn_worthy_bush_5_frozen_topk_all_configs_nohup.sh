#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/gnn_worthy_bush_5_frozen_topk_${TS}}"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/gnn_worthy_bush_5_frozen_topk_sweep_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
MODEL_PATH="models/worthy-bush-5.pt"
TIMEOUT=3600
SEED=42

export GNN_MODEL_PATH="${MODEL_PATH}"
export GNN_DECODE_MODE="${GNN_DECODE_MODE:-frozen_topk}"
export GNN_DECODE_TOP_K="${GNN_DECODE_TOP_K:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*"
}

log "=== GNN worthy-bush-5 frozen_topk sweep start ==="
log "Model: ${MODEL_PATH}"
log "Decode: ${GNN_DECODE_MODE} (top_k=${GNN_DECODE_TOP_K})"
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

log "=== GNN worthy-bush-5 frozen_topk sweep complete ==="
