#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="simulation_data/normal_sim_sweeps/gnn_worthy_bush_5_decode_ablation_${TS}"
RES_DIR="${OUT_ROOT}/results"
mkdir -p "$RES_DIR" logs

export GNN_MODEL_PATH="models/worthy-bush-5.pt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GNN_DECODE_TOP_K="${GNN_DECODE_TOP_K:-10}"

CONFIG="simulation_data/space_with_network.json"
WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
TIMEOUT=3600
SEED=42

log() { echo "[$(date -Is)] $*"; }

run_mode() {
  local mode="$1"
  local out="${RES_DIR}/default_20_20_p50_${mode}.json"
  if [[ -f "$out" ]]; then
    log "SKIP ${mode} (exists)"
    return 0
  fi
  export GNN_DECODE_MODE="$mode"
  log "Running ${mode} -> ${out}"
  local start elapsed
  start=$(date +%s)
  pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn \
    --config "$CONFIG" \
    --workload "$WORKLOAD" \
    --output "$out" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"
  elapsed=$(( $(date +%s) - start ))
  log "${mode} SUCCESS ${elapsed}s"
}

log "=== worthy-bush-5 decode ablation (default_20_20_p50) ==="
log "Output: ${OUT_ROOT}"

run_mode "argmax"
run_mode "frozen"
run_mode "frozen_topk"

log "=== decode ablation complete ==="
