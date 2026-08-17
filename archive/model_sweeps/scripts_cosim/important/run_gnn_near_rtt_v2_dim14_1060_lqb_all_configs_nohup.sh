#!/usr/bin/env bash
# Full 7-config sweep: dim14-full model with LQB log1p-queue-blend decode.
# GNN_LQB_LAMBDA=1.5: Score = Logit - 1.5*log1p(queue)
# Hypothesis: ranking model wins dense configs (01,02,04) AND recovers
# small/sparse configs (default,00,05) that pure argmax collapsed on.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_1060_lqb15_$(date +%Y%m%d_%H%M%S)}"
TS="$(date +%Y%m%d_%H%M%S)"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/gnn_near_rtt_v2_dim14_1060_lqb15_sweep_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
MODEL_PATH="models/near-rtt-v2-dim14-1060.pt"
TIMEOUT=3600
SEED=42

export GNN_MODEL_PATH="${MODEL_PATH}"
export GNN_DECODE_MODE="argmax"
export GNN_LQB_LAMBDA="1.5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date -Is)] $*" | tee -a "$PROGRESS_LOG"; }

log "=== dim14-full LQB lambda=1.5 sweep start ==="
log "Model: ${MODEL_PATH}  GNN_LQB_LAMBDA=${GNN_LQB_LAMBDA}"
log "Output: ${OUT_ROOT}"

run_one() {
  local name="$1" config="$2" output="$3"
  if [[ -f "$output" ]]; then
    log "${name} SKIP (exists)"; return 0
  fi
  log "Running ${name}"
  local start=$( date +%s )
  if pipenv run python3 scripts_cosim/run_simulation.py \
      --gnn --config "$config" --workload "$WORKLOAD" \
      --output "$output" --timeout "$TIMEOUT" --seed "$SEED"; then
    local rtt
    rtt=$(pipenv run python3 -c "import json; print(f\"{json.load(open('$output'))['total_rtt']/1e6:.3f}M\")" 2>/dev/null || echo "?")
    log "${name} SUCCESS $(( $(date +%s) - start ))s  RTT=${rtt}"
    echo "${name} SUCCESS RTT=${rtt}" >> "$PROGRESS_LOG"
  else
    log "${name} FAILED $(( $(date +%s) - start ))s"
    echo "${name} FAILED" >> "$PROGRESS_LOG"
  fi
}

run_one "default_20_20_p50" "$BASE_CONFIG" "${RES_DIR}/default_20_20_p50.json"

[[ -d "$SWEEP_CFG_SRC" ]] || { log "ERROR: sweep config source missing: ${SWEEP_CFG_SRC}"; exit 1; }
cp "$SWEEP_CFG_SRC"/*.json "$CFG_DIR"/

for cfg in "$CFG_DIR"/??_*.json; do
  base_name="$(basename "$cfg" .json)"
  run_one "$base_name" "$cfg" "${RES_DIR}/${base_name}.json"
done

log "=== dim14-full LQB lambda=1.5 sweep complete ==="
