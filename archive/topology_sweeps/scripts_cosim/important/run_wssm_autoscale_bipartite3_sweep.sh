#!/usr/bin/env bash
# wssm GNN vs Knative on 3 bipartite hub configs × 3 queue_lengths (18 runs).
# Same stack as bipartite_v2_skew_merged: node_disk_v2, dim22 GNN, ql sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/wssm_autoscale_bipartite3_20260614}"
RES_DIR="${SWEEP_DIR}/results"
LOG_DIR="${SWEEP_DIR}/logs"
CFG_DIR="${CFG_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"

mkdir -p "$RES_DIR" "$LOG_DIR"

SEED="${SEED:-42}"
BASE_QL="${BASE_QL:-100}"
TIMEOUT="${TIMEOUT:-7200}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt}"

# 3 of 9 bipartite v2 grid: k4/k6/k8 @ seek50 (Kn only wins k6_seek50 in v2 wssm gate)
CONFIGS="${CONFIGS:-hub_k4_seek50 hub_k6_seek50 hub_k8_seek50}"
MULTS=(0.5 1.0 2.0)

export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export GNN_MODEL_PATH="$GNN_MODEL"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${SWEEP_DIR}/progress.log"; }

run_one() {
  local cfg_name="$1"
  local ql="$2"
  local policy="$3"

  local config_path="${CFG_DIR}/${cfg_name}.json"
  [[ -f "$config_path" ]] || { log "ERROR missing $config_path"; exit 1; }

  local suffix flag
  if [[ "$policy" == "gnn" ]]; then
    suffix="gnn_wssm"
    flag="--gnn"
    export INFERENCE_FEATURE_LAYOUT=dim22
  else
    suffix="knative"
    flag="--knative_network"
    unset INFERENCE_FEATURE_LAYOUT 2>/dev/null || true
  fi

  local out="${RES_DIR}/${cfg_name}__${suffix}__ql${ql}.json"
  if [[ -f "$out" ]]; then
    local rtt
    rtt=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('total_rtt',''))" 2>/dev/null || true)
    if [[ -n "$rtt" && "$rtt" != "None" && "$rtt" != "0" ]]; then
      log "SKIP ${cfg_name} ${suffix} ql=${ql} (rtt=${rtt})"
      return 0
    fi
  fi

  log "RUN ${cfg_name} ${suffix} ql=${ql}"
  local run_log="${LOG_DIR}/${cfg_name}__${suffix}__ql${ql}.log"

  pipenv run python3 scripts_cosim/run_simulation.py \
    $flag \
    --config "$config_path" \
    --workload "$WORKLOAD" \
    --output "$out" \
    --seed "$SEED" \
    --queue-length "$ql" \
    --timeout "$TIMEOUT" \
    2>&1 | tee "$run_log" || {
      log "FAILED ${cfg_name} ${suffix} ql=${ql}"
      return 1
    }
}

log "=== wssm GNN vs Knative · bipartite 3-config × ql sweep ==="
log "SWEEP_DIR=$SWEEP_DIR CONFIGS=$CONFIGS BASE_QL=$BASE_QL mults=${MULTS[*]}"
log "GNN_MODEL=$GNN_MODEL HEROSIM_WARMTH_PHYSICS=node_disk_v2 WORKLOAD=$WORKLOAD"

[[ -f "$GNN_MODEL" ]] || { log "ERROR missing $GNN_MODEL"; exit 1; }
[[ -f "$WORKLOAD" ]] || { log "ERROR missing $WORKLOAD"; exit 1; }

for cfg_name in $CONFIGS; do
  for mult in "${MULTS[@]}"; do
    ql=$(python3 -c "print(int(round(${BASE_QL} * ${mult})))")
    run_one "$cfg_name" "$ql" "gnn"
    run_one "$cfg_name" "$ql" "knative"
  done
done

log "=== Compare ==="
pipenv run python3 scripts_cosim/important/compare_wssm_autoscale_bipartite3.py \
  --sweep-dir "$SWEEP_DIR" | tee -a "${SWEEP_DIR}/progress.log"

log "Done. Results: ${RES_DIR}/"
