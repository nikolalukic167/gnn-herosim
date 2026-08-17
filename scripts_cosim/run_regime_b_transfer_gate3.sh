#!/usr/bin/env bash
# Transfer /compare: Regime B distill ckpt on real 100-100 gate3.
# Expect a loss vs Kn / 873. Needed so the paper can say so.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SWEEP="simulation_data/normal_sim_sweeps/regime_b_transfer_gate3_20260813"
CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
DISTILL="models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill-multiseed.pt"
CE873="models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt"
SEED="${SEED:-42}"
TIMEOUT_KN="${TIMEOUT_KN:-3600}"
TIMEOUT_GNN="${TIMEOUT_GNN:-7200}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/regime_b_transfer_gate3_${TS}.log"
mkdir -p "$SWEEP/results" "$SWEEP/configs" logs

cp -f "$CFG_SRC/01_balanced_40_40_p50.json" "$SWEEP/configs/"
cp -f "$CFG_SRC/05_sparse_40_40_p25.json" "$SWEEP/configs/"

declare -A CONFIGS=(
  [default_20_20_p50]="simulation_data/space_with_network.json"
  [01_balanced_40_40_p50]="$SWEEP/configs/01_balanced_40_40_p50.json"
  [05_sparse_40_40_p25]="$SWEEP/configs/05_sparse_40_40_p25.json"
)

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

run_one() {
  local name="$1" policy_flag="$2" config="$3" output="$4" timeout="$5"
  if [[ -f "$output" ]]; then
    local rtt
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${output}')).get('total_rtt','NA'))" 2>/dev/null || echo NA)
    if [[ "$rtt" != "NA" && "$rtt" != "None" && "$rtt" != "0" ]]; then
      log "${name} SKIP (exists total_rtt=${rtt})"
      return 0
    fi
    log "${name} RE-RUN (exists but total_rtt missing/zero)"
  fi
  log "START ${name}"
  local start elapsed
  start=$(date +%s)
  if pipenv run python3 scripts_cosim/run_simulation.py \
    $policy_flag \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$timeout" \
    --seed "$SEED"; then
    elapsed=$(( $(date +%s) - start ))
    local rtt
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${output}')).get('total_rtt','NA'))")
    log "${name} SUCCESS ${elapsed}s total_rtt=${rtt}"
  else
    elapsed=$(( $(date +%s) - start ))
    log "${name} FAILED ${elapsed}s"
    return 1
  fi
}

log "=== Regime B transfer gate3 ${TS} ==="
log "sweep=${SWEEP} seed=${SEED} workload=100-100"

# Order: cheap baselines first, distill last (seq_reforward_pull is slow).
for cfg in default_20_20_p50 01_balanced_40_40_p50 05_sparse_40_40_p25; do
  config="${CONFIGS[$cfg]}"
  [[ -f "$config" ]] || { log "ERROR missing config $config"; exit 1; }

  unset GNN_MODEL_PATH GNN_DECODE_MODE INFERENCE_FEATURE_LAYOUT || true
  run_one "${cfg}/knative" --knative_network "$config" \
    "$SWEEP/results/${cfg}_knative.json" "$TIMEOUT_KN"
  run_one "${cfg}/ect_pull" --knative_network_ect_pull "$config" \
    "$SWEEP/results/${cfg}_ect_pull.json" "$TIMEOUT_KN"

  export GNN_MODEL_PATH="$CE873"
  export GNN_DECODE_MODE="argmax"
  unset INFERENCE_FEATURE_LAYOUT || true
  run_one "${cfg}/gnn873_argmax" --gnn "$config" \
    "$SWEEP/results/${cfg}_gnn873_argmax.json" "$TIMEOUT_GNN"

  export GNN_MODEL_PATH="$DISTILL"
  export GNN_DECODE_MODE="seq_reforward_pull"
  export INFERENCE_FEATURE_LAYOUT="dim24"
  run_one "${cfg}/distill_seq_reforward_pull" --gnn "$config" \
    "$SWEEP/results/${cfg}_distill_seq_reforward_pull.json" "$TIMEOUT_GNN"
done

log "=== writing compare.json ==="
pipenv run python3 scripts_cosim/compare_regime_b_transfer_gate3.py --sweep-dir "$SWEEP"
log "=== transfer gate3 done ==="
