#!/usr/bin/env bash
# Minimal autoscale sensitivity: vary target_concurrency (queue_length) only.
# Policies: gnn + knative_network (matched Knative-family autoscaler).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/autoscale_sensitivity_$(date +%Y%m%d)}"
RES_DIR="${SWEEP_DIR}/results"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "$RES_DIR" "$LOG_DIR"

SEED="${SEED:-42}"
BASE_QL="${BASE_QL:-100}"
TIMEOUT_DEFAULT="${TIMEOUT_DEFAULT:-3600}"
TIMEOUT_HUB="${TIMEOUT_HUB:-7200}"
GNN_MODEL="${GNN_MODEL_PATH:-models/near-rtt-v2-dim14-ce-only.pt}"

# Multipliers × baseline target concurrency (Knative autoscale knob)
MULTS=(0.5 1.0 2.0)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${SWEEP_DIR}/progress.log"; }

run_one() {
  local cfg_name="$1"
  local config_path="$2"
  local workload_path="$3"
  local policy="$4"
  local ql="$5"
  local timeout="$6"

  local out="${RES_DIR}/${cfg_name}__${policy}__ql${ql}.json"
  if [[ -f "$out" ]]; then
    local rtt
    rtt=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('total_rtt',''))" 2>/dev/null || true)
    if [[ -n "$rtt" && "$rtt" != "None" ]]; then
      log "SKIP ${cfg_name} ${policy} ql=${ql} (exists rtt=${rtt})"
      return 0
    fi
  fi

  log "RUN  ${cfg_name} ${policy} ql=${ql} config=${config_path}"
  export GNN_MODEL_PATH="$GNN_MODEL"
  export GNN_DECODE_MODE=argmax
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  local run_log="${LOG_DIR}/${cfg_name}__${policy}__ql${ql}.log"
  local flag=""
  case "$policy" in
    gnn) flag="--gnn" ;;
    knative_network) flag="--knative_network" ;;
    *) log "ERROR unknown policy $policy"; return 1 ;;
  esac

  ${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py \
    $flag \
    --config "$config_path" \
    --workload "$workload_path" \
    --output "$out" \
    --seed "$SEED" \
    --queue-length "$ql" \
    --timeout "$timeout" \
    2>&1 | tee "$run_log" || {
      log "FAILED ${cfg_name} ${policy} ql=${ql} (see ${run_log})"
      return 1
    }
}

log "=== Autoscale sensitivity sweep ==="
log "SWEEP_DIR=$SWEEP_DIR BASE_QL=$BASE_QL mults=${MULTS[*]} seed=$SEED"

# Config grid: name | config | workload | timeout
declare -a JOBS=(
  "triangle_default|simulation_data/space_with_network.json|data/nofs-ids/traces/workload-100-100.json|${TIMEOUT_DEFAULT}"
  "degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json|data/nofs-ids/traces/workload-100-100.json|${TIMEOUT_DEFAULT}"
  "hub_k6_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k6_seek50.json|data/nofs-ids/traces/workload-125-225.json|${TIMEOUT_HUB}"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r cfg_name config_path workload_path timeout <<< "$job"
  if [[ ! -f "$config_path" ]]; then
    log "ERROR missing config: $config_path"
    exit 1
  fi
  if [[ ! -f "$workload_path" ]]; then
    log "ERROR missing workload: $workload_path"
    exit 1
  fi
  for mult in "${MULTS[@]}"; do
    ql=$(python3 -c "print(int(round(${BASE_QL} * ${mult})))")
    for policy in gnn knative_network; do
      run_one "$cfg_name" "$config_path" "$workload_path" "$policy" "$ql" "$timeout"
    done
  done
done

log "=== Compare ==="
${HEROSIM_PY:-pipenv run python3} scripts_cosim/important/compare_autoscale_sensitivity.py \
  --sweep-dir "$SWEEP_DIR" | tee -a "${SWEEP_DIR}/progress.log"

log "Done. Results: ${RES_DIR}/"
