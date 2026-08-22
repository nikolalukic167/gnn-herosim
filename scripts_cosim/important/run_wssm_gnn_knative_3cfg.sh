#!/usr/bin/env bash
# Knative vs wssm GNN on 3 configs (6 runs), ql=100 default autoscale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/wssm_kn_gnn_3cfg_20260614}"
RES_DIR="${SWEEP_DIR}/results"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "$RES_DIR" "$LOG_DIR"

SEED="${SEED:-42}"
QL="${QUEUE_LENGTH:-100}"
TIMEOUT_DEFAULT="${TIMEOUT_DEFAULT:-3600}"
TIMEOUT_HUB="${TIMEOUT_HUB:-7200}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt}"

export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export INFERENCE_FEATURE_LAYOUT=dim22
export GNN_MODEL_PATH="$GNN_MODEL"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${SWEEP_DIR}/progress.log"; }

run_one() {
  local cfg_name="$1"
  local config_path="$2"
  local workload_path="$3"
  local policy="$4"
  local timeout="$5"

  local suffix="gnn_wssm"
  local flag="--gnn"
  if [[ "$policy" == "knative_network" ]]; then
    suffix="knative"
    flag="--knative_network"
    unset INFERENCE_FEATURE_LAYOUT
  else
    export INFERENCE_FEATURE_LAYOUT=dim22
  fi

  local out="${RES_DIR}/${cfg_name}__${suffix}.json"
  if [[ -f "$out" ]]; then
    local rtt
    rtt=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('total_rtt',''))" 2>/dev/null || true)
    if [[ -n "$rtt" && "$rtt" != "None" && "$rtt" != "0" ]]; then
      log "SKIP ${cfg_name} ${suffix} (rtt=${rtt})"
      return 0
    fi
  fi

  if [[ ! -f "$GNN_MODEL" && "$policy" == "gnn" ]]; then
    log "ERROR missing GNN model: $GNN_MODEL"
    exit 1
  fi

  log "RUN ${cfg_name} ${suffix} ql=${QL}"
  local run_log="${LOG_DIR}/${cfg_name}__${suffix}.log"

  ${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py \
    $flag \
    --config "$config_path" \
    --workload "$workload_path" \
    --output "$out" \
    --seed "$SEED" \
    --queue-length "$QL" \
    --timeout "$timeout" \
    2>&1 | tee "$run_log" || {
      log "FAILED ${cfg_name} ${suffix}"
      return 1
    }
}

log "=== wssm GNN vs Knative 3-config (ql=${QL}) ==="
log "GNN_MODEL=$GNN_MODEL HEROSIM_WARMTH_PHYSICS=node_disk_v2"

declare -a JOBS=(
  "triangle_default|simulation_data/space_with_network.json|data/nofs-ids/traces/workload-100-100.json|${TIMEOUT_DEFAULT}"
  "degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json|data/nofs-ids/traces/workload-100-100.json|${TIMEOUT_DEFAULT}"
  "hub_k6_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k6_seek50.json|data/nofs-ids/traces/workload-125-225.json|${TIMEOUT_HUB}"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r cfg_name config_path workload_path timeout <<< "$job"
  [[ -f "$config_path" ]] || { log "ERROR missing $config_path"; exit 1; }
  [[ -f "$workload_path" ]] || { log "ERROR missing $workload_path"; exit 1; }
  run_one "$cfg_name" "$config_path" "$workload_path" "gnn" "$timeout"
  run_one "$cfg_name" "$config_path" "$workload_path" "knative_network" "$timeout"
done

log "=== Summary ==="
${HEROSIM_PY:-pipenv run python3} - <<'PY' | tee -a "${SWEEP_DIR}/progress.log"
import json
from pathlib import Path

res = Path("simulation_data/normal_sim_sweeps/wssm_kn_gnn_3cfg_20260614/results")
rows = []
for p in sorted(res.glob("*.json")):
    if "decode_stats" in p.name:
        continue
    d = json.loads(p.read_text())
    name = p.stem
    rows.append((name, d.get("total_rtt"), d.get("stats", {}).get("averageQueueTime")))

print(f"\n{'config':<40} {'RTT':>12} {'avgQ':>8}")
print("-" * 62)
for name, rtt, aq in rows:
    rtt_s = f"{rtt/1e6:.3f}M" if rtt else "—"
    aq_s = f"{aq:.2f}s" if aq is not None else "—"
    print(f"{name:<40} {rtt_s:>12} {aq_s:>8}")

by_cfg = {}
for name, rtt, _ in rows:
    if "__" not in name:
        continue
    cfg, pol = name.rsplit("__", 1)
    by_cfg.setdefault(cfg, {})[pol] = rtt

print("\nWinner (lower RTT):")
for cfg in sorted(by_cfg):
    g = by_cfg[cfg].get("gnn_wssm")
    k = by_cfg[cfg].get("knative")
    if g is None or k is None:
        continue
    w = "GNN" if g < k else "Kn"
    pct = (g - k) / k * 100
    print(f"  {cfg}: {w}  GvsKn {pct:+.1f}%")
PY

log "Done. Results: ${RES_DIR}/"
