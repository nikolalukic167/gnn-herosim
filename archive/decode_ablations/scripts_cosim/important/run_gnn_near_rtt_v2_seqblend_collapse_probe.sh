#!/usr/bin/env bash
# Step-0 probe: does queue-aware seqblend decode fix baseline collapse on the
# two small-topology configs (default, 00) using the EXISTING v2 model?
# No retraining. Sweeps GNN_SEQBLEND_QUEUE_MARGIN in {1,2,3}.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_seqblend_probe_20260607"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/gnn_near_rtt_v2_seqblend_probe.log"
mkdir -p "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
MODEL_PATH="models/near-rtt-v2-ssc-trash-1.pt"
TIMEOUT=3600
SEED=42

DEFAULT_CFG="simulation_data/space_with_network.json"
CFG00="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"

export GNN_MODEL_PATH="${MODEL_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GNN_DECODE_MODE="seqblend"

log() { echo "[$(date -Is)] $*"; }

run_one() {
  local name="$1" config="$2" margin="$3"
  local output="${RES_DIR}/${name}_seqblend_m${margin}.json"
  if [[ -f "$output" ]]; then
    log "${name} m${margin} SKIP (exists)"
    return 0
  fi
  log "Running ${name} seqblend margin=${margin}"
  local start; start=$(date +%s)
  if GNN_SEQBLEND_QUEUE_MARGIN="${margin}" pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn --config "$config" --workload "$WORKLOAD" \
    --output "$output" --timeout "$TIMEOUT" --seed "$SEED"; then
    local elapsed=$(( $(date +%s) - start ))
    local rtt; rtt=$(python3 -c "import json;print(json.load(open('$output'))['total_rtt'])" 2>/dev/null || echo "NA")
    log "${name} m${margin} SUCCESS ${elapsed}s total_rtt=${rtt}"
    echo "${name} m${margin} SUCCESS ${elapsed}s total_rtt=${rtt}" >> "$PROGRESS_LOG"
  else
    log "${name} m${margin} FAILED"
    echo "${name} m${margin} FAILED" >> "$PROGRESS_LOG"
  fi
}

log "=== seqblend collapse probe start (model=${MODEL_PATH}) ==="
for margin in 1 2 3; do
  run_one "default_20_20_p50" "$DEFAULT_CFG" "$margin"
  run_one "00_balanced_30_30_p35" "$CFG00" "$margin"
done
log "=== seqblend collapse probe complete ==="
