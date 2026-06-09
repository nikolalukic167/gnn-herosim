#!/usr/bin/env bash
# Full 7-config Regime A triangle legs: xgboost_batch (ranking) + mlp_batch.
# Workload: workload-100-100.json · seed 42 · argmax decode (default).
# Skips configs whose result JSON already exists for this workload.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/reviewer_triangle_all7_$(date +%Y%m%d)}"
TS="$(basename "$OUT_ROOT" | grep -oE '[0-9]{8}(_[0-9]{6})?' | tail -1 || date +%Y%m%d_%H%M%S)"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
SWEEP_LOG="${OUT_ROOT}/sweep.log"
PROGRESS_LOG="logs/reviewer_triangle_all7_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
XGB_MODEL="models/tabular/batch_edge_ranker.json"
MLP_MODEL="models/tabular/batch_edge_mlp.pt"
SEED=42
XGB_TIMEOUT=7200
MLP_TIMEOUT=3600

export GNN_DECODE_MODE=argmax
export XGB_MODEL_PATH="${XGB_MODEL}"
export MLP_MODEL_PATH="${MLP_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*" | tee -a "$SWEEP_LOG" "$PROGRESS_LOG"
}

result_done() {
  local output="$1"
  if [[ ! -f "$output" ]]; then
    return 1
  fi
  pipenv run python3 -c "
import json, sys
from pathlib import Path
p = Path('$output')
try:
    d = json.load(open(p))
except Exception:
    sys.exit(1)
wl = str(d.get('workload_file', ''))
if 'workload-100-100.json' not in wl:
    sys.exit(1)
rtt = d.get('total_rtt')
if rtt is None or not isinstance(rtt, (int, float)) or rtt <= 0:
    sys.exit(1)
"
}

run_policy_config() {
  local policy="$1"
  local name="$2"
  local config="$3"
  local output="$4"
  local timeout="$5"
  local extra_args=("${@:6}")

  if result_done "$output"; then
    local rtt
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${output}'))['total_rtt'])" 2>/dev/null || echo "NA")
    log "${policy} ${name} SKIP (exists, workload-100-100) total_rtt=${rtt}"
    echo "${policy} ${name} SKIP total_rtt=${rtt}" >> "$PROGRESS_LOG"
    return 0
  fi

  log "Running ${policy} ${name} -> ${output}"
  local start elapsed rtt ec=0
  start=$(date +%s)
  if pipenv run python3 scripts_cosim/run_simulation.py \
    --"${policy}" \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$timeout" \
    --seed "$SEED" \
    "${extra_args[@]}"; then
    :
  else
    ec=$?
  fi
  elapsed=$(( $(date +%s) - start ))
  if [[ $ec -eq 0 ]] && result_done "$output"; then
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${output}'))['total_rtt'])" 2>/dev/null || echo "NA")
    log "${policy} ${name} SUCCESS ${elapsed}s total_rtt=${rtt}"
    echo "${policy} ${name} SUCCESS ${elapsed}s total_rtt=${rtt}" >> "$PROGRESS_LOG"
  else
    log "${policy} ${name} FAILED exit=${ec} ${elapsed}s"
    echo "${policy} ${name} FAILED exit=${ec} ${elapsed}s" >> "$PROGRESS_LOG"
    return 1
  fi
}

log "=== Reviewer Triangle Regime A all7 sweep start ==="
log "Output: ${OUT_ROOT}"
log "Workload: ${WORKLOAD} · seed ${SEED} · decode ${GNN_DECODE_MODE}"
log "XGB model: ${XGB_MODEL} · timeout ${XGB_TIMEOUT}s"
log "MLP model: ${MLP_MODEL} · timeout ${MLP_TIMEOUT}s"

if [[ ! -f "$XGB_MODEL" ]]; then
  log "ERROR: XGB model missing: ${XGB_MODEL}"
  exit 1
fi
if [[ ! -f "$MLP_MODEL" ]]; then
  log "ERROR: MLP model missing: ${MLP_MODEL}"
  exit 1
fi
if [[ ! -d "$SWEEP_CFG_SRC" ]]; then
  log "ERROR: sweep config source missing: ${SWEEP_CFG_SRC}"
  exit 1
fi

cp "$SWEEP_CFG_SRC"/*.json "$CFG_DIR"/

CONFIGS=("default_20_20_p50:${BASE_CONFIG}")
for cfg in "$CFG_DIR"/??_*.json; do
  base_name="$(basename "$cfg" .json)"
  CONFIGS+=("${base_name}:${cfg}")
done

# Sequential: MLP first (GPU, fast), then XGB (CPU, slow) — avoids GPU contention during MLP leg.
for entry in "${CONFIGS[@]}"; do
  name="${entry%%:*}"
  config="${entry#*:}"
  run_policy_config "mlp_batch" "$name" "$config" "${RES_DIR}/${name}_mlp_batch.json" "$MLP_TIMEOUT" \
    --mlp-model "$MLP_MODEL"
done

for entry in "${CONFIGS[@]}"; do
  name="${entry%%:*}"
  config="${entry#*:}"
  run_policy_config "xgboost_batch" "$name" "$config" "${RES_DIR}/${name}_xgboost_batch.json" "$XGB_TIMEOUT" \
    --xgb-model "$XGB_MODEL"
done

log "=== Reviewer Triangle Regime A all7 sweep complete ==="
