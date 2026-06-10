#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_atomic21_ce_only_$(date +%Y%m%d)}"
TS="$(basename "$OUT_ROOT" | grep -oE '[0-9]{8}(_[0-9]{6})?' | tail -1 || date +%Y%m%d_%H%M%S)"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/mlp_atomic21_skew_sweep_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/atomic21_skew_configs"
SPARSE_UNIFORM="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
MLP_MODEL="models/tabular/batch_edge_mlp_atomic21.pt"
TIMEOUT=3600
SEED=42

export GNN_DECODE_MODE=argmax
export MLP_MODEL_PATH="${MLP_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*"
}

log "=== MLP atomic21 skew sweep start ==="
log "Model: ${MLP_MODEL}"
log "Output dir: ${RES_DIR}"
log "Progress log: ${PROGRESS_LOG}"

if [[ ! -f "$MLP_MODEL" ]]; then
  log "ERROR: MLP model missing: ${MLP_MODEL}"
  exit 1
fi

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
  local start elapsed rtt
  start=$(date +%s)
  if pipenv run python3 scripts_cosim/run_simulation.py \
    --mlp_batch \
    --mlp-model "$MLP_MODEL" \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"; then
    elapsed=$(( $(date +%s) - start ))
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${output}'))['total_rtt'])" 2>/dev/null || echo "NA")
    log "${name} SUCCESS ${elapsed}s total_rtt=${rtt}"
    echo "${name} SUCCESS ${elapsed}s total_rtt=${rtt}" >> "$PROGRESS_LOG"
  else
    elapsed=$(( $(date +%s) - start ))
    log "${name} FAILED ${elapsed}s"
    echo "${name} FAILED ${elapsed}s" >> "$PROGRESS_LOG"
    return 1
  fi
}

run_one "default_20_20_p50" "$BASE_CONFIG" "${RES_DIR}/default_20_20_p50_mlp_atomic21.json"
run_one "05_sparse_40_40_p25" "$SPARSE_UNIFORM" "${RES_DIR}/05_sparse_40_40_p25_mlp_atomic21.json"

if [[ ! -d "$SWEEP_CFG_SRC" ]]; then
  log "ERROR: skew config source missing: ${SWEEP_CFG_SRC}"
  exit 1
fi
cp "$SWEEP_CFG_SRC"/*.json "$CFG_DIR"/ 2>/dev/null || true

for cfg in "$CFG_DIR"/*.json; do
  [[ -f "$cfg" ]] || continue
  base_name="$(basename "$cfg" .json)"
  run_one "$base_name" "$cfg" "${RES_DIR}/${base_name}_mlp_atomic21.json"
done

log "=== MLP atomic21 skew sweep complete ==="
