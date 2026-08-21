#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 OUT_ROOT MODEL_PATH LABEL DECODE_MODE [TOP_K]" >&2
  exit 2
fi

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="$1"
MODEL_PATH="$2"
LABEL="$3"
DECODE_MODE="$4"
TOP_K="${5:-10}"

TS="$(basename "$OUT_ROOT" | grep -oE '[0-9]{8}(_[0-9]{6})?' | tail -1 || date +%Y%m%d_%H%M%S)"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/${LABEL}_5cfg_${DECODE_MODE}_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
TIMEOUT="${TIMEOUT:-3600}"
SEED="${SEED:-42}"

export GNN_MODEL_PATH="${MODEL_PATH}"
export GNN_DECODE_MODE="${DECODE_MODE}"
export GNN_DECODE_TOP_K="${TOP_K}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*"
}

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
  if ${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py \
    --gnn \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"; then
    elapsed=$(( $(date +%s) - start ))
    rtt=$(${HEROSIM_PY:-pipenv run python3} -c "import json; print(json.load(open('${output}'))['total_rtt'])" 2>/dev/null || echo "NA")
    log "${name} SUCCESS ${elapsed}s total_rtt=${rtt}"
    echo "${name} SUCCESS ${elapsed}s total_rtt=${rtt}" >> "$PROGRESS_LOG"
  else
    elapsed=$(( $(date +%s) - start ))
    log "${name} FAILED ${elapsed}s"
    echo "${name} FAILED ${elapsed}s" >> "$PROGRESS_LOG"
  fi
}

log "=== ${LABEL} 5-config sweep start ==="
log "Model: ${MODEL_PATH}"
log "Decode: ${DECODE_MODE} (top_k=${TOP_K})"
log "Output: ${OUT_ROOT}"
log "Progress log: ${PROGRESS_LOG}"

if [[ ! -d "$SWEEP_CFG_SRC" ]]; then
  log "ERROR: sweep config source missing: ${SWEEP_CFG_SRC}"
  exit 1
fi

for cfg_name in \
  "01_balanced_40_40_p50" \
  "02_balanced_50_50_p60" \
  "03_client_heavy_50_35_p50" \
  "05_sparse_40_40_p25"; do
  cp "${SWEEP_CFG_SRC}/${cfg_name}.json" "$CFG_DIR/"
done

run_one "default_20_20_p50" "$BASE_CONFIG" "${RES_DIR}/default_20_20_p50.json"
run_one "01_balanced_40_40_p50" "${CFG_DIR}/01_balanced_40_40_p50.json" "${RES_DIR}/01_balanced_40_40_p50.json"
run_one "02_balanced_50_50_p60" "${CFG_DIR}/02_balanced_50_50_p60.json" "${RES_DIR}/02_balanced_50_50_p60.json"
run_one "03_client_heavy_50_35_p50" "${CFG_DIR}/03_client_heavy_50_35_p50.json" "${RES_DIR}/03_client_heavy_50_35_p50.json"
run_one "05_sparse_40_40_p25" "${CFG_DIR}/05_sparse_40_40_p25.json" "${RES_DIR}/05_sparse_40_40_p25.json"

log "=== ${LABEL} 5-config sweep complete ==="
