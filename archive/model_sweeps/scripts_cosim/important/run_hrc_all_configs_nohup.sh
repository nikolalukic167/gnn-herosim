#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="simulation_data/normal_sim_sweeps/herocache_network_${TS}"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/herocache_sweep_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
TIMEOUT=1800
SEED=42

log() {
  echo "[$(date -Is)] $*"
}

log "=== HeroCache (hrc) sweep start ==="
log "Output: ${OUT_ROOT}"
log "Progress log: ${PROGRESS_LOG}"

run_one() {
  local name="$1"
  local config="$2"
  local output="$3"
  log "Running ${name}"
  local start elapsed
  start=$(date +%s)
  if pipenv run python3 scripts_cosim/run_simulation.py \
    --herocache_network \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"; then
    elapsed=$(( $(date +%s) - start ))
    log "${name} SUCCESS ${elapsed}s"
    echo "${name} SUCCESS ${elapsed}s" >> "$PROGRESS_LOG"
  else
    elapsed=$(( $(date +%s) - start ))
    log "${name} FAILED ${elapsed}s"
    echo "${name} FAILED ${elapsed}s" >> "$PROGRESS_LOG"
  fi
}

# Baseline default config
run_one "default_20_20_p50" "$BASE_CONFIG" "${RES_DIR}/default_20_20_p50.json"

# Six sweep configs
if [[ ! -d "$SWEEP_CFG_SRC" ]]; then
  log "ERROR: sweep config source missing: ${SWEEP_CFG_SRC}"
  exit 1
fi
cp "$SWEEP_CFG_SRC"/*.json "$CFG_DIR"/

for cfg in "$CFG_DIR"/??_*.json; do
  base_name="$(basename "$cfg" .json)"
  run_one "$base_name" "$cfg" "${RES_DIR}/${base_name}.json"
done

log "=== HeroCache (hrc) sweep complete ==="
