#!/usr/bin/env bash
# Sweep the same 4 skew configs as atomic21, using legacy dim14 CE-GNN + dim22 MLP.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/dim14_old_models_skew4_$(date +%Y%m%d)}"
TS="$(basename "$OUT_ROOT" | grep -oE '[0-9]{8}(_[0-9]{6})?' | tail -1 || date +%Y%m%d_%H%M%S)"
CFG_DIR="${OUT_ROOT}/configs"
RES_DIR="${OUT_ROOT}/results"
PROGRESS_LOG="logs/dim14_old_models_skew4_sweep_${TS}.log"
mkdir -p "$CFG_DIR" "$RES_DIR" logs

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
BASE_CONFIG="simulation_data/space_with_network.json"
SPARSE_UNIFORM="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
SWEEP_CFG_SRC="simulation_data/normal_sim_sweeps/atomic21_skew_configs"
GNN_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp.pt"
TIMEOUT=3600
SEED=42

export GNN_DECODE_MODE=argmax
export INFERENCE_FEATURE_LAYOUT=dim22
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*" | tee -a "$PROGRESS_LOG"
}

log "=== dim14 old models skew-4 sweep start ==="
log "GNN: ${GNN_MODEL}"
log "MLP: ${MLP_MODEL}"
log "Output: ${OUT_ROOT}"

if [[ ! -f "$GNN_MODEL" ]]; then
  log "ERROR: GNN model missing: ${GNN_MODEL}"
  exit 1
fi
if [[ ! -f "$MLP_MODEL" ]]; then
  log "ERROR: MLP model missing: ${MLP_MODEL}"
  exit 1
fi

run_gnn() {
  local name="$1"
  local config="$2"
  local output="$3"
  if [[ -f "$output" ]]; then
    log "GNN ${name} SKIP (exists)"
    return 0
  fi
  log "GNN Running ${name}"
  local start elapsed rtt
  start=$(date +%s)
  export GNN_MODEL_PATH="${GNN_MODEL}"
  if pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn \
    --config "$config" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"; then
    elapsed=$(( $(date +%s) - start ))
    rtt=$(pipenv run python3 -c "import json; print(json.load(open('${output}'))['total_rtt'])" 2>/dev/null || echo "NA")
    log "GNN ${name} SUCCESS ${elapsed}s total_rtt=${rtt}"
  else
    elapsed=$(( $(date +%s) - start ))
    log "GNN ${name} FAILED ${elapsed}s"
    return 1
  fi
}

run_mlp() {
  local name="$1"
  local config="$2"
  local output="$3"
  if [[ -f "$output" ]]; then
    log "MLP ${name} SKIP (exists)"
    return 0
  fi
  log "MLP Running ${name}"
  local start elapsed rtt
  start=$(date +%s)
  export MLP_MODEL_PATH="${MLP_MODEL}"
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
    log "MLP ${name} SUCCESS ${elapsed}s total_rtt=${rtt}"
  else
    elapsed=$(( $(date +%s) - start ))
    log "MLP ${name} FAILED ${elapsed}s"
    return 1
  fi
}

cp "$SWEEP_CFG_SRC"/*.json "$CFG_DIR"/ 2>/dev/null || true

CONFIGS=(
  "default_20_20_p50|${BASE_CONFIG}"
  "05_sparse_40_40_p25|${SPARSE_UNIFORM}"
  "default_20_20_degree_skew|${CFG_DIR}/default_20_20_degree_skew.json"
  "05_sparse_40_40_p25_degree_skew|${CFG_DIR}/05_sparse_40_40_p25_degree_skew.json"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  config="${entry#*|}"
  if [[ ! -f "$config" ]]; then
    log "ERROR: config missing for ${name}: ${config}"
    exit 1
  fi
  run_gnn "$name" "$config" "${RES_DIR}/${name}_gnn_dim14.json"
  run_mlp "$name" "$config" "${RES_DIR}/${name}_mlp_dim22.json"
done

log "=== dim14 old models skew-4 sweep complete ==="
RES_DIR="${RES_DIR}" pipenv run python3 - <<'PY' | tee -a "$PROGRESS_LOG"
import json
import os
from pathlib import Path

res = Path(os.environ["RES_DIR"])
configs = [
    "default_20_20_p50",
    "05_sparse_40_40_p25",
    "default_20_20_degree_skew",
    "05_sparse_40_40_p25_degree_skew",
]

def load_rtt(path):
    if not path.exists():
        return None
    rtt = json.loads(path.read_text()).get("total_rtt")
    return float(rtt) if rtt else None

print("\nSummary (dim14 CE-GNN vs legacy MLP):")
print(f"{'Config':<40} {'GNN RTT':>14} {'MLP RTT':>14} {'MLP-GNN %':>12}")
print("-" * 84)
for name in configs:
    gnn = load_rtt(res / f"{name}_gnn_dim14.json")
    mlp = load_rtt(res / f"{name}_mlp_dim22.json")
    gnn_s = f"{gnn:,.0f}" if gnn is not None else "MISSING"
    mlp_s = f"{mlp:,.0f}" if mlp is not None else "MISSING"
    if gnn is not None and mlp is not None and mlp > 0:
        delta = 100.0 * (mlp - gnn) / mlp
        delta_s = f"{delta:+.1f}%"
    else:
        delta_s = "—"
    marker = " <<" if "degree_skew" in name else ""
    print(f"{name:<40} {gnn_s:>14} {mlp_s:>14} {delta_s:>12}{marker}")
PY
