#!/usr/bin/env bash
# Fixed MLP bipartite v2 run — uses dim22 layout (is_warm + is_cold restored).
# Usage: run_bipartite_mlp_dim22_fix_one.sh <config_name> <config_json_path>
#
# MLP: batch_edge_mlp_warmth_sparse_skew_merged_dim22.pt  (dim22 layout, FIXED)
# Physics: node_disk_v2
# Workload: workload-125-225.json (562k tasks)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CONFIG_NAME="${1:?config name}"
CONFIG_PATH="${2:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22.pt}"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/bipartite_v2_mlp_dim22_fix_20260614}/results"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-7200}"

export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export MLP_MODEL_PATH="${MLP_MODEL}"
export INFERENCE_FEATURE_LAYOUT=dim22
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_wssm_dim22.json"

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config missing: $CONFIG_PATH" >&2; exit 1; }
[[ -f "$MLP_MODEL" ]] || { echo "ERROR: model missing: $MLP_MODEL" >&2; exit 1; }

if [[ -f "$OUTPUT" && "${FORCE_RERUN:-0}" != "1" ]]; then
  rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt', 0))" 2>/dev/null || echo 0)
  if [[ "${rtt}" != "0" && "${rtt}" != "0.0" ]]; then
    echo "SKIP (exists): $OUTPUT  total_rtt=${rtt}"
    exit 0
  fi
  echo "WARN: stale output ${OUTPUT} (total_rtt=${rtt}); re-running"
fi

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

echo "=== bipartite v2 MLP dim22 FIX (node_disk_v2): ${CONFIG_NAME} ==="
echo "  HEROSIM_WARMTH_PHYSICS=${HEROSIM_WARMTH_PHYSICS}"
echo "  INFERENCE_FEATURE_LAYOUT=${INFERENCE_FEATURE_LAYOUT}"
echo "  MODEL=${MLP_MODEL}"
echo "  WORKLOAD=${WORKLOAD}"

python3 scripts_cosim/run_simulation.py \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --mlp_batch --mlp-model "$MLP_MODEL"

rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt','?'))" 2>/dev/null || echo "?")
echo "DONE: $OUTPUT  total_rtt=${rtt}"
