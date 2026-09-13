#!/usr/bin/env bash
# One skew4 new-models job (node_disk_v2, 125-225 workload).
# Usage: run_skew4_new_models_one.sh <gnn|mlp|knative> <config_name> <config_json_path>
#
# GNN: near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt  (dim22 layout)
# MLP: batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt  (ce_reduced layout)
# Knative: per-arrival knative_network with node_disk_v2 physics (new: previous skew4 used legacy physics)
# Workload: workload-125-225.json (562k tasks, skew stress)
# Configs: default_20_20_p50 | 05_sparse_40_40_p25 | default_20_20_degree_skew | 05_sparse_40_40_p25_degree_skew
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?gnn, mlp, or knative}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt}"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/skew4_new_models_20260614}/results"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-7200}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
unset INFERENCE_FEATURE_LAYOUT 2>/dev/null || true
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

MODEL_CHECK=""
case "$POLICY" in
  gnn)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_wssm.json"
    export GNN_MODEL_PATH="${GNN_MODEL}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$GNN_MODEL"
    RUN_ARGS=(--gnn)
    ;;
  mlp)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_wssm.json"
    export MLP_MODEL_PATH="${MLP_MODEL}"
    export INFERENCE_FEATURE_LAYOUT=ce_reduced
    MODEL_CHECK="$MLP_MODEL"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_MODEL")
    ;;
  knative)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_knative.json"
    RUN_ARGS=(--knative_network)
    ;;
  *)
    echo "ERROR: unknown policy: ${POLICY}" >&2
    exit 1
    ;;
esac

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config missing: $CONFIG_PATH" >&2; exit 1; }
[[ -z "$MODEL_CHECK" || -f "$MODEL_CHECK" ]] || { echo "ERROR: model missing: $MODEL_CHECK" >&2; exit 1; }

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

echo "=== skew4 new models (${HEROSIM_WARMTH_PHYSICS}): ${POLICY} / ${CONFIG_NAME} ==="
[[ -n "$MODEL_CHECK" ]] && echo "  MODEL=${MODEL_CHECK}"
echo "  WORKLOAD=${WORKLOAD}"

python3 scripts_cosim/run_simulation.py \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  "${RUN_ARGS[@]}"

rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt','?'))" 2>/dev/null || echo "?")
echo "DONE: $OUTPUT  total_rtt=${rtt}"
