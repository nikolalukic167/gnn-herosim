#!/usr/bin/env bash
# Weighted-merge contention live gate (node_disk_v2, workload-125-225).
# Usage: run_merged_contention_weighted_live_gate_one.sh <knative|mlp|gnn> <config_name> <config_json> [output_suffix]
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?knative, mlp, or gnn}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
OUT_SUFFIX="${4:-}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt}"
OUT_DIR="${SWEEP_DIR:?SWEEP_DIR required}/results"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-18000}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

case "$POLICY" in
  gnn)
    tag="${OUT_SUFFIX:-gnn}"
    export GNN_DECODE_MODE="${GNN_DECODE_MODE:-argmax_uniq}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    export GNN_MODEL_PATH="${GNN_MODEL}"
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_${tag}.json"
    RUN_ARGS=(--gnn)
    MODEL_CHECK="$GNN_MODEL"
    ;;
  mlp)
    tag="mlp_dim22"
    export MLP_MODEL_PATH="${MLP_MODEL}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_${tag}.json"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_MODEL")
    MODEL_CHECK="$MLP_MODEL"
    ;;
  knative)
    tag="knative"
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_${tag}.json"
    RUN_ARGS=(--knative_network)
    MODEL_CHECK=""
    ;;
  *)
    echo "ERROR: policy must be knative, mlp, or gnn" >&2
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

echo "=== weighted merged contention live gate: ${POLICY} / ${CONFIG_NAME} / ${tag} ==="
[[ "$POLICY" == "gnn" ]] && echo "  GNN_DECODE_MODE=${GNN_DECODE_MODE}"

python3 scripts_cosim/run_simulation.py \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  "${RUN_ARGS[@]}"

rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt','?'))" 2>/dev/null || echo "?")
[[ "$rtt" == "0" || "$rtt" == "0.0" || "$rtt" == "?" ]] && { echo "ERROR: invalid total_rtt in ${OUTPUT}" >&2; exit 1; }
echo "DONE: $OUTPUT  total_rtt=${rtt}"
