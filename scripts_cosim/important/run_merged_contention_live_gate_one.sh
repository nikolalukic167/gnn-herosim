#!/usr/bin/env bash
# One merged-contention live gate job: knative | mlp | gnn (optional output suffix for decode variant).
# Usage: run_merged_contention_live_gate_one.sh <knative|mlp|gnn> <config_name> <config_json> [output_suffix]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

POLICY="${1:?knative, mlp, or gnn}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
OUT_SUFFIX="${4:-}"
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
  rtt=$(${HEROSIM_PY:-pipenv run python3} -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt', 0))" 2>/dev/null || echo 0)
  if [[ "${rtt}" != "0" && "${rtt}" != "0.0" ]]; then
    echo "SKIP (exists): $OUTPUT total_rtt=${rtt}"
    exit 0
  fi
fi

echo "=== merged live gate: ${POLICY} / ${CONFIG_NAME} / ${tag} ==="
[[ "$POLICY" == "gnn" ]] && echo "  GNN_DECODE_MODE=${GNN_DECODE_MODE}"

${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  "${RUN_ARGS[@]}"

rtt=$(${HEROSIM_PY:-pipenv run python3} -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt','?'))" 2>/dev/null || echo "?")
echo "DONE: $OUTPUT total_rtt=${rtt}"
