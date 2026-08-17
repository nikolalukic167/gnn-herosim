#!/usr/bin/env bash
# One mega-compare all7 job.
# Usage: run_mega_compare_all7_one.sh <policy> <config_name> <config_json_path>
#
# Policies:
#   gnn_warmth   — near-rtt-v2-warmth-dim14-ce-only.pt           (warmth_v2 cache, no skew)
#   gnn_wsm      — near-rtt-v2-warmth-sparse-merged-dim14-ce-only.pt (warmth+sparse, 824 ds)
#   gnn_wssm     — near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt (824+skew, newest)
#   mlp_warmth   — batch_edge_mlp_warmth_v2.pt                    (warmth_v2 seq cache)
#   mlp_wsm      — batch_edge_mlp_warmth_sparse_merged.pt         (merged full, dim22)
#   mlp_wssm     — batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt (newest, ce_reduced)
#
# No warmth physics (standard vanilla sim — test raw model quality on legacy benchmark).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?policy (gnn_warmth|gnn_wsm|gnn_wssm|mlp_warmth|mlp_wsm|mlp_wssm)}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/mega_compare_all7_20260614}/results"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-3600}"

GNN_WARMTH="models/near-rtt-v2-warmth-dim14-ce-only.pt"
GNN_WSM="models/near-rtt-v2-warmth-sparse-merged-dim14-ce-only.pt"
GNN_WSSM="models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt"
MLP_WARMTH="models/tabular/batch_edge_mlp_warmth_v2.pt"
MLP_WSM="models/tabular/batch_edge_mlp_warmth_sparse_merged.pt"
MLP_WSSM="models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt"

export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
unset INFERENCE_FEATURE_LAYOUT HEROSIM_WARMTH_PHYSICS 2>/dev/null || true
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

MODEL_CHECK=""
case "$POLICY" in
  gnn_warmth)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_warmth.json"
    export GNN_MODEL_PATH="${GNN_WARMTH}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$GNN_WARMTH"
    RUN_ARGS=(--gnn)
    ;;
  gnn_wsm)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_wsm.json"
    export GNN_MODEL_PATH="${GNN_WSM}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$GNN_WSM"
    RUN_ARGS=(--gnn)
    ;;
  gnn_wssm)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_wssm.json"
    export GNN_MODEL_PATH="${GNN_WSSM}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$GNN_WSSM"
    RUN_ARGS=(--gnn)
    ;;
  mlp_warmth)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_warmth.json"
    export MLP_MODEL_PATH="${MLP_WARMTH}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$MLP_WARMTH"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_WARMTH")
    ;;
  mlp_wsm)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_wsm.json"
    export MLP_MODEL_PATH="${MLP_WSM}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$MLP_WSM"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_WSM")
    ;;
  mlp_wssm)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_wssm.json"
    export MLP_MODEL_PATH="${MLP_WSSM}"
    export INFERENCE_FEATURE_LAYOUT=ce_reduced
    MODEL_CHECK="$MLP_WSSM"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_WSSM")
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

echo "=== mega-compare all7: ${POLICY} / ${CONFIG_NAME} ==="
echo "  INFERENCE_FEATURE_LAYOUT=${INFERENCE_FEATURE_LAYOUT:-<default>}"
echo "  MODEL=${MODEL_CHECK}"
echo "  WORKLOAD=${WORKLOAD}"
echo "  OUTPUT=${OUTPUT}"

python3 scripts_cosim/run_simulation.py \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  "${RUN_ARGS[@]}"

rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt','?'))" 2>/dev/null || echo "?")
echo "DONE: $OUTPUT  total_rtt=${rtt}"
