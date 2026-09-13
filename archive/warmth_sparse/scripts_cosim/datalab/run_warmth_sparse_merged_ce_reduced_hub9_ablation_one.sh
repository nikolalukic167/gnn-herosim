#!/usr/bin/env bash
# One hub9 MLP ablation job (node_disk_v2, 125-225): mlp_seqblend1 | mlp_dim22.
# Usage: run_warmth_sparse_merged_ce_reduced_hub9_ablation_one.sh <policy> <config_name> <config_json_path>
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?mlp_seqblend1 or mlp_dim22}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
MLP_CE_REDUCED="${MLP_CE_REDUCED:-models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt}"
MLP_DIM22="${MLP_DIM22:-models/tabular/batch_edge_mlp.pt}"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_merged_ce_reduced_hub9_20260611}/results"
SEED=42
TIMEOUT="${TIMEOUT:-7200}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export PYTHONUNBUFFERED=1
unset GNN_QUEUE_FILTER_MAX_DELTA

case "$POLICY" in
  mlp_seqblend1)
    export GNN_DECODE_MODE=seqblend
    export GNN_SEQBLEND_QUEUE_MARGIN="${GNN_SEQBLEND_QUEUE_MARGIN:-1}"
    export INFERENCE_FEATURE_LAYOUT=ce_reduced
    export MLP_MODEL_PATH="${MLP_CE_REDUCED}"
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_sparse_merged_ce_reduced_seqblend1.json"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_CE_REDUCED")
    MODEL_CHECK="$MLP_CE_REDUCED"
    ;;
  mlp_dim22)
    export GNN_DECODE_MODE=argmax
    unset GNN_SEQBLEND_QUEUE_MARGIN
    export INFERENCE_FEATURE_LAYOUT=dim22
    export MLP_MODEL_PATH="${MLP_DIM22}"
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_dim22.json"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_DIM22")
    MODEL_CHECK="$MLP_DIM22"
    ;;
  *)
    echo "ERROR: policy must be mlp_seqblend1 or mlp_dim22; got: ${POLICY}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR" logs

if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: workload missing: $WORKLOAD" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config missing: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$MODEL_CHECK" ]]; then
  echo "ERROR: model missing: $MODEL_CHECK" >&2
  exit 1
fi
if [[ -f "$OUTPUT" && "${FORCE_RERUN:-0}" != "1" ]]; then
  rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt', 0))" 2>/dev/null || echo 0)
  if [[ "${rtt}" != "0" && "${rtt}" != "0.0" ]]; then
    echo "SKIP (exists): $OUTPUT total_rtt=${rtt}"
    exit 0
  fi
  echo "WARN: stale output ${OUTPUT} (total_rtt=${rtt}); re-running"
fi

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

python3 -c 'import importlib.util, sys; req=["torch"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
  || { echo "ERROR: micromamba env ${ENV_NAME} missing torch" >&2; exit 1; }

echo "=== hub9 ablation ${POLICY} ${CONFIG_NAME} ==="
echo "HEROSIM_WARMTH_PHYSICS=${HEROSIM_WARMTH_PHYSICS}"
echo "GNN_DECODE_MODE=${GNN_DECODE_MODE}"
echo "GNN_SEQBLEND_QUEUE_MARGIN=${GNN_SEQBLEND_QUEUE_MARGIN:-}"
echo "INFERENCE_FEATURE_LAYOUT=${INFERENCE_FEATURE_LAYOUT}"
echo "MLP_MODEL=${MODEL_CHECK}"
echo "config=${CONFIG_PATH} -> ${OUTPUT}"
start=$(date +%s)

python3 scripts_cosim/run_simulation.py \
  "${RUN_ARGS[@]}" \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED"

elapsed=$(( $(date +%s) - start ))
python3 - <<PY
import json
from pathlib import Path

out = Path("${OUTPUT}")
d = json.loads(out.read_text())
rtt = d.get("total_rtt")
qv = d.get("chosen_queue_vs_min", {})
print(
    f"OK elapsed=${elapsed}s total_rtt={rtt} "
    f"chosen_queue_vs_min_mean={qv.get('mean', '?')} p95={qv.get('p95', '?')}"
)
PY
