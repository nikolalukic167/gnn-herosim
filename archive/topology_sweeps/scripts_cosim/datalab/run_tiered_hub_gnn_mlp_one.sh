#!/usr/bin/env bash
# One tiered-hub job (5 policy variants). Usage:
#   run_tiered_hub_gnn_mlp_one.sh <policy> <config_name> <config_json_path>
# Policies: gnn_dim22 | gnn_atomic21 | mlp_dim22 | mlp_atomic21 | knative
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?policy}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
GNN_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
MLP_DIM22="models/tabular/batch_edge_mlp.pt"
MLP_ATOMIC21="models/tabular/batch_edge_mlp_atomic21.pt"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_20260610}/results"
SEED=42
TIMEOUT="${TIMEOUT:-3600}"

export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
unset INFERENCE_FEATURE_LAYOUT
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

MODEL_CHECK=""
case "$POLICY" in
  gnn_dim22)
    export INFERENCE_FEATURE_LAYOUT=dim22
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_dim22.json"
    export GNN_MODEL_PATH="${GNN_MODEL}"
    MODEL_CHECK="$GNN_MODEL"
    RUN_ARGS=(--gnn)
    NEED_TORCH=1
    ;;
  gnn_atomic21)
    export INFERENCE_FEATURE_LAYOUT=atomic21
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_atomic21.json"
    export GNN_MODEL_PATH="${GNN_MODEL}"
    MODEL_CHECK="$GNN_MODEL"
    RUN_ARGS=(--gnn)
    NEED_TORCH=1
    ;;
  mlp_dim22)
    export INFERENCE_FEATURE_LAYOUT=dim22
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_dim22.json"
    export MLP_MODEL_PATH="${MLP_DIM22}"
    MODEL_CHECK="$MLP_DIM22"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_DIM22")
    NEED_TORCH=1
    ;;
  mlp_atomic21)
    export INFERENCE_FEATURE_LAYOUT=atomic21
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_atomic21.json"
    export MLP_MODEL_PATH="${MLP_ATOMIC21}"
    MODEL_CHECK="$MLP_ATOMIC21"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_ATOMIC21")
    NEED_TORCH=1
    ;;
  knative)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_knative.json"
    RUN_ARGS=(--knative_network)
    NEED_TORCH=0
    ;;
  *)
    echo "ERROR: unknown policy: ${POLICY}" >&2
    exit 1
    ;;
esac

if [[ -n "$MODEL_CHECK" && ! -f "$MODEL_CHECK" ]]; then
  echo "ERROR: model missing: $MODEL_CHECK" >&2
  exit 1
fi
if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: workload missing: $WORKLOAD" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config missing: $CONFIG_PATH" >&2
  exit 1
fi
if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

if [[ -f "$OUTPUT" ]]; then
  echo "SKIP (exists): $OUTPUT"
  python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
  exit 0
fi

if [[ "$NEED_TORCH" == "1" ]]; then
  python3 -c 'import importlib.util, sys; req=["torch","torch_geometric"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
    || { echo "ERROR: micromamba env ${ENV_NAME} missing torch/torch_geometric" >&2; exit 1; }
fi

echo "Running ${POLICY} ${CONFIG_NAME} -> ${OUTPUT}"
python3 scripts_cosim/run_simulation.py \
  "${RUN_ARGS[@]}" \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED"

python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
