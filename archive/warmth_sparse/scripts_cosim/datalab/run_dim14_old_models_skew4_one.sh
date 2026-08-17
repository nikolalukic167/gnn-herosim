#!/usr/bin/env bash
# One skew-4 job: legacy dim14 CE-GNN or dim22 MLP (Regime A / argmax / dim22 layout).
# Usage: run_dim14_old_models_skew4_one.sh <gnn|mlp> <config_name> <config_json_path>
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?gnn or mlp}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
GNN_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp.pt"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/dim14_old_models_skew4_20260610}/results"
SEED=42
TIMEOUT="${TIMEOUT:-3600}"

export GNN_DECODE_MODE=argmax
export INFERENCE_FEATURE_LAYOUT=dim22
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

case "$POLICY" in
  gnn)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_dim14.json"
    export GNN_MODEL_PATH="${GNN_MODEL}"
    MODEL_CHECK="$GNN_MODEL"
    RUN_ARGS=(--gnn)
    ;;
  mlp)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_dim22.json"
    export MLP_MODEL_PATH="${MLP_MODEL}"
    MODEL_CHECK="$MLP_MODEL"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_MODEL")
    ;;
  *)
    echo "ERROR: policy must be gnn or mlp, got: ${POLICY}" >&2
    exit 1
    ;;
esac

if [[ ! -f "$MODEL_CHECK" ]]; then
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
if [[ -f "$OUTPUT" ]]; then
  echo "SKIP (exists): $OUTPUT"
  python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
  exit 0
fi

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

python3 -c 'import importlib.util, sys; req=["torch","torch_geometric"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
  || { echo "ERROR: micromamba env ${ENV_NAME} missing torch/torch_geometric" >&2; exit 1; }

echo "Running ${POLICY} ${CONFIG_NAME} (dim22 layout) -> ${OUTPUT}"
python3 scripts_cosim/run_simulation.py \
  "${RUN_ARGS[@]}" \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED"

python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
