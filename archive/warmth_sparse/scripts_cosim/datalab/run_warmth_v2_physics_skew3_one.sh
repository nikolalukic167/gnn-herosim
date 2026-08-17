#!/usr/bin/env bash
# One warmth-v2 physics skew3 job (node_disk_v2): knative | mlp | gnn.
# Usage: run_warmth_v2_physics_skew3_one.sh <knative|mlp|gnn> <config_name> <config_json_path>
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?knative, mlp, or gnn}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
GNN_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp.pt"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_v2_physics_skew3_20260611}/results"
SEED=42
TIMEOUT="${TIMEOUT:-3600}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_DECODE_MODE=argmax
export INFERENCE_FEATURE_LAYOUT=dim22
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

case "$POLICY" in
  knative)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_knative.json"
    RUN_ARGS=(--knative_network)
    ;;
  gnn)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_dim14.json"
    export GNN_MODEL_PATH="${GNN_MODEL}"
    RUN_ARGS=(--gnn)
    ;;
  mlp)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_mlp_dim22.json"
    export MLP_MODEL_PATH="${MLP_MODEL}"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_MODEL")
    ;;
  *)
    echo "ERROR: policy must be knative, mlp, or gnn; got: ${POLICY}" >&2
    exit 1
    ;;
esac

if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: workload missing: $WORKLOAD" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config missing: $CONFIG_PATH" >&2
  exit 1
fi
if [[ "$POLICY" == "gnn" && ! -f "$GNN_MODEL" ]]; then
  echo "ERROR: GNN model missing: $GNN_MODEL" >&2
  exit 1
fi
if [[ "$POLICY" == "mlp" && ! -f "$MLP_MODEL" ]]; then
  echo "ERROR: MLP model missing: $MLP_MODEL" >&2
  exit 1
fi
if [[ -f "$OUTPUT" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "SKIP (exists): $OUTPUT"
  python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
  exit 0
fi

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

if [[ "$POLICY" != "knative" ]]; then
  python3 -c 'import importlib.util, sys; req=["torch"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
    || { echo "ERROR: micromamba env ${ENV_NAME} missing torch" >&2; exit 1; }
fi

echo "=== warmth-v2 skew3 ${POLICY} ${CONFIG_NAME} ==="
echo "HEROSIM_WARMTH_PHYSICS=${HEROSIM_WARMTH_PHYSICS}"
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
python3 -c "import json; d=json.load(open('${OUTPUT}')); print(f'OK elapsed=${elapsed}s total_rtt={d[\"total_rtt\"]}')"
