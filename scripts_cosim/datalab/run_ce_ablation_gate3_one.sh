#!/usr/bin/env bash
# Run one gate3 config for CE ablation GNN (legacy 1060 cache family).
# Usage: run_ce_ablation_gate3_one.sh <model_tag> <model_path> <config_name> <config_json_path>
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

MODEL_TAG="${1:?model tag e.g. dim14-ce-only-1060}"
MODEL_PATH="${2:?model path}"
CONFIG_NAME="${3:?config name}"
CONFIG_PATH="${4:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/ce_ablation_gate3_20260611}"
OUT_DIR="${SWEEP_DIR}/results"
OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_${MODEL_TAG}.json"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-3600}"

export GNN_MODEL_PATH="${MODEL_PATH}"
export GNN_DECODE_MODE=argmax
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model missing: $MODEL_PATH" >&2
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

python3 -c 'import importlib.util, sys; req=["torch","torch_geometric"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
  || { echo "ERROR: micromamba env ${ENV_NAME} missing torch/torch_geometric" >&2; exit 1; }

echo "Running GNN ${MODEL_TAG} ${CONFIG_NAME} -> ${OUTPUT}"
python3 scripts_cosim/run_simulation.py \
  --gnn \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED"

python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
