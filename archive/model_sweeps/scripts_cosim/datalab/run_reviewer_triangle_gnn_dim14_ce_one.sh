#!/usr/bin/env bash
# Run one dim14-ce GNN config (Regime A / argmax) into reviewer_triangle_all7 sweep dir.
# Usage: run_reviewer_triangle_gnn_dim14_ce_one.sh <config_name> <config_json_path>
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CONFIG_NAME="${1:?config name}"
CONFIG_PATH="${2:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
MODEL_PATH="models/near-rtt-v2-dim14-ce-only.pt"
OUT_DIR="simulation_data/normal_sim_sweeps/reviewer_triangle_all7_20260609/results"
OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_dim14_ce.json"
SEED=42
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
if [[ -f "$OUTPUT" ]]; then
  echo "SKIP (exists): $OUTPUT"
  exit 0
fi

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

python3 -c 'import importlib.util, sys; req=["torch","torch_geometric"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
  || { echo "ERROR: micromamba env ${ENV_NAME} missing torch/torch_geometric" >&2; exit 1; }

echo "Running GNN dim14-ce ${CONFIG_NAME} -> ${OUTPUT}"
python3 scripts_cosim/run_simulation.py \
  --gnn \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED"

python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
