#!/usr/bin/env bash
# One tiered-hub Knative job (per-arrival --knative_network).
# Usage: run_tiered_hub_knative_one.sh <config_name> <config_json_path>
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CONFIG_NAME="${1:?config name}"
CONFIG_PATH="${2:?config json path}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
OUT_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}/results"
SEED=42
TIMEOUT="${TIMEOUT:-7200}"

export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

OUTPUT="${OUT_DIR}/${CONFIG_NAME}_knative.json"

if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: workload missing: $WORKLOAD" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config missing: $CONFIG_PATH" >&2
  exit 1
fi
if [[ -f "$OUTPUT" ]]; then
  rtt=$(python3 -c "import json; d=json.load(open('${OUTPUT}')); print(d.get('total_rtt', 0))")
  if python3 -c "import sys; sys.exit(0 if float('${rtt}') > 0 else 1)"; then
    echo "SKIP (exists): $OUTPUT total_rtt=${rtt}"
    exit 0
  fi
  echo "WARN: removing invalid output ${OUTPUT} (total_rtt=${rtt})" >&2
  rm -f "$OUTPUT"
fi

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

echo "Running knative_network ${CONFIG_NAME} -> ${OUTPUT}"
python3 scripts_cosim/run_simulation.py \
  --knative_network \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED"

python3 -c "import json; print('total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
