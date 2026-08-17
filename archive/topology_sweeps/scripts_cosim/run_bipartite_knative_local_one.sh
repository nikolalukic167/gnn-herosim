#!/usr/bin/env bash
# Run Knative per-arrival baseline locally for key bipartite-coordination configs.
# Usage: run_bipartite_knative_local_one.sh <config_name>
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

CONFIG_NAME="${1:?config name}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
CONFIG_PATH="${SWEEP_DIR}/configs/${CONFIG_NAME}.json"
OUT_DIR="${SWEEP_DIR}/results"
SEED=42
TIMEOUT="${TIMEOUT:-7200}"
LOG_DIR="${ROOT}/logs/bipartite_knative_local"
OUTPUT="${OUT_DIR}/${CONFIG_NAME}_knative.json"
LOG="${LOG_DIR}/${CONFIG_NAME}.log"

export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: missing workload: $WORKLOAD" >&2
  exit 1
fi
if [[ -f "$OUTPUT" ]]; then
  rtt=$(pipenv run python3 -c "import json; print(json.load(open('${OUTPUT}')).get('total_rtt', 0))")
  if pipenv run python3 -c "import sys; sys.exit(0 if float('${rtt}') > 0 else 1)"; then
    echo "SKIP (exists): $OUTPUT total_rtt=${rtt}"
    exit 0
  fi
  rm -f "$OUTPUT"
fi

echo "Starting knative ${CONFIG_NAME} -> ${OUTPUT} (log: ${LOG})"
pipenv run python3 scripts_cosim/run_simulation.py \
  --knative_network \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED" \
  2>&1 | tee "$LOG"

pipenv run python3 -c "import json; print('DONE', '${CONFIG_NAME}', 'total_rtt=', json.load(open('${OUTPUT}'))['total_rtt'])"
