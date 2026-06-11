#!/usr/bin/env bash
# Knative baseline for CE ablation skew sweep (same workload/config as GNN legs).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CONFIG_NAME="${1:-default_20_20_degree_skew}"
CONFIG_PATH="${2:-simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/ce_ablation_skew_20260611}"
OUT_DIR="${SWEEP_DIR}/results"
OUTPUT="${OUT_DIR}/${CONFIG_NAME}_knative_network.json"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-3600}"

unset HEROSIM_WARMTH_PHYSICS
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

if [[ -f "$OUTPUT" ]]; then
  rtt=$(python3 -c "import json; print(json.load(open('${OUTPUT}')).get('total_rtt',0))")
  if [[ "${rtt}" != "0" && "${rtt}" != "0.0" ]]; then
    echo "SKIP (exists): $OUTPUT total_rtt=${rtt}"
    exit 0
  fi
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
