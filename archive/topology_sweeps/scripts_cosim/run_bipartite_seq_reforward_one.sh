#!/usr/bin/env bash
# Quick seq_reforward probe on one bipartite-coordination config.
# Usage: run_bipartite_seq_reforward_one.sh <config_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

CONFIG_NAME="${1:?config name}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
CONFIG_PATH="${SWEEP_DIR}/configs/${CONFIG_NAME}.json"
OUT_DIR="${SWEEP_DIR}/results"
SEED=42
TIMEOUT="${TIMEOUT:-7200}"
LOG_DIR="${ROOT}/logs/bipartite_seq_reforward"
OUTPUT="${OUT_DIR}/${CONFIG_NAME}_gnn_dim22_seq_reforward.json"
LOG="${LOG_DIR}/${CONFIG_NAME}.log"
GNN_MODEL="${GNN_MODEL_PATH:-models/near-rtt-v2-dim14-ce-only.pt}"

export PYTHONUNBUFFERED=1
export GNN_DECODE_MODE=seq_reforward
export INFERENCE_FEATURE_LAYOUT=dim22
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export GNN_MODEL_PATH="${GNN_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_py() {
  if command -v pipenv >/dev/null 2>&1 && [[ -f Pipfile ]]; then
    pipenv run python3 "$@"
  else
    python3 "$@"
  fi
}

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: missing workload: $WORKLOAD" >&2
  exit 1
fi
if [[ ! -f "$GNN_MODEL" ]]; then
  echo "ERROR: missing model: $GNN_MODEL" >&2
  exit 1
fi

echo "Starting seq_reforward ${CONFIG_NAME} -> ${OUTPUT} (log: ${LOG})"
run_py scripts_cosim/run_simulation.py \
  --gnn \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED" \
  2>&1 | tee "$LOG"

run_py -c "
import json
from pathlib import Path
res = Path('${SWEEP_DIR}/results')
baseline = {
    'argmax_gnn': json.loads((res / '${CONFIG_NAME}_gnn_dim22.json').read_text()).get('total_rtt'),
    'knative': json.loads((res / '${CONFIG_NAME}_knative.json').read_text()).get('total_rtt'),
}
new = json.loads(Path('${OUTPUT}').read_text()).get('total_rtt')
print('DONE', '${CONFIG_NAME}', 'seq_reforward=', new)
for k, v in baseline.items():
    if v:
        print(f'  vs {k}: {(new-v)/v*100:+.1f}%')
"
