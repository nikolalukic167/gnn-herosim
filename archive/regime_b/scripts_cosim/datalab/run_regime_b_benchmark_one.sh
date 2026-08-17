#!/usr/bin/env bash
# One Regime B benchmark job: cold mixed burst on hub topology.
# Usage: run_regime_b_benchmark_one.sh <policy> <config_name> <config_json_path>
#
# Policies: knative | knative_ect | mlp_seqblend1
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/projects/my-herosim}"
cd "$ROOT"

POLICY="${1:?knative, knative_ect, or mlp_seqblend1}"
CONFIG_NAME="${2:?config name e.g. hub_k6_seek50}"
CONFIG_PATH="${3:?config json path}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/regime_b_hub9_20260612}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-cold-burst-mixed.json}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt}"
OUT_DIR="${SWEEP_DIR}/results"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-7200}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export HEROSIM_DEFER_COLD_REPLICA_INIT="${HEROSIM_DEFER_COLD_REPLICA_INIT:-1}"
export HEROSIM_FAST_FORWARD_WARMUP="${HEROSIM_FAST_FORWARD_WARMUP:-1}"
export SIM_FORCE_FULL_STATS="${SIM_FORCE_FULL_STATS:-1}"
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-16}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.02}"
export INFERENCE_FEATURE_LAYOUT="${INFERENCE_FEATURE_LAYOUT:-ce_reduced}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs/regime_b_benchmark

case "$POLICY" in
  knative)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_regime_b_knative.json"
    RUN_ARGS=(--knative_network)
    ;;
  knative_ect)
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_regime_b_knative_ect.json"
    RUN_ARGS=(--knative_network_ect)
    ;;
  mlp_seqblend1)
    export GNN_DECODE_MODE=seqblend
    export GNN_SEQBLEND_QUEUE_MARGIN="${GNN_SEQBLEND_QUEUE_MARGIN:-1}"
    export MLP_MODEL_PATH="${MLP_MODEL}"
    OUTPUT="${OUT_DIR}/${CONFIG_NAME}_regime_b_mlp_seqblend1.json"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_MODEL")
    ;;
  *)
    echo "ERROR: policy must be knative, knative_ect, or mlp_seqblend1; got: ${POLICY}" >&2
    exit 1
    ;;
esac

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config missing: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: workload missing: $WORKLOAD (run generate_cold_burst_workload.py first)" >&2
  exit 1
fi
if [[ "$POLICY" == "mlp_seqblend1" && ! -f "$MLP_MODEL" ]]; then
  echo "ERROR: MLP model missing: $MLP_MODEL" >&2
  exit 1
fi

if [[ -f "$OUTPUT" && "${FORCE_RERUN:-0}" != "1" ]]; then
  score=$(pipenv run python3 -c "
import json
d = json.load(open('${OUTPUT}'))
s = d.get('regime_b_primary_score_s') or (d.get('regime_b') or {}).get('regime_b_primary_score_s')
print(s if s is not None else 0)
" 2>/dev/null || echo 0)
  if pipenv run python3 -c "import sys; sys.exit(0 if float('${score}') > 0 else 1)"; then
    echo "SKIP (exists): $OUTPUT regime_b_primary_score_s=${score}"
    exit 0
  fi
  echo "WARN: stale output ${OUTPUT}; re-running" >&2
fi

LOG="${ROOT}/logs/regime_b_benchmark/${CONFIG_NAME}_${POLICY}.log"
echo "=== Regime B ${POLICY} ${CONFIG_NAME} ==="
echo "HEROSIM_WARMTH_PHYSICS=${HEROSIM_WARMTH_PHYSICS}"
echo "HEROSIM_DEFER_COLD_REPLICA_INIT=${HEROSIM_DEFER_COLD_REPLICA_INIT}"
echo "HEROSIM_FAST_FORWARD_WARMUP=${HEROSIM_FAST_FORWARD_WARMUP}"
echo "SIM_FORCE_FULL_STATS=${SIM_FORCE_FULL_STATS}"
echo "config=${CONFIG_PATH}"
echo "workload=${WORKLOAD}"
echo "output=${OUTPUT}"

pipenv run python3 scripts_cosim/run_simulation.py \
  "${RUN_ARGS[@]}" \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --timeout "$TIMEOUT" \
  --seed "$SEED" \
  2>&1 | tee "$LOG"

pipenv run python3 -c "
import json
d = json.load(open('${OUTPUT}'))
rb = d.get('regime_b_primary_score_s') or (d.get('regime_b') or {}).get('regime_b_primary_score_s')
print('DONE', '${CONFIG_NAME}', '${POLICY}', 'total_rtt=', d.get('total_rtt'), 'regime_b=', rb)
"
