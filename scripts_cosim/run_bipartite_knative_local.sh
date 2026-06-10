#!/usr/bin/env bash
# Launch Knative baselines locally for the 3 phase-anchor configs (k=4,6,8 @ seek50).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
LOG_DIR="${ROOT}/logs/bipartite_knative_local"
mkdir -p "$LOG_DIR"

# Phase-boundary anchors: k=b (4), k>b (6, 8) at moderate seek
CONFIGS=(
  hub_k4_seek50
  hub_k6_seek50
  hub_k8_seek50
)

echo "Launching ${#CONFIGS[@]} Knative jobs locally (parallel)"
for cfg in "${CONFIGS[@]}"; do
  nohup env SWEEP_DIR="$SWEEP_DIR" bash scripts_cosim/run_bipartite_knative_local_one.sh "$cfg" \
    > "${LOG_DIR}/${cfg}.nohup.out" 2>&1 &
  echo "  PID $! · ${cfg}"
done

echo "Logs: ${LOG_DIR}/"
echo "Results: ${SWEEP_DIR}/results/*_knative.json"
