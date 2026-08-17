#!/usr/bin/env bash
# Launch Knative baselines locally for all bipartite-coordination hub configs (9).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
LOG_DIR="${ROOT}/logs/bipartite_knative_local"
mkdir -p "$LOG_DIR"

CONFIGS=(
  hub_k4_seek35 hub_k4_seek50 hub_k4_seek65
  hub_k6_seek35 hub_k6_seek50 hub_k6_seek65
  hub_k8_seek35 hub_k8_seek50 hub_k8_seek65
)

echo "Launching ${#CONFIGS[@]} Knative jobs locally (parallel; skips existing results)"
for cfg in "${CONFIGS[@]}"; do
  nohup env SWEEP_DIR="$SWEEP_DIR" bash scripts_cosim/run_bipartite_knative_local_one.sh "$cfg" \
    > "${LOG_DIR}/${cfg}.nohup.out" 2>&1 &
  echo "  PID $! · ${cfg}"
done

echo "Logs: ${LOG_DIR}/"
echo "Results: ${SWEEP_DIR}/results/*_knative.json"
