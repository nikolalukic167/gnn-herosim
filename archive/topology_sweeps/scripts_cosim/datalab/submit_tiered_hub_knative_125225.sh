#!/bin/bash
# Submit Knative baseline for tiered-hub dim22 125-225 sweep (11 configs).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export TIMEOUT="${TIMEOUT:-7200}"

if [[ ! -f "${SWEEP_DIR}/configs/knative_jobs.tsv" ]]; then
  echo "ERROR: missing ${SWEEP_DIR}/configs/knative_jobs.tsv" >&2
  exit 1
fi

job_count=$(($(wc -l < "${SWEEP_DIR}/configs/knative_jobs.tsv") - 1))
if [[ "$job_count" -ne 11 ]]; then
  echo "ERROR: expected 11 knative jobs, got ${job_count}" >&2
  exit 1
fi

kn_job=$(sbatch --export=ALL scripts_cosim/datalab/tiered_hub_knative_125225.sbatch | awk '{print $NF}')

echo "Submitted Knative array: ${kn_job} (tasks 0-10, TIMEOUT=${TIMEOUT}s)"
echo "SWEEP_DIR=${SWEEP_DIR} · WORKLOAD=${WORKLOAD}"
