#!/bin/bash
# Submit retry jobs for tiered-hub dim22 125-225 sweep (separate from main array).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export TIMEOUT="${TIMEOUT:-7200}"

RETRY_TSV="${SWEEP_DIR}/configs/retry_jobs.tsv"
if [[ ! -f "$RETRY_TSV" ]]; then
  echo "ERROR: missing ${RETRY_TSV}" >&2
  exit 1
fi

job_count=$(($(wc -l < "$RETRY_TSV") - 1))
max_id=$((job_count - 1))

retry_job=$(sbatch --export=ALL scripts_cosim/datalab/tiered_hub_dim22_retry.sbatch | awk '{print $NF}')

echo "Submitted retry array: ${retry_job} (tasks 0-${max_id}, TIMEOUT=${TIMEOUT}s)"
echo "SWEEP_DIR=${SWEEP_DIR} · WORKLOAD=${WORKLOAD}"
