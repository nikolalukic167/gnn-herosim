#!/bin/bash
# Submit sweep_bipartite_coordination_v1 (12 hub configs × 2 policies = 24 GPU jobs).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-7200}"

JOBS_TSV="${SWEEP_DIR}/configs/jobs_dim22.tsv"
if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing ${JOBS_TSV} — run prepare_bipartite_coordination_sweep.sh first" >&2
  exit 1
fi

job_count=$(($(wc -l < "$JOBS_TSV") - 1))
last_idx=$((job_count - 1))
if [[ "$job_count" -lt 1 ]]; then
  echo "ERROR: empty jobs manifest" >&2
  exit 1
fi

gpu_job=$(sbatch --array="0-${last_idx}" --export=ALL \
  scripts_cosim/datalab/bipartite_coordination_gpu.sbatch | awk '{print $NF}')

echo "Submitted bipartite coordination GPU array: ${gpu_job} (tasks 0-${last_idx}, ${job_count} jobs)"
echo "SWEEP_DIR=${SWEEP_DIR} · WORKLOAD=${WORKLOAD} · GNN_BATCH_SIZE=4 · TIMEOUT=${TIMEOUT}s"
