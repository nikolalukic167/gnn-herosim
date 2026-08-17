#!/bin/bash
# Submit bipartite sweep to a chosen GPU partition (keeps 478411 on GPU-l40s untouched).
# Usage:
#   bash submit_bipartite_coordination_partition.sh GPU-a40 gpu:a40:1
#   bash submit_bipartite_coordination_partition.sh GPU-l40s gpu:l40s:1
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${1:?partition e.g. GPU-a40}"
GRES="${2:?gres e.g. gpu:a40:1}"
JOB_TAG="${3:-${PARTITION#GPU-}}"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-7200}"

JOBS_TSV="${SWEEP_DIR}/configs/jobs_dim22.tsv"
if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing ${JOBS_TSV}" >&2
  exit 1
fi

job_count=$(($(wc -l < "$JOBS_TSV") - 1))
last_idx=$((job_count - 1))

echo "Test-only schedule:"
sbatch --test-only \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --array="0-${last_idx}" \
  --export=ALL \
  scripts_cosim/datalab/bipartite_coordination_gpu.sbatch

gpu_job=$(sbatch \
  --job-name="bipart-${JOB_TAG}" \
  --output="/home/nikola.lukic/gnn-herosim/logs/bipartite-${JOB_TAG}-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/bipartite-${JOB_TAG}-%A_%a.err" \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --array="0-${last_idx}" \
  --export=ALL \
  scripts_cosim/datalab/bipartite_coordination_gpu.sbatch | awk '{print $NF}')

echo "Submitted ${PARTITION} array: ${gpu_job} (tasks 0-${last_idx}, ${job_count} jobs, ${GRES})"
echo "SWEEP_DIR=${SWEEP_DIR} · results shared with other partitions (skip-if-exists)"
