#!/bin/bash
# Submit tiered-hub dim22 GNN+MLP sweep (11 configs × 2 policies = 22 GPU jobs).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"

if [[ ! -f "${SWEEP_DIR}/configs/jobs_dim22.tsv" ]]; then
  python3 scripts_cosim/important/generate_tiered_hub_configs.py \
    --out-dir "${SWEEP_DIR}/configs" \
    --policies dim22 \
    --with-controls
fi

job_count=$(($(wc -l < "${SWEEP_DIR}/configs/jobs_dim22.tsv") - 1))
if [[ "$job_count" -ne 22 ]]; then
  echo "ERROR: expected 22 jobs in jobs_dim22.tsv, got ${job_count}" >&2
  exit 1
fi

gpu_job=$(sbatch --export=ALL scripts_cosim/datalab/tiered_hub_dim22_gpu.sbatch | awk '{print $NF}')

echo "Submitted dim22 GPU array: ${gpu_job} (tasks 0-21)"
echo "SWEEP_DIR=${SWEEP_DIR} · WORKLOAD=${WORKLOAD}"
