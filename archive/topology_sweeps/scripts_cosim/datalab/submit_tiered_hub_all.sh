#!/bin/bash
# Submit both tiered-hub SLURM arrays (36 GPU + 9 CPU = 45 jobs).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"

if [[ ! -f "${SWEEP_DIR}/configs/jobs.tsv" ]]; then
  python3 scripts_cosim/important/generate_tiered_hub_configs.py --out-dir "${SWEEP_DIR}/configs"
fi

gpu_job=$(sbatch --export=ALL scripts_cosim/datalab/tiered_hub_gpu.sbatch | awk '{print $NF}')
cpu_job=$(sbatch --export=ALL scripts_cosim/datalab/tiered_hub_knative.sbatch | awk '{print $NF}')

echo "Submitted GPU array: ${gpu_job} (tasks 0-35)"
echo "Submitted Knative array: ${cpu_job} (tasks 0-8)"
echo "Total: 45 jobs · SWEEP_DIR=${SWEEP_DIR}"
