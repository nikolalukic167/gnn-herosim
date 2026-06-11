#!/usr/bin/env bash
# Sweep-only: 3 CE ablation models × 3 degree-skew topologies (workload-100-100).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/ce_ablation_skew_20260611}"

for model_path in \
  models/near-rtt-ce-reduced-features.pt \
  models/near-rtt-v2-dim14-1060.pt \
  models/near-rtt-v2-dim14-ce-only.pt; do
  if [[ ! -f "$model_path" ]]; then
    echo "ERROR: missing $model_path" >&2
    exit 1
  fi
done

for cfg in \
  simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json \
  simulation_data/normal_sim_sweeps/atomic21_skew_configs/01_balanced_40_40_p50_degree_skew.json \
  simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json; do
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: missing $cfg" >&2
    exit 1
  fi
done

reduced_job=$(sbatch --parsable \
  --job-name=ce-ab-sk-red \
  --export=ALL,MODEL_TAG=ce-reduced,GNN_MODEL_PATH=models/near-rtt-ce-reduced-features.pt \
  scripts_cosim/datalab/ce_ablation_skew_gpu.sbatch)
echo "Submitted reduced skew array: ${reduced_job}"

full1060_job=$(sbatch --parsable \
  --job-name=ce-ab-sk-full \
  --export=ALL,MODEL_TAG=dim14-1060-full,GNN_MODEL_PATH=models/near-rtt-v2-dim14-1060.pt \
  scripts_cosim/datalab/ce_ablation_skew_gpu.sbatch)
echo "Submitted dim14-1060-full skew array: ${full1060_job}"

ce_only_job=$(sbatch --parsable \
  --job-name=ce-ab-sk-ce \
  --export=ALL,MODEL_TAG=dim14-ce-only,GNN_MODEL_PATH=models/near-rtt-v2-dim14-ce-only.pt \
  scripts_cosim/datalab/ce_ablation_skew_gpu.sbatch)
echo "Submitted dim14-ce-only skew array: ${ce_only_job}"

echo ""
echo "Monitor: squeue -u nikola.lukic | grep ce-ab-sk"
echo "Sweep dir: ${SWEEP_DIR}/results/"
