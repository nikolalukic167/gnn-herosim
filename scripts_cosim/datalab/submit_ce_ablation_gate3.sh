#!/usr/bin/env bash
# Sweep-only: legacy 1060 CE ablation on gate3 (no training on cluster).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/ce_ablation_gate3_20260611}"
export SWEEP_DIR

for model_path in \
  models/near-rtt-ce-reduced-features.pt \
  models/near-rtt-v2-dim14-1060.pt \
  models/near-rtt-v2-dim14-ce-only.pt; do
  if [[ ! -f "$model_path" ]]; then
    echo "ERROR: missing $model_path (rsync from mitrix first)" >&2
    exit 1
  fi
done

reduced_job=$(sbatch --parsable \
  --job-name=ce-ab-reduced \
  --export=ALL,MODEL_TAG=ce-reduced,GNN_MODEL_PATH=models/near-rtt-ce-reduced-features.pt \
  scripts_cosim/datalab/ce_ablation_gate3_gpu.sbatch)
echo "Submitted reduced gate3 array: ${reduced_job}"

full1060_job=$(sbatch --parsable \
  --job-name=ce-ab-full1060 \
  --export=ALL,MODEL_TAG=dim14-1060-full,GNN_MODEL_PATH=models/near-rtt-v2-dim14-1060.pt \
  scripts_cosim/datalab/ce_ablation_gate3_gpu.sbatch)
echo "Submitted dim14-1060 (full+ranking, legacy cache) gate3 array: ${full1060_job}"

ce_only_job=$(sbatch --parsable \
  --job-name=ce-ab-ceonly \
  --export=ALL,MODEL_TAG=dim14-ce-only,GNN_MODEL_PATH=models/near-rtt-v2-dim14-ce-only.pt \
  scripts_cosim/datalab/ce_ablation_gate3_gpu.sbatch)
echo "Submitted dim14-ce-only (full dim14 CE) gate3 array: ${ce_only_job}"

echo ""
echo "Monitor: squeue -u nikola.lukic | grep ce-ab"
echo "Sweep dir: ${SWEEP_DIR}/results/"
