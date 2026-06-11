#!/bin/bash
#SBATCH --job-name=ce-ab-g3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ce-ab-g3-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ce-ab-g3-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-2

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

: "${MODEL_TAG:?MODEL_TAG required}"
: "${GNN_MODEL_PATH:?GNN_MODEL_PATH required}"

CONFIGS=(
  "default_20_20_p50:simulation_data/space_with_network.json"
  "01_balanced_40_40_p50:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/01_balanced_40_40_p50.json"
  "05_sparse_40_40_p25:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
)

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%:*}"
path="${entry#*:}"

bash scripts_cosim/datalab/run_ce_ablation_gate3_one.sh \
  "$MODEL_TAG" "$GNN_MODEL_PATH" "$name" "$path"
