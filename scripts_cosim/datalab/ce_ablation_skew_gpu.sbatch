#!/bin/bash
#SBATCH --job-name=ce-ab-skew
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ce-ab-skew-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ce-ab-skew-%A_%a.err
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

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/ce_ablation_skew_20260611}"

CONFIGS=(
  "default_20_20_degree_skew:simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "01_balanced_40_40_p50_degree_skew:simulation_data/normal_sim_sweeps/atomic21_skew_configs/01_balanced_40_40_p50_degree_skew.json"
  "05_sparse_40_40_p25_degree_skew:simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%:*}"
path="${entry#*:}"

bash scripts_cosim/datalab/run_ce_ablation_gate3_one.sh \
  "$MODEL_TAG" "$GNN_MODEL_PATH" "$name" "$path"
