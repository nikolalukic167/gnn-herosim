#!/bin/bash
#SBATCH --job-name=bipart-mlp-fix
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bipart-mlp-fix-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bipart-mlp-fix-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8

# Fixed MLP bipartite v2 — dim22 layout (restores is_warm, is_cold, queue signal).
# 9 jobs: hub_k{4,6,8} x seek{35,50,65}%
# Model: batch_edge_mlp_warmth_sparse_skew_merged_dim22.pt
# Submit: sbatch scripts_cosim/datalab/bipartite_mlp_dim22_fix.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/bipartite_v2_mlp_dim22_fix_20260614}"
BIPARTITE_CFG_DIR="simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs"

HUB_CONFIGS=(
  "hub_k4_seek35|${BIPARTITE_CFG_DIR}/hub_k4_seek35.json"
  "hub_k4_seek50|${BIPARTITE_CFG_DIR}/hub_k4_seek50.json"
  "hub_k4_seek65|${BIPARTITE_CFG_DIR}/hub_k4_seek65.json"
  "hub_k6_seek35|${BIPARTITE_CFG_DIR}/hub_k6_seek35.json"
  "hub_k6_seek50|${BIPARTITE_CFG_DIR}/hub_k6_seek50.json"
  "hub_k6_seek65|${BIPARTITE_CFG_DIR}/hub_k6_seek65.json"
  "hub_k8_seek35|${BIPARTITE_CFG_DIR}/hub_k8_seek35.json"
  "hub_k8_seek50|${BIPARTITE_CFG_DIR}/hub_k8_seek50.json"
  "hub_k8_seek65|${BIPARTITE_CFG_DIR}/hub_k8_seek65.json"
)

entry="${HUB_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_bipartite_mlp_dim22_fix_one.sh "$name" "$path"
