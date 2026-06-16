#!/bin/bash
#SBATCH --job-name=bipart-mlp-bc
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bipart-mlp-bc-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bipart-mlp-bc-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8

# MLP dim22 trained from batch cache (same as GNN wssm) — 9 bipartite hub configs.
# Submit: sbatch scripts_cosim/datalab/bipartite_mlp_dim22_batchcache.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/bipartite_v2_mlp_dim22_batchcache_20260614}"
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

bash scripts_cosim/datalab/run_bipartite_mlp_dim22_batchcache_one.sh "$name" "$path"
