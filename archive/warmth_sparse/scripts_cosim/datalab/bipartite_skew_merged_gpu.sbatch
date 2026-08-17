#!/bin/bash
#SBATCH --job-name=bipart-wssm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bipart-wssm-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bipart-wssm-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-17

# Experiment Group 2: bipartite v2 (node_disk_v2) with skew-merged (wssm) models.
# k in {4,6,8} x seek in {35,50,65}% = 9 hub configs x 2 policies = 18 jobs.
#
#   0-8  : gnn_wssm x 9 hub configs
#   9-17 : mlp_wssm x 9 hub configs
#
# Knative+ECT references: regime_b_hub9_20260612/results/ (18 files, all 9 configs, node_disk_v2).
# Submit: sbatch scripts_cosim/datalab/bipartite_skew_merged_gpu.sbatch
# Fallback partition: GPU-l40s (change --partition and --gres if a40 unavailable).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/bipartite_v2_skew_merged_20260614}"
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

if (( SLURM_ARRAY_TASK_ID < 9 )); then
  policy="gnn"
  cfg_idx=$SLURM_ARRAY_TASK_ID
else
  policy="mlp"
  cfg_idx=$(( SLURM_ARRAY_TASK_ID - 9 ))
fi

entry="${HUB_CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_bipartite_skew_merged_one.sh "$policy" "$name" "$path"
