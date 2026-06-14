#!/bin/bash
#SBATCH --job-name=skew3-wssm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew3-wssm-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew3-wssm-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5

# Experiment Group 3a: skew3 full gate (node_disk_v2) — GNN + MLP wssm models, 3 configs each.
#   0-2 : gnn_wssm x 3 skew3 configs
#   3-5 : mlp_wssm x 3 skew3 configs
#
# Knative baselines run separately: skew3_full_gate_knative.sbatch (no GPU needed).
# Submit: sbatch scripts_cosim/datalab/skew3_full_gate_gpu.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/skew3_full_gate_20260614}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2

CONFIGS=(
  "default_20_20_p50|simulation_data/space_with_network.json"
  "default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  policy="gnn"
  cfg_idx=$SLURM_ARRAY_TASK_ID
else
  policy="mlp"
  cfg_idx=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_skew3_full_gate_one.sh "$policy" "$name" "$path"
