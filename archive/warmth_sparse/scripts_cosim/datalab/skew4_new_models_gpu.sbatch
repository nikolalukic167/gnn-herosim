#!/bin/bash
#SBATCH --job-name=skew4-wssm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew4-wssm-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew4-wssm-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-7

# Experiment Group 4a: skew4 new models (node_disk_v2, 125-225 workload).
# 4 configs x 2 policies = 8 jobs.
#   0-3 : gnn_wssm x 4 skew4 configs
#   4-7 : mlp_wssm x 4 skew4 configs
#
# Knative baselines with node_disk_v2: skew4_new_models_knative.sbatch (4 jobs).
# Previous skew4 results (dim14_old_models_skew4_125225_20260610/) used legacy physics.
# Submit: sbatch scripts_cosim/datalab/skew4_new_models_gpu.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/skew4_new_models_20260614}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2

CONFIGS=(
  "default_20_20_p50|simulation_data/space_with_network.json"
  "05_sparse_40_40_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

if (( SLURM_ARRAY_TASK_ID < 4 )); then
  policy="gnn"
  cfg_idx=$SLURM_ARRAY_TASK_ID
else
  policy="mlp"
  cfg_idx=$(( SLURM_ARRAY_TASK_ID - 4 ))
fi

entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_skew4_new_models_one.sh "$policy" "$name" "$path"
