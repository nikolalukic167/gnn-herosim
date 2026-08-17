#!/bin/bash
#SBATCH --job-name=skew4-kn-v2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew4-kn-v2-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew4-kn-v2-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-3

# Experiment Group 4b: skew4 Knative baseline with node_disk_v2, 125-225 workload.
# Note: existing dim14_old_models_skew4_125225_20260610/ used LEGACY physics (no node_disk_v2).
# These 4 runs provide a fair Knative baseline for the wssm model comparison.
# Submit: sbatch scripts_cosim/datalab/skew4_new_models_knative.sbatch

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

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_skew4_new_models_one.sh "knative" "$name" "$path"
