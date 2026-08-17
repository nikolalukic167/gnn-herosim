#!/bin/bash
#SBATCH --job-name=skew4-knat
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew4-knat-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew4-knat-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-3

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/dim14_old_models_skew4_125225_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"

CONFIGS=(
  "default_20_20_p50|simulation_data/space_with_network.json"
  "05_sparse_40_40_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_skew4_knative_one.sh "$name" "$path"
