#!/bin/bash
#SBATCH --job-name=skew4-old
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew4-old-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew4-old-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/dim14_old_models_skew4_20260610}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"

JOBS=(
  "gnn|default_20_20_p50|simulation_data/space_with_network.json"
  "mlp|default_20_20_p50|simulation_data/space_with_network.json"
  "gnn|05_sparse_40_40_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "mlp|05_sparse_40_40_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "gnn|default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "mlp|default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "gnn|05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
  "mlp|05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

entry="${JOBS[$SLURM_ARRAY_TASK_ID]}"
policy="${entry%%|*}"
rest="${entry#*|}"
name="${rest%%|*}"
path="${rest#*|}"

bash scripts_cosim/datalab/run_dim14_old_models_skew4_one.sh "$policy" "$name" "$path"
