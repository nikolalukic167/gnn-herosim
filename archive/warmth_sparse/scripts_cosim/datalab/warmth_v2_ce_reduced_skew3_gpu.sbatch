#!/bin/bash
#SBATCH --job-name=wv2cr-gpu
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wv2cr-gpu-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/wv2cr-gpu-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_v2_ce_reduced_skew3_20260611}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-ce-reduced.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_ce_reduced.pt}"
export TIMEOUT="${TIMEOUT:-3600}"

JOBS=(
  "mlp|default_20_20_p50|simulation_data/space_with_network.json"
  "gnn|default_20_20_p50|simulation_data/space_with_network.json"
  "mlp|default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "gnn|default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "mlp|05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
  "gnn|05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

entry="${JOBS[$SLURM_ARRAY_TASK_ID]}"
policy="${entry%%|*}"
rest="${entry#*|}"
name="${rest%%|*}"
path="${rest#*|}"

bash scripts_cosim/datalab/run_warmth_v2_ce_reduced_skew3_one.sh "$policy" "$name" "$path"
