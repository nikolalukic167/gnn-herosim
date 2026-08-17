#!/bin/bash
#SBATCH --job-name=rev-tri-gnn-ce
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rev-tri-gnn-ce-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rev-tri-gnn-ce-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-6

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

# Must match reviewer_triangle MLP/XGB config order.
CONFIGS=(
  "default_20_20_p50:simulation_data/space_with_network.json"
  "00_balanced_30_30_p35:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "01_balanced_40_40_p50:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/01_balanced_40_40_p50.json"
  "02_balanced_50_50_p60:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/02_balanced_50_50_p60.json"
  "03_client_heavy_50_35_p50:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/03_client_heavy_50_35_p50.json"
  "04_server_heavy_35_50_p50:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/04_server_heavy_35_50_p50.json"
  "05_sparse_40_40_p25:simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
)

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%:*}"
path="${entry#*:}"

bash scripts_cosim/datalab/run_reviewer_triangle_gnn_dim14_ce_one.sh "$name" "$path"
