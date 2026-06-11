#!/bin/bash
#SBATCH --job-name=wv2-kn
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wv2-kn-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/wv2-kn-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_v2_physics_skew3_20260611}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export TIMEOUT="${TIMEOUT:-3600}"

CONFIGS=(
  "default_20_20_p50|simulation_data/space_with_network.json"
  "default_20_20_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
  "05_sparse_40_40_p25_degree_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_warmth_v2_physics_skew3_one.sh knative "$name" "$path"
