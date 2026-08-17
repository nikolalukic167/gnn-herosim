#!/bin/bash
#SBATCH --job-name=skew3-kn
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew3-kn-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew3-kn-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2

# Experiment Group 3b: skew3 Knative baseline (node_disk_v2), no GPU needed.
# Only 2/3 configs missing locally (degree_skew + 05_sparse_degree_skew).
# All 3 run here so the sweep dir is self-contained (skip-if-exists handles overlap).
# Submit: sbatch scripts_cosim/datalab/skew3_full_gate_knative.sbatch

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

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_skew3_full_gate_one.sh "knative" "$name" "$path"
