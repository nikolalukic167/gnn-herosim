#!/bin/bash
#SBATCH --job-name=uniq-probe
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/uniq-probe-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/uniq-probe-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc" 2>/dev/null || true
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn

# Knative-win configs from bipartite v1 (Tier 1 probe)
CONFIGS=(hub_k4_seek50 hub_k6_seek50 hub_k6_seek65)
CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export TIMEOUT="${TIMEOUT:-7200}"

bash scripts_cosim/run_bipartite_uniq_platform_one.sh "$CONFIG_NAME"
