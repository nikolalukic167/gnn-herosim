#!/bin/bash
#SBATCH --job-name=warmth-repair
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-repair-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-repair-%A_%a.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-3

# --repair --force on gnn_datasets_4tasks_1060_warmth_v2 (500 ds, ~127s/ds).
# CPU-only: sequential re-sim per dataset; 4 array shards (~125 ds each).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export REMAINING_START=0
export TOTAL_DATASETS=500
export NUM_SHARDS=4
export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_repair_shard.sh
