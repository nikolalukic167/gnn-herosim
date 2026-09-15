#!/bin/bash
#SBATCH --job-name=warmth-repair-sparse
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-repair-sparse-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-repair-sparse-%A_%a.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-3

# --repair --force on gnn_datasets_4tasks_sparse_warmth_v2 (351 ds).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export REMAINING_START=0
export TOTAL_DATASETS=351
export NUM_SHARDS=4
export OUTPUT_SUBDIR=gnn_datasets_4tasks_sparse_warmth_v2
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_repair_shard.sh
