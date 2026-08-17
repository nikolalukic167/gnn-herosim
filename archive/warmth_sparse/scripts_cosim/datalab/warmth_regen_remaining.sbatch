#!/bin/bash
#SBATCH --job-name=warmth-v2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-v2-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-v2-%A_%a.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# Remaining warmth_v2 co-sim regen (ds_00235+), 4 array tasks.
# CPU-only: no --gres (cluster etiquette — don't hog GPUs for brute-force sim).
# 128 cpus/task on 256-CPU a100s nodes → 127 workers, max 2 tasks/node, distinct CPU sets.
# Each task: ~67 datasets from ds_00235+ (--resume skips done).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export REMAINING_START=235
export TOTAL_DATASETS=500
export NUM_SHARDS=4
export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_regen_shard.sh
