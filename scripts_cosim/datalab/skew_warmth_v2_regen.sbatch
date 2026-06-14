#!/bin/bash
#SBATCH --job-name=skew-warmth
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/skew-warmth-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/skew-warmth-%A_%a.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# skew_warmth_v2: degree_skewed_core + 5/30ms, 288 ds, 4 array shards (~72 ds each).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export REMAINING_START=0
export TOTAL_DATASETS=288
export NUM_SHARDS=4
export OUTPUT_SUBDIR=gnn_datasets_4tasks_skew_warmth_v2
export GRID=skew_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_regen_shard.sh
