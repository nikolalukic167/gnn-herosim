#!/bin/bash
#SBATCH --job-name=cont-v2-fin
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/contention-v2-finish-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/contention-v2-finish-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# Finish 22 timed-out contention_v2 datasets (conn=0.35 heavy combos).
# --only-missing-jsonl re-runs only ds_* lacking placements.jsonl.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_contention_v2
export GRID=contention_v2
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=900
export NUM_SHARDS=4
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=1
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
