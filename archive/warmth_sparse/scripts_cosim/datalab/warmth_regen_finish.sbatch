#!/bin/bash
#SBATCH --job-name=warmth-finish
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-finish-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-finish-%j.err
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-0

# Single array task: sweep ds_00000..ds_00499 with --resume (skip existing best.json).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1
export START_FROM=0
export MAX_DATASETS=500
export SHARD_LABEL=finish
export TOTAL_DATASETS=500
export NUM_SHARDS=1
export REMAINING_START=0

bash scripts_cosim/datalab/run_warmth_regen_range.sh
