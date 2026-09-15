#!/bin/bash
#SBATCH --job-name=sparse-finish
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sparse-finish-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sparse-finish-%j.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Finisher: sweep ds_00000..ds_00350 with --resume (skip existing best.json).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_sparse_warmth_v2
export GRID=sparse_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1
export START_FROM=0
export MAX_DATASETS=351
export SHARD_LABEL=sparse-finish

bash scripts_cosim/datalab/run_warmth_regen_range.sh
