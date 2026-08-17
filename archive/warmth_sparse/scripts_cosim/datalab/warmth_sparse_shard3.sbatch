#!/bin/bash
#SBATCH --job-name=warmth-sparse-s3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-sparse-s3-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-sparse-s3-%j.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Standalone shard 3 for sparse_warmth_v2 (ds_00264–00350).
# 64 CPUs fits alongside 96-CPU array tasks on partially idle a100s nodes.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SHARD_LABEL=sparse-s3
export START_FROM=264
export MAX_DATASETS=87
export OUTPUT_SUBDIR=gnn_datasets_4tasks_sparse_warmth_v2
export GRID=sparse_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_regen_range.sh
