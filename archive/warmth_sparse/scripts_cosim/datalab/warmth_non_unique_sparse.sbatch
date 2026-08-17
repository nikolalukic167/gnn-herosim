#!/bin/bash
#SBATCH --job-name=nu-sparse
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/non-unique-sparse-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/non-unique-sparse-%A_%a.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-3

# Backfill full cartesian (non-unique) placements into sparse_warmth_v2.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_sparse_warmth_v2
export TOTAL_DATASETS=351
export NUM_SHARDS=4
export WORKERS=31
export COSIM_SUPPRESS_SIM_PRINTS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2

bash scripts_cosim/datalab/run_warmth_non_unique_shard.sh
