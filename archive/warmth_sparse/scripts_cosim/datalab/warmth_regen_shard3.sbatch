#!/bin/bash
#SBATCH --job-name=warmth-v2-s3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-v2-shard3-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-v2-shard3-%j.err
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Standalone shard 3 (ds_00436–00499). CPU-only — no --gres.
# Submitted separately because array task 3 could not get 128 CPUs on GPU-a100s.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SLURM_ARRAY_TASK_ID=3
export REMAINING_START=235
export TOTAL_DATASETS=500
export NUM_SHARDS=4
export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_regen_shard.sh
