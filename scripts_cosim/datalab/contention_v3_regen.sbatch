#!/bin/bash
#SBATCH --job-name=cont-v3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/contention-v3-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/contention-v3-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-9

# contention_v3: conn 0.15/0.20 + heavier queues (900 ds, 10 shards × 90).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_contention_v3
export GRID=contention_v3
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=900
export NUM_SHARDS=10
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=1
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
