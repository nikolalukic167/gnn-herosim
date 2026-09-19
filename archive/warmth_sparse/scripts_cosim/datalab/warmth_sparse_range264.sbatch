#!/bin/bash
#SBATCH --job-name=warmth-sp264
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-sp264-%x-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-sp264-%x-%j.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# One shard of sparse_warmth_v2 range ds_00264–00350.
# SHARD_LABEL, START_FROM, MAX_DATASETS set by submit script (--export=ALL).

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

export OUTPUT_SUBDIR=gnn_datasets_4tasks_sparse_warmth_v2
export GRID=sparse_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_regen_range.sh
