#!/bin/bash
#SBATCH --job-name=warmth-tail
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-tail-%x-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-tail-%x-%j.err
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

# SHARD_LABEL, START_FROM, MAX_DATASETS set by submit script (--export=ALL)
bash scripts_cosim/datalab/run_warmth_regen_range.sh
