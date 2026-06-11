#!/bin/bash
#SBATCH --job-name=warmth-p50b
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-p50b-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-p50b-%j.err
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export START_FROM=25
export MAX_DATASETS=25
export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2

bash scripts_cosim/datalab/run_warmth_pilot50_second_half.sh
