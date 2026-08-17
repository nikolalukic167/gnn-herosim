#!/bin/bash
#SBATCH --job-name=ws-merged-cr
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ws-merged-cr-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ws-merged-cr-%j.err
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

ulimit -n 65536 2>/dev/null || true

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

bash scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_train.sh
