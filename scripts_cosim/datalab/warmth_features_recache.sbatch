#!/bin/bash
#SBATCH --job-name=warmth-recache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-recache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-recache-%j.err
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Merged seq graph cache with B1 features (832 repaired co-sim datasets).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export FORCE_RECACHE=1
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_warmth_features_recache.sh
