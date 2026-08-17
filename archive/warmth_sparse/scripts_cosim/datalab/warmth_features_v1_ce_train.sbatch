#!/bin/bash
#SBATCH --job-name=wfv1-ce
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wfv1-ce-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/wfv1-ce-%j.err
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

bash scripts_cosim/datalab/run_warmth_features_v1_ce_train.sh
