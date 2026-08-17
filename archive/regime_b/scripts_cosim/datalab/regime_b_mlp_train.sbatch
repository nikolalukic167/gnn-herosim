#!/bin/bash
#SBATCH --job-name=rb-mlp
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rb-mlp-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rb-mlp-%j.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim
bash scripts_cosim/datalab/run_regime_b_mlp_train.sh
