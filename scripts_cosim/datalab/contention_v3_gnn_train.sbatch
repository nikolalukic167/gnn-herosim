#!/bin/bash
#SBATCH --job-name=cv3-gnn
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv3-gnn-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv3-gnn-%j.err
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
bash scripts_cosim/datalab/run_contention_v3_gnn_train.sh
