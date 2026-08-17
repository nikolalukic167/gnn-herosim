#!/bin/bash
#SBATCH --job-name=fc-siv1-mlp
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/fc-siv1-mlp-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/fc-siv1-mlp-%j.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

cd /home/nikola.lukic/gnn-herosim
bash scripts_cosim/datalab/run_full_corpus_siv1_mlp_train.sh
