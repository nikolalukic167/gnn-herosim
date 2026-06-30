#!/bin/bash
#SBATCH --job-name=cv3-recache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv3-recache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv3-recache-%j.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

cd /home/nikola.lukic/gnn-herosim
bash scripts_cosim/datalab/run_contention_v3_recache.sh
