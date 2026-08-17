#!/bin/bash
#SBATCH --job-name=rb-recache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rb-recache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rb-recache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim
export FORCE_RECACHE="${FORCE_RECACHE:-1}"
bash scripts_cosim/datalab/run_regime_b_recache.sh
