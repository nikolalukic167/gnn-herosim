#!/bin/bash
#SBATCH --job-name=tt-v1-recache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/tt-v1-recache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/tt-v1-recache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

cd /home/nikola.lukic/gnn-herosim
bash scripts_cosim/datalab/run_topo_transfer_v1_recache.sh
