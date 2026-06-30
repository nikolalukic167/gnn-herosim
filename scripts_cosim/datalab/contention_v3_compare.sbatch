#!/bin/bash
#SBATCH --job-name=cv3-cmp
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv3-cmp-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv3-cmp-%j.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

cd /home/nikola.lukic/gnn-herosim
export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v3_live_gate_20260620}"
bash scripts_cosim/datalab/run_contention_v3_compare.sh
