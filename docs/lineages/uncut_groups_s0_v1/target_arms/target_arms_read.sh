#!/bin/bash
#SBATCH --job-name=s6-target-read
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/s6-target-read-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $HOME/s6_scripts/target_arms
[[ $(md5sum < target_arms_read.py | cut -c1-32) == ec8f7490ad386112272bfbb0c13ce607 ]] || { echo "FAIL LOUD: target_arms_read.py md5"; exit 1; }
python3 target_arms_read.py target_arms_cells.txt || { echo "read FAILED"; exit 1; }
echo "[read] target_arms_read.txt md5 $(md5sum < /share/nikola.lukic/uncut_groups/anomaly/target_arms_read.txt | cut -c1-32)"
