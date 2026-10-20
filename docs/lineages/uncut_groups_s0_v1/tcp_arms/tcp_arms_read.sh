#!/bin/bash
#SBATCH --job-name=s6-tcparms-read
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/s6-tcparms-read-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $HOME/s6_scripts/tcp_arms
[[ $(md5sum < tcp_arms_read.py | cut -c1-32) == c0d46bca0a04b89ab5934e9ae105b647 ]] || { echo "FAIL LOUD: tcp_arms_read.py md5"; exit 1; }
python3 tcp_arms_read.py tcp_arms_cells.txt || { echo "read FAILED"; exit 1; }
echo "[read] tcp_arms_read.txt md5 $(md5sum < /share/nikola.lukic/uncut_groups/anomaly/tcp_arms_read.txt | cut -c1-32)"
