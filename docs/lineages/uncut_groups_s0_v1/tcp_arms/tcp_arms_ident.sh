#!/bin/bash
#SBATCH --job-name=s6-tcparms-ident
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/s6-tcparms-ident-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $HOME/s6_scripts/tcp_arms
[[ $(md5sum < tcp_arms_ident.py | cut -c1-32) == 374d637d1150562b6a588db7f67d4921 ]] || { echo "FAIL LOUD: tcp_arms_ident.py md5"; exit 1; }
python3 tcp_arms_ident.py tcp_arms_cells.txt || { echo "ident FAILED"; exit 1; }
echo "[ident] tcp_arms_ident_check.txt md5 $(md5sum < /share/nikola.lukic/uncut_groups/anomaly/tcp_arms_ident_check.txt | cut -c1-32)"
