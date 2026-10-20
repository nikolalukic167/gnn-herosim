#!/bin/bash
#SBATCH --job-name=s6-tcp-read
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/s6-tcp-read-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
[[ $(md5sum < $HOME/s6_scripts/tcp_read.py | cut -c1-32) == c4caac082a46ba09bea30316137a3ac8 ]] || { echo "FAIL LOUD: read script md5"; exit 1; }
python3 $HOME/s6_scripts/tcp_read.py || { echo "READ FAILED"; exit 1; }
echo "[read] tcp_read.txt md5 $(md5sum < /share/nikola.lukic/uncut_groups/anomaly/tcp_read.txt | cut -c1-32)"
