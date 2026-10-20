#!/bin/bash
#SBATCH --job-name=load-anomaly-ext
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/load-anomaly-ext-%j.out
# Additions (a)-(c) to load_anomaly.py: filed summaries plus the cells' per-task lock sidecars (~2 GB of reads, so not on the login node).
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd /share/nikola.lukic/uncut_groups
[[ $(md5sum < $HOME/s6_scripts/load_anomaly_ext.py | cut -c1-32) == "$EXT_MD5" ]] || { echo "FAIL LOUD: script md5"; exit 1; }
python3 $HOME/s6_scripts/load_anomaly_ext.py || { echo "FAIL rc=$?"; exit 1; }
echo "[ext] load_anomaly_ext.txt md5 $(md5sum < anomaly/load_anomaly_ext.txt | cut -c1-32)"
