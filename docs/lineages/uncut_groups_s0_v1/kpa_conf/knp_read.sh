#!/bin/bash
#SBATCH --job-name=knp-read
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --nodelist=os-cpu-slurm-19
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/knp-read-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
[[ $(md5sum < $HOME/s6_scripts/knp_read.py | cut -c1-32) == e15677cd8e7765fe9a28cbe20a2715c6 ]] || { echo "FAIL LOUD: read script md5"; exit 1; }
python3 $HOME/s6_scripts/knp_read.py || { echo "READ FAILED"; exit 1; }
echo "[read] knpanic_read.txt md5 $(md5sum < /share/nikola.lukic/uncut_groups/anomaly/knpanic_read.txt | cut -c1-32)"
