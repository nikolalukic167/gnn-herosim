#!/bin/bash
#SBATCH --job-name=s6-rho-report
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/fairshare/logs/s6-rho-report-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 $HOME/s6_scripts/rho/rho_report.py || { echo FAILED; exit 1; }
echo "[rho_report] md5 $(md5sum < /share/nikola.lukic/uncut_groups/anomaly/fairshare/rho_report.txt | cut -c1-32)"
