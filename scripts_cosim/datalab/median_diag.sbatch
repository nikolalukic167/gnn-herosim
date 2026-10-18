#!/bin/bash
#SBATCH --job-name=median-diag
#SBATCH --output=/share/nikola.lukic/uncut_groups/median_diag/median-diag-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
O=/share/nikola.lukic/uncut_groups/median_diag
echo "node $(hostname) job $SLURM_JOB_ID"
python3 $O/median_diag.py $O
md5sum $O/median_diag.json $O/median_diag.txt $O/median_diag.py
