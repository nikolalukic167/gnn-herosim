#!/bin/bash
#SBATCH --job-name=s4-reqstats-conf
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/logs/%x-%j.out
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd /share/nikola.lukic/uncut_groups/anomaly || exit 1
echo "node $(hostname) $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
rc=0
for S in A B; do python3 request_stats_conf.py $S || rc=1; done
exit $rc
