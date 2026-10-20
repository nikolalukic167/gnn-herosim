#!/bin/bash
#SBATCH --job-name=s6-openloop-replay
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/fairshare/logs/s6-openloop-replay-%j.out
# usage: sbatch --export=ALL,PIN=<sha> openloop_replay.sh
set -uo pipefail
PIN=${PIN:?}; U=/share/nikola.lukic/uncut_groups; R=$U/anomaly/fairshare/openloop
WTP=$HOME/gnn-herosim-wt/s6_fs_${PIN:0:8}; [[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree"; exit 2; }
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $WTP; export PYTHONPATH=$WTP
python3 $HOME/s6_scripts/rho/openloop_replay.py $R/xfers.jsonl $R/replay.txt || { echo FAILED; exit 1; }
echo "[openloop] replay.txt md5 $(md5sum < $R/replay.txt | cut -c1-32)"
