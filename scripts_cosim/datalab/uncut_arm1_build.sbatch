#!/bin/bash
# Arm 1 dataset build: 24 shards (6 array tasks x 4 processes) of uncut_arm1_build.py into $CORPUS/all, manifests in $CORPUS/manifests.
#   sbatch --array=0-5 --export=ALL,WT=<pin>,CORPUS=<dir> uncut_arm1_build.sbatch
#SBATCH --job-name=a1-build
#SBATCH --output=/share/nikola.lukic/uncut_arm1/build-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -uo pipefail
WT=${WT:?}; CORPUS=${CORPUS:?}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd "$WT"; export PYTHONPATH=$PWD HEROSIM_PY=python3; git rev-parse HEAD
mkdir -p $CORPUS/all $CORPUS/manifests
R=/share/nikola.lukic/uncut_arm1
for j in 0 1 2 3; do
  S=$((SLURM_ARRAY_TASK_ID * 4 + j))
  python3 scripts_cosim/uncut_arm1_build.py $R/arm1_state_ids.json $R/labels/labels_s*.jsonl --out $CORPUS/all --shard $S/24 --manifest $CORPUS/manifests/build_s$S.jsonl > $CORPUS/manifests/log_s$S.txt 2>&1 &
done
wait
