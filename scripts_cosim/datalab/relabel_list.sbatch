#!/bin/bash
# Re-evaluate plans of fidelity-replay datasets at a pinned worktree (scripts_cosim/relabel_plans.py).
# usage: WT=<worktree> EXPECT_HEAD=<sha> LIST=<file of dataset dirs> OUT=<out.json> [MAXPLANS=300] [EXACT=1] sbatch --export=ALL ...
#SBATCH --job-name=relabel
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/relabel-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=48
#SBATCH --mem=190G
#SBATCH --time=08:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; LIST=${LIST:?}; OUT=${OUT:?}; cd $WT
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
[[ $(git rev-parse HEAD) == $EXPECT_HEAD* ]] || { echo "HEAD is not $EXPECT_HEAD"; exit 1; }; [[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "worktree dirty"; exit 1; }
export HEROSIM_PY=python3 PYTHONPATH=$WT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= SIM_FORCE_FULL_STATS=1 PYTHONHASHSEED=0
export HEROSIM_REPLAY_EXACT_BATCH=${EXACT:-1}
mkdir -p "$(dirname $OUT)"
python3 scripts_cosim/relabel_plans.py $OUT --datasets $LIST --max-plans ${MAXPLANS:-300} --workers 44 > ${OUT%.json}.txt 2>&1
tail -5 ${OUT%.json}.txt | cut -c1-200
