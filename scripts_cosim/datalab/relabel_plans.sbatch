#!/bin/bash
#SBATCH --job-name=relabel
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/relabel-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=48
#SBATCH --mem=190G
#SBATCH --time=08:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; cd $WT
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
[[ $(git rev-parse HEAD) == $EXPECT_HEAD* ]]; [[ -z "$(git status --porcelain -- src scripts_cosim)" ]]
export HEROSIM_PY=python3 PYTHONPATH=$WT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= SIM_FORCE_FULL_STATS=1
OUT=$HOME/gnn-herosim/simulation_data/workload_fix_v1/relabel_$EXPECT_HEAD; mkdir -p $OUT
echo "== A: ds_03200, pre-fix replay (HEROSIM_REPLAY_EXACT_BATCH=0), must reproduce the stored labels"
HEROSIM_REPLAY_EXACT_BATCH=0 python3 scripts_cosim/relabel_plans.py $OUT/ds3200_old.json --datasets ${DS3200_LIST:?set DS3200_LIST} --max-plans 200 --workers 4
echo "== B: ds_03200 with the fix"
HEROSIM_REPLAY_EXACT_BATCH=1 python3 scripts_cosim/relabel_plans.py $OUT/ds3200_new.json --datasets ${DS3200_LIST:?set DS3200_LIST} --max-plans 200 --workers 4
echo "== C: the 36 datasets with a net-stage ingress ghost on a candidate node, with the fix"
HEROSIM_REPLAY_EXACT_BATCH=1 python3 scripts_cosim/relabel_plans.py $OUT/net36_new.json --datasets ${NET_LIST:?set NET_LIST} --max-plans 300 --workers 36
echo done
