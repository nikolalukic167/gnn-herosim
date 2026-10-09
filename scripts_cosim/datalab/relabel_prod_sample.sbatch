#!/bin/bash
# Registered measurement (coordinator node 5, 2026-10-09): 200 production datasets per rung relabelled at the pinned replay vs the stored E labels.
# usage: WT=<worktree> EXPECT_HEAD=<sha> OUTDIR=<scratch dir> sbatch --export=ALL ...   (reads the production E datasets, writes only to OUTDIR)
#SBATCH --job-name=relabelprod
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/relabelprod-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=48
#SBATCH --mem=190G
#SBATCH --time=10:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; OUTDIR=${OUTDIR:?}; cd $WT
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
[[ $(git rev-parse HEAD) == $EXPECT_HEAD* ]] || { echo "HEAD is not $EXPECT_HEAD"; exit 1; }
if ! timeout 300 git diff --quiet HEAD -- src scripts_cosim; then echo "WARN: worktree dirty or git diff timed out (300 s)"; fi
export HEROSIM_PY=python3 PYTHONPATH=$WT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= SIM_FORCE_FULL_STATS=1 PYTHONHASHSEED=0
P=/share/nikola.lukic/workload_fix_v1/corpus_prod
mkdir -p $OUTDIR
python3 scripts_cosim/relabel_prod_sample.py sample $OUTDIR/list.txt --roots $P/gnn_datasets_wf1_train_E8894a08,$P/gnn_datasets_wf1_heldout_E8894a08 --per-rung 200 --seed 20261009
wc -l $OUTDIR/list.txt
python3 scripts_cosim/relabel_plans.py $OUTDIR/relabel.json --datasets $OUTDIR/list.txt --max-plans 300 --workers 44 > $OUTDIR/relabel.txt 2>&1
python3 scripts_cosim/relabel_prod_sample.py report $OUTDIR/relabel.json $OUTDIR/list.txt | tee $OUTDIR/report.txt
