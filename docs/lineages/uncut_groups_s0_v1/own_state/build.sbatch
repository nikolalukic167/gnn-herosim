#!/bin/bash
# Own-state corpus stage 1: datasets of one seed, 8 shards (array 0-7). usage: sbatch --array=0-7 --export=ALL,ARM=twin,SEED=1,PIN=<sha8> build.sbatch
#   ARM names the state set (twin | fl); the per-seed dataset base dir is CORPUS/dagger_<ARM>_s<SEED> (parents gnn_datasets_wf1_dagger_<ARM>_s<SEED>/ds_N).
#SBATCH --job-name=dag-build
#SBATCH --output=/share/nikola.lukic/uncut_groups/dagger/%x-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17
set -uo pipefail
ARM=${ARM:?}; SEED=${SEED:?}; PIN=${PIN:?}; K=${SLURM_ARRAY_TASK_ID:?}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
WT=$HOME/gnn-herosim-wt/dagc_${PIN}; cd "$WT"; export PYTHONPATH=$PWD HEROSIM_PY=python3
[[ "$(git rev-parse HEAD)" == ${PIN}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD)"; exit 1; }
D=/share/nikola.lukic/uncut_groups/dagger; L=$D/${ARM}_labels; C=$D/corpus_$ARM
mkdir -p $C/dagger_${ARM}_s$SEED $C/manifests
python3 scripts_cosim/uncut_arm1_build.py $L/states_s$SEED.json $L/seed$SEED/labels/labels_s*.jsonl --root $L/seed$SEED --out $C/dagger_${ARM}_s$SEED --shard $K/8 --manifest $C/manifests/build_s${SEED}_$K.jsonl > $C/manifests/log_s${SEED}_$K.txt 2>&1
echo "rc $?"
