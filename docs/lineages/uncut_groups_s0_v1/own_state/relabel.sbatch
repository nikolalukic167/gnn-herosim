#!/bin/bash
# Own-state corpus stage 2: the simulator relabel of the NEW datasets only (uncut_label_screen.py relabel, version uncut_relabel/1 = the version that built sim_labels_arm1.json). 8 shards.
#   usage: sbatch --array=0-7 --export=ALL,ARM=twin,PIN=<sha8> relabel.sbatch   (the new-only split must exist: dagger_corpus.py split --new-only --arm ${ARM}_s1 ${ARM}_s2 --out $C/split_newonly.json)
#SBATCH --job-name=dag-relabel
#SBATCH --output=/share/nikola.lukic/uncut_groups/dagger/%x-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17
set -uo pipefail
ARM=${ARM:?}; PIN=${PIN:?}; K=${SLURM_ARRAY_TASK_ID:?}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
WT=$HOME/gnn-herosim-wt/dagc_${PIN}; cd "$WT"; export PYTHONPATH=$PWD HEROSIM_PY=python3
[[ "$(git rev-parse HEAD)" == ${PIN}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD)"; exit 1; }
C=/share/nikola.lukic/uncut_groups/dagger/corpus_$ARM
python3 -u scripts_cosim/uncut_label_screen.py relabel --split $C/split_newonly.json --corpus $C --shard $K/8 --out $C/relabel_${K}_of_8.jsonl --workers 4
