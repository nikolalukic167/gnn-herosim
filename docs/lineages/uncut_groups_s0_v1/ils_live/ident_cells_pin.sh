#!/bin/bash
# identity rerun at the merged pin e8c14124 (the ILS-live / exact-live build touched shared code): bar and FL gnn s1, cc40s19537 g0 heavy, vs the filed dose_7cb483c5 cells
set -euo pipefail
PIN=${1:?pin sha}; TAG=${2:?tag}
P8=${PIN:0:8}; U=/share/nikola.lukic/uncut_groups; A=$U/ils_ident; IN=$U/comb/in_comball; mkdir -p $A/logs
WT=$HOME/gnn-herosim-wt/s7_idp_$P8
[ -d $WT ] || git -C $HOME/gnn-herosim worktree add --detach $WT $PIN >/dev/null
[[ $(git -C $WT rev-parse HEAD) == $PIN ]] || { echo "FAIL LOUD: worktree HEAD"; exit 2; }
[[ $(git -C $WT status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: dirty worktree"; exit 2; }
EXC=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17,os-cpu-slurm-18,os-cpu-slurm-19,os-cpu-slurm-20,os-cpu-slurm-22
for k in bar gnn; do
  if [ $k = bar ]; then ARM=cd_ib0_inflightfull_k100; else ARM=ra_gnn_eng_combfl_nosplit_k100; fi
  OUT=$A/ident_${TAG}_$k; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  ( export WT EXPECT_HEAD=$P8 IN OUT SELECT=$U/arm1_selected.json R1A_ARMS=$ARM R1A_SEEDS=1 R1A_TOPOS=19537 R1A_WINDOWS=g0 WF1_RUNGS=heavy PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_QUEUE_LENGTH=0.7
    j=$(sbatch --parsable --export=ALL -p CPU-amd --exclusive --nodes=1 --mem=0 -t 03:00:00 -x $EXC -J s7-idp-$k -o $A/logs/%x-%j.out $WT/scripts_cosim/datalab/r1a_gate.sbatch); echo "ident $k job $j pin $PIN" )
done
