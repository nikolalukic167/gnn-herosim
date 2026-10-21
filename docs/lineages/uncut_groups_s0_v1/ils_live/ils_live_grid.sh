#!/bin/bash
# ILS-live / exact-live grid (uncut_groups_s0_v1, "Label search and exact search under the decision charge"): 2 arms x 4 shards = 8 jobs, each an exclusive node.
# usage: ils_live_grid.sh <arm pin sha>   (the w-only pin; the pin must be on origin, the cell list md5 filed, the reader frozen, the bar identity cell passed)
# R1A_STALL_S = 7200 for these two arms only (SCIENTIST). A tripped cell is a failed cell; it is never rerun at a higher limit without SCIENTIST.
set -euo pipefail
PIN=$1; P8=${PIN:0:8}; U=/share/nikola.lukic/uncut_groups; A=$U/ils_live; IN=$U/comb/in_comball; mkdir -p $A/logs
WT=$HOME/gnn-herosim-wt/s7_ilslive_$P8
[ -d $WT ] || { git -C $HOME/gnn-herosim fetch -q origin; git -C $HOME/gnn-herosim worktree add --detach $WT $PIN; }
[[ $(git -C $WT rev-parse HEAD) == $PIN ]] || { echo "FAIL LOUD: worktree HEAD"; exit 2; }
[[ $(git -C $WT status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: dirty worktree"; exit 2; }
[ $(squeue -u nikola.lukic -h | wc -l) -le 0 ] || { echo "FAIL LOUD: queue is not empty (the budget is 8 slots)"; exit 2; }
T=$(seq -s, 19537 19548)
EXC=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17
for arm in ils exact; do
  case $arm in ils) KIND=cd_ib0_ils_k100;; exact) KIND=cd_exactS_ib0_live_k100;; esac
  OUT=$A/cells_$arm; mkdir -p $OUT
  for i in 0 1 2 3; do
    ( export WT EXPECT_HEAD=$P8 IN OUT SELECT=$U/arm1_selected.json R1A_ARMS=$KIND R1A_SEEDS=0 R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=24 CELL_MEM=16G LIMIT=43200 R1A_STALL_S=7200 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_SHARD=$i/4 R1A_QUEUE_LENGTH=0.7
      j=$(sbatch --parsable --export=ALL -p CPU-amd --exclusive --nodes=1 --mem=0 -t 24:00:00 --exclude=$EXC -J s7-ilslive-$arm-$i -o $A/logs/%x-%j.out $WT/scripts_cosim/datalab/r1a_gate.sbatch); echo "$arm shard $i job $j" )
  done
done
