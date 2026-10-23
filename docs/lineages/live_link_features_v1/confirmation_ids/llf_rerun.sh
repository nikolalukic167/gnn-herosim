#!/bin/bash
# Rerun of reactive admission cells killed by the per-cell LIMIT (infrastructure failure; expert's ruling 2026-10-23, (a)+(b)): same pin a9733951, inputs and D flags, LIMIT 28800, job wall 10 h, PAR 2.
# usage: llf_rerun.sh <topo> <windows csv> <rungs csv> <tag>     The killed .failed.json files were moved (not deleted) to cells_react_killed/ first.
set -euo pipefail
TOPO=$1; WIN=$2; RUNGS=$3; TAG=$4; PIN=a9733951
U=/share/nikola.lukic/uncut_groups; X=$U/llf_conf; WL=$HOME/gnn-herosim-wt/s5_llf_${PIN}
[[ $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + 1 )) -le 44 ] || { echo "FAIL LOUD: queue would exceed 44"; exit 2; }
env IN=$X/in_cap200 SELECT=$X/selected.json R1A_TOPOS=$TOPO R1A_WINDOWS=$WIN WF1_RUNGS=$RUNGS CELL_MEM=16G LIMIT=28800 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_EXCHANGE_TCP=1 R1A_EXCHANGE_FAIRSHARE=1 \
  PAR=2 WT=$WL EXPECT_HEAD=${PIN:0:8} OUT=$X/cells_react R1A_ARMS=reactive R1A_SEEDS=1,2 \
  sbatch --parsable --export=ALL -p CPU-amd -c 4 --mem=40G -t 10:00:00 -J s5-llf-rr-$TAG -o $X/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch
