#!/bin/bash
# usage (login node): knp_ident.sh <pin sha> <0|1 switch> <out dir name> <rung> <arm> <seed>
# Re-run ONE gate cell (cc40s19537 / g0<rung>) at <pin> through that pin's own r1a_gate.sbatch, inputs exactly comb/in_comball,
# with R1A_KPA_KNATIVE_PANIC unset (0) or 1. NODE=<one of S6 nodes 18, 19, 20, 22 (split agreed with S4)>.
set -euo pipefail
PIN=$1; SW=$2; NAME=$3; RUNG=$4; ARM=$5; SEED=$6
U=/share/nikola.lukic/uncut_groups; OUT=$U/kpa_conf/$NAME; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
WTP=$HOME/gnn-herosim-wt/s6_knp_${PIN:0:8}
git -C $HOME/gnn-herosim fetch -q origin
[ -d $WTP ] || git -C $HOME/gnn-herosim worktree add -q --detach $WTP $PIN
[[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree HEAD"; exit 2; }
( export WT=$WTP EXPECT_HEAD=${PIN:0:8} IN=$U/comb/in_comball OUT=$OUT SELECT=$U/arm1_selected.json R1A_ARMS=$ARM R1A_SEEDS=$SEED R1A_TOPOS=19537 R1A_WINDOWS=g0 WF1_RUNGS=$RUNG PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
  if [[ $SW == 1 ]]; then export R1A_KPA_KNATIVE_PANIC=1; else unset R1A_KPA_KNATIVE_PANIC; fi
  sbatch --parsable --export=ALL -p CPU-amd -w ${NODE:?set NODE to one of S6s nodes 18,19,20,22} --cpus-per-task=2 --mem=24G -t 01:00:00 -J s6-knp-$NAME -o $U/kpa_conf/%x-%j.out $WTP/scripts_cosim/datalab/r1a_gate.sbatch )
