#!/bin/bash
# usage (login node): fs_cell.sh <pin sha> <off|tcp|fs|fs_serial|fs_tcp> <out dir name> <rung> <arm> <seed>   (NODE=<node> required)
# Re-run ONE gate cell (cc40s19537 / g0<rung>) at <pin> through that pin's own r1a_gate.sbatch, inputs exactly comb/in_comball.
# off: no exchange switch; tcp: R1A_EXCHANGE_TCP=1 only; fs: R1A_EXCHANGE_FAIRSHARE=1; fs_serial: + R1A_EXCHANGE_FAIRSHARE_PARTNERS=serial; fs_tcp: fair share + R1A_EXCHANGE_TCP=1.
set -euo pipefail
PIN=$1; MODE=$2; NAME=$3; RUNG=$4; ARM=$5; SEED=$6
U=/share/nikola.lukic/uncut_groups; OUT=$U/anomaly/fairshare/$NAME; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
mkdir -p $U/anomaly/fairshare/logs
WTP=$HOME/gnn-herosim-wt/s6_fs_${PIN:0:8}
git -C $HOME/gnn-herosim fetch -q origin
[ -d $WTP ] || git -C $HOME/gnn-herosim worktree add -q --detach $WTP $PIN
[[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree HEAD"; exit 2; }
grep -q 'R1A_EXCHANGE_FAIRSHARE' $WTP/scripts_cosim/datalab/r1a_gate.sbatch || { echo "FAIL LOUD: pass-through missing at pin"; exit 2; }
( export WT=$WTP EXPECT_HEAD=${PIN:0:8} IN=$U/comb/in_comball OUT=$OUT SELECT=$U/arm1_selected.json R1A_ARMS=$ARM R1A_SEEDS=$SEED R1A_TOPOS=19537 R1A_WINDOWS=g0 WF1_RUNGS=$RUNG PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
  unset R1A_EXCHANGE_FAIRSHARE R1A_EXCHANGE_FAIRSHARE_PARTNERS R1A_EXCHANGE_TCP
  case $MODE in
    off) ;;
    tcp) export R1A_EXCHANGE_TCP=1 ;;
    fs) export R1A_EXCHANGE_FAIRSHARE=1 ;;
    fs_serial) export R1A_EXCHANGE_FAIRSHARE=1 R1A_EXCHANGE_FAIRSHARE_PARTNERS=serial ;;
    fs_tcp) export R1A_EXCHANGE_FAIRSHARE=1 R1A_EXCHANGE_TCP=1 ;;
    *) echo "FAIL LOUD: mode $MODE"; exit 2 ;;
  esac
  sbatch --parsable --export=ALL -p CPU-amd -w ${NODE:?set NODE} --cpus-per-task=2 --mem=24G -t 02:00:00 -J s6-fs-$NAME -o $U/anomaly/fairshare/logs/%x-%j.out $WTP/scripts_cosim/datalab/r1a_gate.sbatch )
