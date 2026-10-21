#!/bin/bash
# ILS-live / exact-live frozen-w calibration (S7, 2026-10-21): the filed hand-series recipe (cot_comb_steps.sh) for ONE arm on ONE exclusive SMT-off node.
# H3pre (cd_ib0 at 114cc0e7), the arm at the merged pin (k = 0, no charge), H3end. 19461 g0, both rungs, seed 0, PAR 1.
# Usage: WT=<pin worktree> EXPECT_HEAD=<pin8> cot_ils_steps.sh <arm kind, no _k suffix> <series dir>
set -euo pipefail
ARM=${1:?}; SER=${2:?}; W=${WT:?FAIL LOUD: set WT}; HEAD8=${EXPECT_HEAD:?}
C=/share/nikola.lukic/uncut_groups/cotenancy; U=/share/nikola.lukic/uncut_groups
export COT_DIR=$C PAR=1 WF1_RUNGS=${RUNGS:-moderate,heavy} R1A_WINDOWS=g0 LOAD_THREADS=8 R1A_TOPOS=19461
export R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped SELECT=$U/hand_selected.json
# A3: the node must have the CPU model and SMT state of the PER 10 / 50 calibration (WANT_SMT, WANT_CPU_MODEL), else the series is redone elsewhere, not averaged
export EXPECT_SMT=${WANT_SMT:-$(cat /sys/devices/system/cpu/smt/control)}
export EXPECT_CPU_MODEL=${WANT_CPU_MODEL:-$(lscpu | sed -n "s/^Model name:[[:space:]]*//p")}
[[ "$(cat /sys/devices/system/cpu/smt/control)" == "$EXPECT_SMT" && "$EXPECT_SMT" == off ]] || { echo "FAIL LOUD: smt/control=$(cat /sys/devices/system/cpu/smt/control), want $EXPECT_SMT (the recipe is the SMT-off venue)"; exit 1; }
[[ "$(lscpu | sed -n "s/^Model name:[[:space:]]*//p")" == "$EXPECT_CPU_MODEL" ]] || { echo "FAIL LOUD: CPU model differs from $EXPECT_CPU_MODEL"; exit 1; }
echo "[series $ARM] host=$(hostname) smt=$EXPECT_SMT model=\"$EXPECT_CPU_MODEL\" nproc=$(nproc) cpuset=$(taskset -pc $$ | sed "s/.*: //")"
mkdir -p "$SER"
H3IN=$U/gate_inputs_e2738e37/inputs_group
step() {  # name worktree head arm stall_s
  echo "[series $ARM] step $1 start $(date -Is) arm=$4 head=$3"
  env WT="$2" EXPECT_HEAD="$3" N_LOAD=0 PIN=0 IN="$H3IN" R1A_ARMS="$4" R1A_SEEDS=0 CELL_MEM=16G LIMIT=${LIMIT_S:-43200} ${5:+R1A_STALL_S=$5} OUT="$SER/$1" bash "$C/cot_calib_v2.sbatch"
  echo "[series $ARM] step $1 done $(date -Is)"
}
A1=$HOME/gnn-herosim-wt/s6_arm1_114cc0e7
step H3pre $A1 114cc0e7 cd_ib0
step ARM "$W" "$HEAD8" "$ARM" 7200
step H3end $A1 114cc0e7 cd_ib0
echo "[series $ARM] all steps done"
