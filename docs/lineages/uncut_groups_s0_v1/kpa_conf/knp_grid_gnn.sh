#!/bin/bash
# usage (login node): knp_grid_gnn.sh <shared pin sha> <node_gnn_mod> <node_gnn_heavy>
# The two FL gnn roots of knp_grid.sh (md5 dfc029f1), unchanged, resubmitted alone on two of S6s nodes (the first submission waited on 18 and 22).
# KPA closed-loop sensitivity: 288 cells with R1A_KPA_KNATIVE_PANIC=1 at the shared pin, one root per exclusive node.
# Everything else exactly as S4's 0.7 arms: gate topologies 19537-19548, g0-g3, inputs comb/in_comball, arm1_selected.json,
# uncut_arm1_split.json, declared_pruning_v1_uncapped.
set -euo pipefail
PIN=$1; shift; NODES=("$@"); [ ${#NODES[@]} = 2 ] || { echo "FAIL LOUD: two nodes"; exit 2; }
for n in "${NODES[@]}"; do [[ $n =~ ^os-cpu-slurm-(21|15|13|14|16|10|17)$ ]] && { echo "FAIL LOUD: excluded node $n"; exit 2; }; done
U=/share/nikola.lukic/uncut_groups; A=$U/anomaly; mkdir -p $A/logs
WTP=$HOME/gnn-herosim-wt/s6_knp_${PIN:0:8}
git -C $HOME/gnn-herosim fetch -q origin
[ -d $WTP ] || git -C $HOME/gnn-herosim worktree add -q --detach $WTP $PIN
[[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree HEAD"; exit 2; }
grep -q 'R1A_KPA_KNATIVE_PANIC' $WTP/scripts_cosim/datalab/r1a_gate.sbatch || { echo "FAIL LOUD: pass-through missing at pin"; exit 2; }
T=$(seq -s, 19537 19548)
i=0
for spec in gnn_mod:moderate:ra_gnn_eng_combfl_nosplit_k100:1,2 gnn_heavy:heavy:ra_gnn_eng_combfl_nosplit_k100:1,2; do
  IFS=: read -r name rung arm seeds <<< "$spec"; OUT=$A/cells_knpanic_$name; n=${NODES[$i]}; i=$((i+1))
  [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  ( export WT=$WTP EXPECT_HEAD=${PIN:0:8} IN=$U/comb/in_comball OUT=$OUT SELECT=$U/arm1_selected.json R1A_ARMS=$arm R1A_SEEDS=$seeds R1A_TOPOS=$T \
      R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=$rung PAR=60 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped \
      R1A_KPA_KNATIVE_PANIC=1
    j=$(sbatch --parsable --export=ALL -p CPU-amd -w $n --exclusive --nodes=1 --mem=0 -t 06:00:00 -J s6-knp-$name -o $A/logs/%x-%j.out $WTP/scripts_cosim/datalab/r1a_gate.sbatch)
    echo "$name rung=$rung arm=$arm node=$n job=$j" )
done
