#!/bin/bash
# S4. Identity reruns for the confirmation-sets request-level read (usage: req_ident_conf.sh [cases, default "A_bar A_gnn B_bar B_gnn B_desc"]): one cell per arm per set (first topology, g2, heavy), each at its cells' own pin and
# launched as the filed cells were (conf_cells.sh / conf2_cells.sh COMMON line), nothing added. B_gnn = FL gnn seed 3 (conf2/cells_learned), B_desc = FL gnn seed 1 (conf2/cells_desc).
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; A=$U/anomaly; mkdir -p $A/logs
PB=a459dd986e7abde358c15d9c45d5f464b96e0d2d; PL=74ddf70e1effbdc1eea0a69b8573c81a5f23384e; WB=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98; WL=$HOME/gnn-herosim-wt/s6_p2_74ddf70e
for k in ${1:-A_bar A_gnn B_bar B_gnn B_desc}; do
  case $k in
    A_*) IN=$U/conf/in_conf; SEL=$U/fl_selected.json; TOPO=19707;;
    B_*) IN=$U/conf2/in_conf2; SEL=$U/fl2_selected.json; TOPO=19953;;
  esac
  case $k in
    *_bar) PIN=$PB; WT=$WB; ARM=cd_ib0_inflightfull_k100; SEEDS=1,2;;
    A_gnn|B_desc) PIN=$PL; WT=$WL; ARM=ra_gnn_eng_combfl_nosplit_k100; SEEDS=1;;
    B_gnn) PIN=$PL; WT=$WL; ARM=ra_gnn_eng_combfl_nosplit_k100; SEEDS=3;;
  esac
  [[ $(git -C $WT rev-parse HEAD) == $PIN && $(git -C $WT status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WT"; exit 2; }
  OUT=$A/req_ident_conf_$k; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  ( export WT EXPECT_HEAD=${PIN:0:8} IN OUT SELECT=$SEL R1A_ARMS=$ARM R1A_SEEDS=$SEEDS R1A_TOPOS=$TOPO R1A_WINDOWS=g2 WF1_RUNGS=heavy PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
    j=$(sbatch --parsable --export=ALL -p CPU-amd --exclusive --nodes=1 --mem=0 -t 04:00:00 -x os-cpu-slurm-[21,15,13,14,16,10,17] -J s4-req-ident-conf-$k -o $A/logs/%x-%j.out $WT/scripts_cosim/datalab/r1a_gate.sbatch); echo "ident conf $k job $j" )
done
