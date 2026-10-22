#!/bin/bash
# S4. Identity reruns for the 64/128-server request-level read (usage: req_ident_size.sh 64|128 [arms, default "bar gnn mlp"]): one cell per arm (first topology, g2, heavy), each at its cells' own pin and launched as
# the filed cells were (size64b_cells.sh / size128_cells.sh / s128_mlp.sh COMMON line), nothing added.
set -euo pipefail
S=${1:?size 64|128}; ARMS=${2:-"bar gnn mlp"}
U=/share/nikola.lukic/uncut_groups; A=$U/anomaly; mkdir -p $A/logs
if [ $S = 64 ]; then TOPO=30129; SEL=$U/size64b_selected.json; IN_BG=$U/size64/in_size64; IN_M=$IN_BG; else TOPO=30241; SEL=$U/size128_selected.json; IN_BG=$U/size128/in_size128; IN_M=$U/size128/in_mlp128; fi
for k in $ARMS; do
  case $k in
    bar) PIN=a459dd986e7abde358c15d9c45d5f464b96e0d2d; WT=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98; ARM=cd_ib0_inflightfull_k100; SEEDS=1,2; IN=$IN_BG;;
    gnn) PIN=74ddf70e1effbdc1eea0a69b8573c81a5f23384e; WT=$HOME/gnn-herosim-wt/s6_p2_74ddf70e; ARM=ra_gnn_eng_combfl_nosplit_k100; SEEDS=1; IN=$IN_BG;;
    mlp) PIN=74ddf70e1effbdc1eea0a69b8573c81a5f23384e; WT=$HOME/gnn-herosim-wt/s6_p2_74ddf70e; ARM=ra_mlp_same_combfl_nosplit_k100; SEEDS=1; IN=$IN_M;;
  esac
  [[ $(git -C $WT rev-parse HEAD) == $PIN && $(git -C $WT status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WT"; exit 2; }
  OUT=$A/req_ident_s${S}_$k; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  ( export WT EXPECT_HEAD=${PIN:0:8} IN OUT SELECT=$SEL R1A_ARMS=$ARM R1A_SEEDS=$SEEDS R1A_TOPOS=$TOPO R1A_WINDOWS=g2 WF1_RUNGS=heavy PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
    j=$(sbatch --parsable --export=ALL -p CPU-amd --exclusive --nodes=1 --mem=0 -t 04:00:00 -x os-cpu-slurm-[21,15,13,14,16,10,17] -J s4-req-ident-s$S-$k -o $A/logs/%x-%j.out $WT/scripts_cosim/datalab/r1a_gate.sbatch); echo "ident s$S $k job $j" )
done
