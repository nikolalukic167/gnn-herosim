#!/bin/bash
# S4. Identity reruns for the request-level read, each at its cells' own pin: one bar, one FL gnn s1, one FL MLP s1 cell (19538 g2 heavy), no env beyond the filed cells' launch.
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; A=$U/anomaly; IN=$U/comb/in_comball; mkdir -p $A/logs
for k in bar gnn mlp; do
  case $k in
    bar) PIN=a459dd986e7abde358c15d9c45d5f464b96e0d2d; WT=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98; ARM=cd_ib0_inflightfull_k100; SEEDS=1,2;;
    gnn) PIN=a868153755b1fd51c66a1e892df85381ba0ce9ce; WT=$HOME/gnn-herosim-wt/s6_comb_a8681537; ARM=ra_gnn_eng_combfl_nosplit_k100; SEEDS=1;;
    mlp) PIN=a868153755b1fd51c66a1e892df85381ba0ce9ce; WT=$HOME/gnn-herosim-wt/s6_comb_a8681537; ARM=ra_mlp_same_combfl_nosplit_k100; SEEDS=1;;
  esac
  [[ $(git -C $WT rev-parse HEAD) == $PIN && $(git -C $WT status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WT"; exit 2; }
  OUT=$A/req_ident_$k; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  ( export WT EXPECT_HEAD=${PIN:0:8} IN OUT SELECT=$U/arm1_selected.json R1A_ARMS=$ARM R1A_SEEDS=$SEEDS R1A_TOPOS=19538 R1A_WINDOWS=g2 WF1_RUNGS=heavy PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
    j=$(sbatch --parsable --export=ALL -p CPU-amd --exclusive --nodes=1 --mem=0 -t 02:00:00 -x os-cpu-slurm-[21,15,13,14,16,10,17] -J s4-req-ident-$k -o $A/logs/%x-%j.out $WT/scripts_cosim/datalab/r1a_gate.sbatch); echo "ident $k job $j" )
done
