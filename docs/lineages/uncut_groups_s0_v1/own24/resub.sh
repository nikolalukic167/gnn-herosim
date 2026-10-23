#!/bin/bash
# Resubmit two never-started shards (PENDING ReqNodeNotAvail on slurm-18) without a node pin, PAR 16 / -c 32, same shard index, same OUT, same everything else. Cancelled by the expert's ruling 2026-10-23.
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; R=$U/own24; PIN=05f52abb; WL=$HOME/gnn-herosim-wt/s5_own24_$PIN; IN=$R/in_new; T=$(seq -s, 20508 20531); SEL=$U/own24_selected.json
for j in 903142 903147; do [ "$(squeue -j $j -h -o %T)" = PENDING ] || { echo "FAIL LOUD: $j is not PENDING"; exit 2; }; done
scancel 903142 903147
sub() { # name arms out shardidx nshards
  env IN=$IN SELECT=$SEL R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=16 CELL_MEM=8G LIMIT=14400 R1A_SPLIT=uncut_dagger_full_split.json R1A_SLATE=declared_pruning_v1_uncapped \
    WT=$WL EXPECT_HEAD=$PIN OUT=$3 R1A_ARMS=$2 R1A_SEEDS=1,2,3,4,5,6 R1A_SHARD=$4/$5 \
    sbatch --parsable --export=ALL -p CPU-amd -c 32 --mem=64G -t 10:00:00 -J s5-own24-$1-$4 -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch
}
G=ra_gnn_eng_combfl_dgu_nosplit; W=ra_twin_eng_combfl_dgu_nosplit
echo "newg shard 4/6 job $(sub newg ${G}_k100,${G}_kr $R/cells_new_gnn 4 6)"
echo "newt shard 3/5 job $(sub newt ${W}_k100,${W}_kr $R/cells_new_twin 3 5)"
