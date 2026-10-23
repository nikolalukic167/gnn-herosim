#!/bin/bash
# Old-arm shard 0/4 (job 903157) never started (needs 60 free CPUs). Cancelled by the expert's ruling 2026-10-23 and resubmitted as the two shards 0/8 and 4/8 (their union is exactly cells[0::4]), -c 22 / PAR 11 each, no node pin.
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; R=$U/own24; PIN=05f52abb; WL=$HOME/gnn-herosim-wt/s5_own24_$PIN; IN=$R/in_old; T=$(seq -s, 20508 20531); SEL=$U/own24_selected.json
[ "$(squeue -j 903157 -h -o %T)" = PENDING ] || { echo "FAIL LOUD: 903157 is not PENDING"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + 1 )) -le 44 ] || { echo "FAIL LOUD: cap"; exit 2; }
scancel 903157
for i in 0 4; do
  j=$(env IN=$IN SELECT=$SEL R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=11 CELL_MEM=8G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped \
    WT=$WL EXPECT_HEAD=$PIN OUT=$R/cells_old R1A_ARMS=ra_gnn_eng_combfl_nosplit_k100,ra_twin_eng_combfl_nosplit_k100 R1A_SEEDS=1,2 R1A_SHARD=$i/8 \
    sbatch --parsable --export=ALL -p CPU-amd -c 22 --mem=50G -t 10:00:00 -J s5-own24-old-$i-of8 -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "old shard $i/8 job $j"
done
