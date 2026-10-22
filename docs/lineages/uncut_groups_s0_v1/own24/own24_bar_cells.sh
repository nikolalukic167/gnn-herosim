#!/bin/bash
# Own-state confirmation bar cells: 24 fresh topologies x g0-g3 x moderate/heavy, gate root uncut_groups/own24. NOT SUBMITTED until the mint is OK and the expert says go.
# Bar: ARMS (default cd_ib0_inflightfull_k100 = 192 cells, the registration; add cd_k100 for 384) seed 0 at G8 a459dd98, 4 shards on 4 exclusive idle nodes by sinfo load.
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; R=$U/own24; IN=$R/in_own24; OUT=$R/cells_bar; mkdir -p $R/logs
[ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
T=$(seq -s, 20508 20531); [[ $(echo -n $T | sha256sum | cut -c1-64) == d4b1fb268c14af65a7c201ef27a97fbfb3a3033a964218cfbf443f060032ac37 ]] || { echo "FAIL LOUD: id list sha256"; exit 2; }
G=$U/gate_inputs_own24/inputs_group; [[ $(md5sum < $G/MD5SUMS | cut -c1-32) == FILL_AFTER_MINT_inputs_group_MD5SUMS_md5 ]] || { echo "FAIL LOUD: inputs_group MD5SUMS"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: inputs_group md5sum -c"; exit 2; }
[[ $(md5sum < $U/own24_selected.json | cut -c1-32) == FILL_AFTER_MINT_selected_md5 ]] || { echo "FAIL LOUD: own24_selected.json"; exit 2; }
if [ ! -e $IN ]; then
  mkdir -p $IN/models
  for w in wf1_heavy wf1_moderate; do ln -s $G/$w $IN/$w; done
  cp -p $U/comb/in_comball/uncut_arm1_split.json $IN/
fi
[[ $(md5sum < $IN/uncut_arm1_split.json | cut -c1-32) == cb4f540d4010cba5b83194fae177cf98 ]] || { echo "FAIL LOUD: split md5"; exit 2; }
WB=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98
[[ $(git -C $WB rev-parse HEAD) == a459dd98* && $(git -C $WB status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: bar worktree"; exit 2; }
nodes=()
for n in $(sinfo -p CPU-amd -h -N -t idle -o "%O %N" | sort -n | awk '{print $2}' | awk '!s[$0]++' | grep -v -x -e os-cpu-slurm-21 -e os-cpu-slurm-13 -e os-cpu-slurm-14 -e os-cpu-slurm-15 -e os-cpu-slurm-16); do
  [ "$(squeue -w $n -h | wc -l)" = 0 ] || continue
  nodes+=($n); [ ${#nodes[@]} -ge 4 ] && break
done
echo "nodes: ${nodes[*]:-none}"
[ ${#nodes[@]} -ge 4 ] || { echo "FAIL LOUD: fewer than 4 idle nodes; not submitting"; exit 2; }
[ $(squeue -u nikola.lukic -h | wc -l) -le 40 ] || { echo "FAIL LOUD: queue over cap"; exit 2; }
COMMON="IN=$IN SELECT=$U/own24_selected.json R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=60 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped"
for i in 0 1 2 3; do n=${nodes[$i]}
  j=$(env $COMMON WT=$WB EXPECT_HEAD=a459dd98 OUT=$OUT R1A_ARMS=${ARMS:-cd_ib0_inflightfull_k100} R1A_SEEDS=1,2 R1A_SHARD=$i/4 \
      sbatch --parsable --export=ALL -p CPU-amd -w $n --exclusive --nodes=1 --mem=0 -t 08:00:00 -J s5-own24-bar-$i -o $R/logs/%x-%j.out $WB/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "bar shard $i node $n job $j"
done
