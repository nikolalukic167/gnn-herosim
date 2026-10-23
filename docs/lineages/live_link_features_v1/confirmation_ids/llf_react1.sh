#!/bin/bash
# live_link_features_v1 confirmation, replacement id 30297 (for 30288): its 8 reactive admission cells under option D (cap200 inputs, TCP=1 FAIRSHARE=1), pin a9733951, LIMIT 43200, one cell per job, spread over the 4 freest nodes.
# usage: llf_react1.sh <cap200 inputs_group MD5SUMS md5>
set -euo pipefail
GM5=${1:?md5}; PIN=a9733951; ID=30297
U=/share/nikola.lukic/uncut_groups; X=$U/llf_conf_r1; G=$X/gate_inputs_cap200/inputs_group; IN=$X/in_cap200; OUT=$X/cells_react; SEL=$X/selected.json
[[ $(md5sum < $G/MD5SUMS | cut -c1-32) == $GM5 ]] || { echo "FAIL LOUD: cap200 inputs_group MD5SUMS"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: md5sum -c"; exit 2; }
[ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
python3 -c "import json;json.dump({'verdict':'DESIGN-READY','lineage':'live_link_features_v1','clients':160,'servers':24,'connection_probability':0.6,'topologies':[$ID],'windows':['g0','g1','g2','g3']},open('$SEL','w'),indent=1)"
WL=$HOME/gnn-herosim-wt/s5_llf_${PIN}
[[ $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WL"; exit 2; }
nodes=(); while read idle n; do [ "$n" = os-cpu-slurm-21 ] && continue; [ $idle -ge 2 ] || continue; nodes+=($n); [ ${#nodes[@]} -ge 4 ] && break
done < <(sinfo -p CPU-amd -h -N -o "%N %C" | sort -u | awk '{split($2,a,"/"); print a[2], $1}' | sort -rn)
echo "nodes: ${nodes[*]}"; [ ${#nodes[@]} -ge 4 ] || { echo "FAIL LOUD: fewer than 4 nodes"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + 8 )) -le 44 ] || { echo "FAIL LOUD: 8 more jobs would exceed 44"; exit 2; }
k=0
for rung in moderate heavy; do for w in g0 g1 g2 g3; do n=${nodes[$((k % 4))]}; k=$((k+1))
  j=$(env IN=$IN SELECT=$SEL R1A_TOPOS=$ID R1A_WINDOWS=$w WF1_RUNGS=$rung PAR=1 CELL_MEM=16G LIMIT=43200 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_EXCHANGE_TCP=1 R1A_EXCHANGE_FAIRSHARE=1 \
      WT=$WL EXPECT_HEAD=$PIN OUT=$OUT R1A_ARMS=reactive R1A_SEEDS=1,2 \
      sbatch --parsable --export=ALL -p CPU-amd -w $n -c 2 --mem=20G -t 14:00:00 -J s5-llf1-$w$rung -o $X/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "$ID $w $rung node $n job $j"; done; done
