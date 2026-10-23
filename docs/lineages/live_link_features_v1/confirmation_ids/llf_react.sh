#!/bin/bash
# live_link_features_v1 confirmation ids: reactive admission cells under option D (payload cap 200 MB, cellular 4 MB/s, TCP=1 FAIRSHARE=1), reactive s0, 12 topologies x g0-g3 x moderate/heavy = 96 cells, 4 shards on 4 nodes.
# Rule (live_link_features_v1.md, a9733951): admitted iff all 8 reactive cells of a topology have queue_share <= 0.80; unknown is not a pass; nothing else is read.
# usage: llf_react.sh "<12 ids>" <sha256 of the comma-joined list> <cap200 inputs_group MD5SUMS md5>
set -euo pipefail
IDS=$1; WANT=$2; GM5=$3; PIN=a9733951
U=/share/nikola.lukic/uncut_groups; X=$U/llf_conf; G=$X/gate_inputs_cap200/inputs_group; IN=$X/in_cap200; OUT=$X/cells_react; SEL=$X/selected.json
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256"; exit 2; }
[ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
[[ $(md5sum < $G/MD5SUMS | cut -c1-32) == $GM5 ]] || { echo "FAIL LOUD: cap200 inputs_group MD5SUMS"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: md5sum -c"; exit 2; }
T=$(echo $IDS | tr ' ' ',')
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 -c "import json;json.dump({'verdict':'DESIGN-READY','lineage':'live_link_features_v1','clients':160,'servers':24,'connection_probability':0.6,'topologies':[int(t) for t in '$T'.split(',')],'windows':['g0','g1','g2','g3']},open('$SEL','w'),indent=1)"
WL=$HOME/gnn-herosim-wt/s5_llf_${PIN}
[[ -d $WL ]] || git -C $HOME/gnn-herosim worktree add -q --detach $WL $PIN
[[ $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WL not clean at $PIN"; exit 2; }
NS=4
EXCL=" os-cpu-slurm-21 os-cpu-slurm-13 os-cpu-slurm-14 os-cpu-slurm-15 os-cpu-slurm-16 os-cpu-slurm-10 os-cpu-slurm-17 "
declare -A CPUS PARN MEMG
pairs=$(sinfo -p CPU-amd -h -N -o "%N %C" | sort -u | awk '{split($2,a,"/"); print a[2], $1}' | sort -rn)
nodes=()
while read idle n; do
  [[ "$EXCL" == *" $n "* ]] && continue
  fm=$(scontrol show node $n | awk -F'[= ]+' '/RealMemory/{for(i=1;i<=NF;i++){if($i=="RealMemory")r=$(i+1); if($i=="AllocMem")a=$(i+1)}} END{print int((r-a)/1024)}')
  par=$(( idle / 2 )); [ $par -gt 16 ] && par=16; [ $(( par * 16 + 8 )) -gt $fm ] && par=$(( (fm - 8) / 16 ))
  [ $par -ge 4 ] || continue
  nodes+=($n); CPUS[$n]=$(( par * 2 )); PARN[$n]=$par; MEMG[$n]=$(( par * 16 + 8 ))
  [ ${#nodes[@]} -ge $NS ] && break
done <<< "$pairs"
echo "nodes: ${nodes[*]:-none}"; for n in "${nodes[@]}"; do echo "  $n -c ${CPUS[$n]} PAR ${PARN[$n]} mem ${MEMG[$n]}G"; done
[ ${#nodes[@]} -ge $NS ] || { echo "FAIL LOUD: fewer than $NS usable nodes; not submitting"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + NS )) -le 44 ] || { echo "FAIL LOUD: $NS more jobs would exceed 44"; exit 2; }
mkdir -p $X/logs
COMMON="IN=$IN SELECT=$SEL R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_EXCHANGE_TCP=1 R1A_EXCHANGE_FAIRSHARE=1"
for i in $(seq 0 $((NS-1))); do n=${nodes[$i]}
  j=$(env $COMMON PAR=${PARN[$n]} WT=$WL EXPECT_HEAD=${PIN:0:8} OUT=$OUT R1A_ARMS=reactive R1A_SEEDS=1,2 R1A_SHARD=$i/$NS \
      sbatch --parsable --export=ALL -p CPU-amd -w $n -c ${CPUS[$n]} --mem=${MEMG[$n]}G -t 06:00:00 -J s5-llf-react-$i -o $X/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "react shard $i/$NS node $n job $j"
done
