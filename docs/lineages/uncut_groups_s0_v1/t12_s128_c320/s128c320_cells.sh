#!/bin/bash
# 128 x 320 test cells, one pin for all arms. usage: s128c320_cells.sh cells|react <inputs_group MD5SUMS md5> [PIN=b092fad2]
#   cells: bar cd_ib0_inflightfull_k100 s0 + FL gnn ra_gnn_eng_combfl_nosplit_k100 s1-2 + FL MLP ra_mlp_same_combfl_nosplit_k100 s1-2, 12 topologies x g0-g3 x moderate/heavy = 480 cells, 4 shards, non-exclusive on the 4 freest allowed nodes.
#   react: reactive s0, 96 cells, admission only (queue share <= 0.80, unknown is not a pass), 1 job.
# NOT SUBMITTED until the mint is signed and OK and the expert says go; fails loud on the queue cap (44).
set -euo pipefail
KIND=${1:?cells|react}; GM5=${2:?inputs_group MD5SUMS md5}; PIN=${3:-b092fad2}
U=/share/nikola.lukic/uncut_groups; R=$U/t12_s128_c320; G=$U/gate_inputs_s128_c320_t12/inputs_group; IN=$R/in; OUT=$R/cells_$KIND; SEL=$R/selected.json
mkdir -p $R/logs; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
T=$(seq -s, 30264 30275); [[ $(echo -n $T | sha256sum | cut -c1-64) == ${IDSHA:?set IDSHA} ]] || { echo "FAIL LOUD: id list sha256"; exit 2; }
[[ $(md5sum < $G/MD5SUMS | cut -c1-32) == $GM5 ]] || { echo "FAIL LOUD: inputs_group MD5SUMS"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: md5sum -c"; exit 2; }
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
[ -e $SEL ] || python3 - <<P
import json
d=json.load(open("$U/topo_explore/ladder/sel_s128_c320.json")); d["topologies"]=list(range(30264,30276)); json.dump(d,open("$SEL","w"),indent=1)
P
if [ ! -e $IN ]; then
  mkdir -p $IN/models; for w in wf1_heavy wf1_moderate; do ln -s $G/$w $IN/$w; done
  cp -p $U/comb/in_comball/uncut_arm1_split.json $IN/
  for s in 1 2; do for st in uncut-comb-fl-gnn-eng uncut-comb-fl-mlp-same; do cp -p $U/comb/in_comball/models/$st-seed$s.{pt,contract.json} $IN/models/; done; done
fi
python3 - <<'P'
import hashlib, json
m=lambda p: hashlib.md5(open(p,"rb").read()).hexdigest()
IN="/share/nikola.lukic/uncut_groups/t12_s128_c320/in"; ck=json.load(open("/share/nikola.lukic/uncut_groups/comb/md5_comball.json"))
for k,st in (("ra_gnn_eng_combfl_nosplit","uncut-comb-fl-gnn-eng"),("ra_mlp_same_combfl_nosplit","uncut-comb-fl-mlp-same")):
    for s in (1,2):
        g=m(f"{IN}/models/{st}-seed{s}.pt"); assert g==ck[f"{k}|s{s}"], (k,s,g); print("ckpt",k,s,g)
assert m(IN+"/uncut_arm1_split.json")=="cb4f540d4010cba5b83194fae177cf98"
P
WL=$HOME/gnn-herosim-wt/s5_quick_${PIN:0:8}
[[ $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WL not clean at $PIN"; exit 2; }
if [ $KIND = cells ]; then ARMS=cd_ib0_inflightfull_k100,ra_gnn_eng_combfl_nosplit_k100,ra_mlp_same_combfl_nosplit_k100; NS=4; else ARMS=reactive; NS=1; fi
# non-exclusive packing (expert's ruling for this run; the exclusive-idle fail-loud is dropped): the 4 non-excluded nodes with the most idle CPUs and free memory;
# SMT is off, so a job of -c C gets C/2 physical cores and the gate needs PAR <= that.
EXCL=" os-cpu-slurm-21 os-cpu-slurm-13 os-cpu-slurm-14 os-cpu-slurm-15 os-cpu-slurm-16 os-cpu-slurm-10 os-cpu-slurm-17 "
declare -A CPUS PARN MEMG
pairs=$(sinfo -p CPU-amd -h -N -o "%N %C" | sort -u | awk '{split($2,a,"/"); print a[2], $1}' | sort -rn)
nodes=()
while read idle n; do
  [[ "$EXCL" == *" $n "* ]] && continue
  fm=$(scontrol show node $n | awk -F'[= ]+' '/RealMemory/{for(i=1;i<=NF;i++){if($i=="RealMemory")r=$(i+1); if($i=="AllocMem")a=$(i+1)}} END{print int((r-a)/1024)}')
  par=$(( idle / 2 )); [ $par -gt 60 ] && par=60; [ $(( par * 16 + 8 )) -gt $fm ] && par=$(( (fm - 8) / 16 ))
  [ $par -ge 4 ] || continue
  nodes+=($n); CPUS[$n]=$(( par * 2 )); PARN[$n]=$par; MEMG[$n]=$(( par * 16 + 8 ))
  [ ${#nodes[@]} -ge $NS ] && break
done <<< "$pairs"
echo "nodes: ${nodes[*]:-none}"; for n in "${nodes[@]}"; do echo "  $n -c ${CPUS[$n]} PAR ${PARN[$n]} mem ${MEMG[$n]}G"; done
[ ${#nodes[@]} -ge $NS ] || { echo "FAIL LOUD: fewer than $NS usable nodes (>= 4 physical cores and memory free); not submitting"; exit 2; }
[ -n "${OVER_CAP:-}" ] || [ $(( $(squeue -u nikola.lukic -h | wc -l) + NS )) -le 44 ] || { echo "FAIL LOUD: $NS more jobs would exceed 44"; exit 2; }; [ -z "${OVER_CAP:-}" ] || echo "NOTE: queue cap override (OVER_CAP set; the user said go over the limit)"
COMMON="IN=$IN SELECT=$SEL R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped"
for i in $(seq 0 $((NS-1))); do n=${nodes[$i]}
  j=$(env $COMMON PAR=${PARN[$n]} WT=$WL EXPECT_HEAD=${PIN:0:8} OUT=$OUT R1A_ARMS=$ARMS R1A_SEEDS=1,2 R1A_SHARD=$i/$NS \
      sbatch --parsable --export=ALL -p CPU-amd -w $n -c ${CPUS[$n]} --mem=${MEMG[$n]}G -t 08:00:00 -J s5-t12-$KIND-$i -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "$KIND shard $i/$NS node $n job $j"
done
