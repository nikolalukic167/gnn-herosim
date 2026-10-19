#!/bin/bash
# FL gnn seeds 3-6 served at 64 servers (descriptive, no label, no registration): the four -final.pt of conf2/md5_conf2.json (no retraining),
# _k100 and _kr, topologies 30129-30140 x g0-g3 x moderate/heavy at the P2 pin 74ddf70e, same recipe and node settings as size64/cells_learned.
# 768 cells -> size64/cells_learned_s36 (cell list: cells_learned_s36_list.txt). Idle exclusive nodes by sinfo load, never os-cpu-slurm-13..19/21/22 (conf2).
# usage: size64_s36_cells.sh <sha256 of the comma-joined id list> <MD5SUMS md5 of inputs_group> <md5 of the cell list>   [NSHARD=4..6]
set -euo pipefail
WANT=$1; GM5=$2; LM5=$3
U=/share/nikola.lukic/uncut_groups; R=$U/size64; P=$U/size64_s36; G=$U/gate_inputs_size64b/inputs_group; IN=$R/in_size64_s36; OUT=$R/cells_learned_s36
mkdir -p $R/logs
[ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
T=$(seq -s, 30129 30140)
[[ $(echo -n $T | sha256sum | cut -c1-64) == $WANT ]] || { echo "FAIL LOUD: id list sha256"; exit 2; }
[[ $(md5sum < $G/MD5SUMS | cut -c1-32) == $GM5 ]] || { echo "FAIL LOUD: inputs_group MD5SUMS"; exit 2; }
[[ $(md5sum < $P/cells_learned_s36_list.txt | cut -c1-32) == $LM5 ]] || { echo "FAIL LOUD: cell list md5"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: inputs_group md5sum -c"; exit 2; }
if [ ! -e $IN ]; then
  mkdir -p $IN/models
  for w in wf1_heavy wf1_moderate; do ln -s $G/$w $IN/$w; done
  cp -p $R/in_size64/uncut_arm1_split.json $IN/
  for s in 3 4 5 6; do cp -p $U/conf2/in_conf2/models/uncut-comb-fl-gnn-eng-seed$s.pt $U/conf2/in_conf2/models/uncut-comb-fl-gnn-eng-seed$s.contract.json $IN/models/; done
fi
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 - <<'P'
import hashlib, json
U = "/share/nikola.lukic/uncut_groups"; IN = U + "/size64/in_size64_s36"
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
ck = json.load(open(U + "/conf2/md5_conf2.json"))
for s in (3, 4, 5, 6):
    got = m(f"{IN}/models/uncut-comb-fl-gnn-eng-seed{s}.pt"); want = ck[f"ra_gnn_eng_combfl_nosplit|s{s}"]
    if got != want: raise SystemExit(f"FAIL LOUD: staged s{s} md5 {got} != {want}")
    print(f"staged s{s} {got}")
if m(IN + "/uncut_arm1_split.json") != "cb4f540d4010cba5b83194fae177cf98": raise SystemExit("FAIL LOUD: split md5")
for tag in ("moderate", "heavy"):
    for t in range(30129, 30141):
        c = json.load(open(f"{IN}/wf1_{tag}/cfg/cc40s{t}.json"))
        if (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"]) != (160, 64): raise SystemExit(f"FAIL LOUD: {tag} {t} is not 160 x 64")
print("inputs OK")
P
WL=$HOME/gnn-herosim-wt/s6_p2_74ddf70e
[[ $(git -C $WL rev-parse HEAD) == 74ddf70e* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: learned worktree"; exit 2; }
NS=${NSHARD:-4}
nodes=()
for n in $(sinfo -p CPU-amd -h -N -t idle -o "%O %N" | sort -n | awk '{print $2}' | awk '!s[$0]++' | grep -v -x -e os-cpu-slurm-21 -e os-cpu-slurm-13 -e os-cpu-slurm-14 -e os-cpu-slurm-15 -e os-cpu-slurm-16 -e os-cpu-slurm-17 -e os-cpu-slurm-18 -e os-cpu-slurm-19 -e os-cpu-slurm-22); do
  [ "$(squeue -w $n -h | wc -l)" = 0 ] || continue
  nodes+=($n); [ ${#nodes[@]} -ge $NS ] && break
done
echo "nodes: ${nodes[*]:-none}"
[ ${#nodes[@]} -ge 4 ] || { echo "FAIL LOUD: fewer than 4 idle nodes; not submitting"; exit 2; }
[ $(squeue -u nikola.lukic -h | wc -l) -le 36 ] || { echo "FAIL LOUD: queue too full"; exit 2; }
NS=${#nodes[@]}
COMMON="IN=$IN SELECT=$U/size64b_selected.json R1A_TOPOS=$T R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=60 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_SEEDS=3,4,5,6"
for i in $(seq 0 $((NS-1))); do n=${nodes[$i]}
  j=$(env $COMMON WT=$WL EXPECT_HEAD=74ddf70e OUT=$OUT R1A_ARMS=ra_gnn_eng_combfl_nosplit_k100,ra_gnn_eng_combfl_nosplit_kr R1A_SHARD=$i/$NS sbatch --parsable --export=ALL -p CPU-amd -w $n --exclusive --nodes=1 --mem=0 -t 08:00:00 -J s5-s64s36-$i -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "s36 shard $i/$NS node $n job $j"
done
