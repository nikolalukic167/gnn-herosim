#!/bin/bash
# Server-count ladder cells: window g0, moderate and heavy, bar cd_ib0_inflightfull_k100 (s0) and reactive (s0) at a459dd98 (as the size pilot), FL gnn ra_gnn_eng_combfl_nosplit_k100 s1-2 at 74ddf70e;
# PAR 60, CELL_MEM from env (default 16G), R1A_SLATE declared_pruning_v1_uncapped. Dirs: topo_explore/ladder/cells_s<servers>_c<clients>/.
#   MODE=list   : write cells_list.txt (the cell names, md5 filed before any submission) and nothing else
#   MODE=smoke  : one 256-server, 320-client heavy FL gnn cell (seed 1, first topology of s256_c320) with CELL_MEM; submits it, prints the job id
#   MODE=queue  : write the driver queue (ladder/queue.txt), 3 lines per root (bar, reactive, FL gnn)
# usage: MODE=list|smoke|queue ladder_cells.sh "<18 ids>" <sha256>
set -euo pipefail
MODE=${MODE:?}; IDS=$1; WANT=$2
U=/share/nikola.lukic/uncut_groups; L=$U/topo_explore/ladder; ids=($IDS); CM=${CELL_MEM:-16G}
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256"; exit 2; }
roots=("96 160" "128 160" "192 160" "256 160" "128 320" "256 320")
WB=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98; WL=$HOME/gnn-herosim-wt/s6_p2_74ddf70e
if [ $MODE = list ]; then
  : > $L/cells_list.txt
  for i in 0 1 2 3 4 5; do read NS NC <<< "${roots[$i]}"
    for k in 0 1 2; do t=${ids[$((3*i+k))]}; for r in moderate heavy; do for a in cd_ib0_inflightfull_k100_s0 reactive_s0 ra_gnn_eng_combfl_nosplit_k100_s1 ra_gnn_eng_combfl_nosplit_k100_s2; do
      echo "cells_s${NS}_c${NC}/cc40s${t}__g0${r}__${a}" >> $L/cells_list.txt; done; done; done; done
  sort -o $L/cells_list.txt $L/cells_list.txt; echo "cells $(wc -l < $L/cells_list.txt) md5 $(md5sum < $L/cells_list.txt | cut -c1-32)"; exit 0
fi
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 - <<'P'
import hashlib, json, os, shutil
U = "/share/nikola.lukic/uncut_groups"; D = U + "/topo_explore/ladder/models_in"; os.makedirs(D, exist_ok=True)
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
ck = json.load(open(U + "/comb/md5_comball.json"))
for s in (1, 2):
    src = f"{U}/comb/in_comball/models/uncut-comb-fl-gnn-eng-seed{s}.pt"; dst = f"{D}/uncut-comb-fl-gnn-eng-seed{s}.pt"
    if m(src) != ck[f"ra_gnn_eng_combfl_nosplit|s{s}"]: raise SystemExit(f"FAIL LOUD: {src} md5")
    if not os.path.exists(dst): shutil.copy2(src, dst); shutil.copy2(src[:-3] + ".contract.json", dst[:-3] + ".contract.json")
    if m(dst) != ck[f"ra_gnn_eng_combfl_nosplit|s{s}"]: raise SystemExit("FAIL LOUD: staged model")
print("models OK (the conf2 / size-64 FL gnn s1-2, comb/md5_comball.json)")
P
mkin() { NS=$1; NC=$2; shift 2; T="$*"; R=$L/gate_inputs_s${NS}_c${NC}/inputs_group; IN=$L/in_s${NS}_c${NC}
  (cd $R && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: $R md5sum -c"; exit 2; }
  if [ ! -e $IN ]; then mkdir -p $IN/models; for w in wf1_heavy wf1_moderate; do ln -s $R/$w $IN/$w; done; cp -p $U/conf2/in_conf2/uncut_arm1_split.json $IN/; cp -p $L/models_in/* $IN/models/; fi
  for t in $T; do for tag in moderate heavy; do python3 - $IN/wf1_$tag/cfg/cc40s$t.json $NS $NC <<'P'
import json, sys
c = json.load(open(sys.argv[1])); got = (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"])
assert got == (int(sys.argv[3]), int(sys.argv[2])), (sys.argv[1], got)
P
  done; done
  python3 -c "import json;json.dump({'verdict':'DESIGN-READY','lineage':'uncut_groups_s0_v1','clients':$NC,'servers':$NS,'connection_probability':0.6,'topologies':[int(t) for t in '$T'.split()],'windows':['g0']},open('$L/sel_s${NS}_c${NC}.json','w'),indent=1)"
}
COMMON="CELL_MEM=$CM LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped"
if [ $MODE = smoke ]; then
  T=${ids[15]}; mkin 256 320 $T
  [[ $(git -C $WL rev-parse HEAD) == 74ddf70e* && -z "$(git -C $WL status --porcelain)" ]] || { echo "FAIL LOUD: learned worktree"; exit 2; }
  mkdir -p $L/smoke
  [ ! -e $L/smoke/cells ] || { echo "FAIL LOUD: smoke dir exists"; exit 2; }
  j=$(env $COMMON IN=$L/in_s256_c320 SELECT=$L/sel_s256_c320.json R1A_TOPOS=$T R1A_WINDOWS=g0 WF1_RUNGS=heavy PAR=1 R1A_SEEDS=1 WT=$WL EXPECT_HEAD=74ddf70e OUT=$L/smoke/cells R1A_ARMS=ra_gnn_eng_combfl_nosplit_k100 R1A_SHARD=0/1 \
      sbatch --parsable --export=ALL -p CPU-amd --exclude=os-cpu-slurm-5,os-cpu-slurm-10,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-15,os-cpu-slurm-16,os-cpu-slurm-17,os-cpu-slurm-21 --exclusive --nodes=1 --mem=0 -t 04:00:00 -J lad-smoke -o $L/smoke/lad-smoke-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "smoke job $j (topology $T, CELL_MEM=$CM, PAR=1)"; exit 0
fi
Q=$L/queue.txt; line() { echo "$*" >> $Q; }
for i in 0 1 2 3 4 5; do read NS NC <<< "${roots[$i]}"; T="${ids[$((3*i))]} ${ids[$((3*i+1))]} ${ids[$((3*i+2))]}"; mkin $NS $NC $T
  V="IN=$L/in_s${NS}_c${NC} SELECT=$L/sel_s${NS}_c${NC}.json TOPOS=${T// /,} WINDOWS=g0 RUNGS=moderate,heavy PAR=60 CELL_MEM=$CM OUT=$L/cells_s${NS}_c${NC}"
  line "NAME=bar-s${NS}-c${NC} WT=$WB HEAD=a459dd98 ARMS=cd_ib0_inflightfull_k100 SEEDS=0 SHARD=0/1 $V WALL=04:00:00"
  line "NAME=react-s${NS}-c${NC} WT=$WB HEAD=a459dd98 ARMS=reactive SEEDS=0 SHARD=0/1 $V WALL=04:00:00"
  line "NAME=fl-s${NS}-c${NC} WT=$WL HEAD=74ddf70e ARMS=ra_gnn_eng_combfl_nosplit_k100 SEEDS=1,2 SHARD=0/1 $V WALL=08:00:00"
done
echo "queue lines: $(wc -l < $Q)"
