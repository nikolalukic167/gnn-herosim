#!/bin/bash
# Exchange-sensitivity study: GENERATE the job queue (does not submit; xs_driver.sh submits with a node throttle). 7 variants x 12 ids x g0-g3 x moderate/heavy.
#   STAGE=0 smoke : per variant, one bar cell (cd_ib0_inflightfull_k100 s0 @a459dd98) and one FL gnn cell (ra_gnn_eng_combfl_nosplit_k100 s1 @74ddf70e) on 19461 g0 heavy
#                   (variant workload + 19461's cfg), completion status only.  x0 first.
#   STAGE=1       : reactive admission cells (seed 0, @a459dd98), 96 per variant -> xs/<v>/cells_reactive
#   STAGE=2       : per fully admitted variant: bar cd_ib0_inflightfull_k100 s0 (96, @a459dd98) -> cells_bar; FL gnn + FL MLP _k100 seeds 1-2 (384, @74ddf70e, 3 shards) -> cells_learned
# usage: STAGE=0|1|2 xs_cells.sh "<12 ids>" <sha256 of the comma-joined list> [variants...]
set -euo pipefail
STAGE=${STAGE:?set STAGE=0|1|2}
IDS=$1; WANT=$2; shift 2; VARS=${*:-x0 x1 x0p1 x0p3 x3 tier0 azblob}
U=/share/nikola.lukic/uncut_groups; X=$U/xs; Q=$X/queue.txt; mkdir -p $X/logs
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
T=$(echo $IDS | tr ' ' ',')
WB=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98; WL=$HOME/gnn-herosim-wt/s6_p2_74ddf70e
[[ $(git -C $WB rev-parse HEAD) == a459dd98* && -z "$(git -C $WB status --porcelain)" ]] || { echo "FAIL LOUD: bar worktree"; exit 2; }
[[ $(git -C $WL rev-parse HEAD) == 74ddf70e* && -z "$(git -C $WL status --porcelain)" ]] || { echo "FAIL LOUD: learned worktree"; exit 2; }
echo $IDS > $X/ids.txt
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
mkdir -p $X/models_in
python3 - <<'P'
import hashlib, json, os, shutil
U = "/share/nikola.lukic/uncut_groups"; D = U + "/xs/models_in"
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
ck = json.load(open(U + "/comb/md5_comball.json"))
for kind, stem in (("ra_gnn_eng_combfl_nosplit", "uncut-comb-fl-gnn-eng"), ("ra_mlp_same_combfl_nosplit", "uncut-comb-fl-mlp-same")):
    for s in (1, 2):
        src = f"{U}/comb/in_comball/models/{stem}-seed{s}.pt"; dst = f"{D}/{stem}-seed{s}.pt"
        if m(src) != ck[f"{kind}|s{s}"]: raise SystemExit(f"FAIL LOUD: {src} md5")
        if not os.path.exists(dst): shutil.copy2(src, dst); shutil.copy2(src[:-3] + ".contract.json", dst[:-3] + ".contract.json")
        if m(dst) != ck[f"{kind}|s{s}"]: raise SystemExit(f"FAIL LOUD: staged {dst}")
print("models OK")
P
SEL=$X/xs_selected.json
python3 -c "import json;json.dump({'verdict':'DESIGN-READY','lineage':'uncut_groups_s0_v1','clients':160,'servers':24,'connection_probability':0.6,'topologies':[int(t) for t in '$T'.split(',')],'windows':['g0','g1','g2','g3']},open('$SEL','w'),indent=1)"
mkin() { # variant -> $X/in_<v>: wf1 links to the variant tree, split, models
  v=$1; IN=$X/in_$v; R=$X/gate_inputs_xs_$v/inputs_group
  [ -d $R ] || { echo "FAIL LOUD: variant root $R missing"; exit 2; }
  (cd $R && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: $v inputs_group md5sum -c"; exit 2; }
  if [ ! -e $IN ]; then
    mkdir -p $IN/models
    for w in wf1_heavy wf1_moderate; do ln -s $R/$w $IN/$w; done
    cp -p $U/conf2/in_conf2/uncut_arm1_split.json $IN/; cp -p $X/models_in/* $IN/models/
  fi
  for t in ${T//,/ }; do for tag in moderate heavy; do python3 - "$IN/wf1_$tag/cfg/cc40s$t.json" <<'P'
import json, sys
c = json.load(open(sys.argv[1])); assert (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"]) == (160, 24), sys.argv[1]
P
  done; done
}
line() { echo "$*" >> $Q; }
COMMON="SELECT=$SEL TOPOS=$T WINDOWS=g0,g1,g2,g3"
for v in $VARS; do
  case $STAGE in
  0) R=$X/gate_inputs_xs_$v/inputs_group; [ -d $R ] || { echo "FAIL LOUD: $R missing"; exit 2; }
     S=$X/in_smoke_$v
     if [ ! -e $S ]; then
       mkdir -p $S/models; for w in wf1_heavy wf1_moderate; do mkdir -p $S/$w; ln -s $U/gate_inputs_e2738e37/inputs_group/$w/cfg $S/$w/cfg; ln -s $R/$w/wl $S/$w/wl; done
       cp -p $U/conf2/in_conf2/uncut_arm1_split.json $S/; cp -p $X/models_in/uncut-comb-fl-gnn-eng-seed1.* $S/models/
     fi
     line "NAME=smoke-$v-bar WT=$WB HEAD=a459dd98 OUT=$X/smoke_$v/bar ARMS=cd_ib0_inflightfull_k100 SEEDS=0 SHARD=0/1 IN=$S SELECT=$U/hand_selected.json TOPOS=19461 WINDOWS=g0 RUNGS=heavy PAR=2 WALL=02:00:00 STAGE=0"
     line "NAME=smoke-$v-gnn WT=$WL HEAD=74ddf70e OUT=$X/smoke_$v/gnn ARMS=ra_gnn_eng_combfl_nosplit_k100 SEEDS=1 SHARD=0/1 IN=$S SELECT=$U/hand_selected.json TOPOS=19461 WINDOWS=g0 RUNGS=heavy PAR=2 WALL=02:00:00 STAGE=0" ;;
  1) mkin $v; [ ! -e $X/$v/cells_reactive ] || { echo "FAIL LOUD: $X/$v/cells_reactive exists"; exit 2; }
     line "NAME=react-$v WT=$WB HEAD=a459dd98 OUT=$X/$v/cells_reactive ARMS=reactive SEEDS=1,2 SHARD=0/1 IN=$X/in_$v $COMMON RUNGS=moderate,heavy PAR=60 WALL=02:00:00 STAGE=1" ;;
  2) mkin $v; for d in cells_bar cells_learned; do [ ! -e $X/$v/$d ] || { echo "FAIL LOUD: $X/$v/$d exists"; exit 2; }; done
     python3 -c "import json,sys;a=json.load(open('$X/$v/admission.json'));sys.exit(0 if len(a['admitted'])==12 else 1)" || { echo "NOT QUEUED: $v is not fully admitted (see $X/$v/admission.json); the expert decides"; continue; }
     line "NAME=bar-$v WT=$WB HEAD=a459dd98 OUT=$X/$v/cells_bar ARMS=cd_ib0_inflightfull_k100 SEEDS=1,2 SHARD=0/1 IN=$X/in_$v $COMMON RUNGS=moderate,heavy PAR=60 WALL=04:00:00 STAGE=2"
     for i in 0 1 2; do
       line "NAME=learned-$v-$i WT=$WL HEAD=74ddf70e OUT=$X/$v/cells_learned ARMS=ra_gnn_eng_combfl_nosplit_k100,ra_mlp_same_combfl_nosplit_k100 SEEDS=1,2 SHARD=$i/3 IN=$X/in_$v $COMMON RUNGS=moderate,heavy PAR=60 WALL=04:00:00 STAGE=2"
     done ;;
  esac
done
echo "queue lines now: $(wc -l < $Q)  (stage $STAGE, variants: $VARS); expected cells per variant: reactive 96; bar 96; learned 384 (gnn 192 + mlp 192)"
