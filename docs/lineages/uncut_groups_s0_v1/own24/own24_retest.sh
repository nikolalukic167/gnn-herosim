#!/bin/bash
# Own-state round retest launch (registered 6d6e1e54, A1 c810a270). NOT SUBMITTED until: the 12 new checkpoints are trained and their md5 file is filed, parity says "PARITY OK",
# the w-only retest pin exists (a child touching only the w file, adding the kinds ra_gnn_eng_combfl_dgu / ra_twin_eng_combfl_dgu), the frozen reader is merged, and the expert says go.
# usage: own24_retest.sh ident|new|old <RETEST_PIN full sha> <NEW_CKPT_MD5_FILE>
#   ident: one job, 3 cells at the retest pin: bar, old FL gnn s1, old twin s1 on fl3 cc40s30149 g0 heavy; compare with ident_cells_cmp2.py against fl3/cells_{bar,learned,twin} (a miss stops the study).
#   new:   4,608 cells = new gnn + new twin, seeds 1-6, _k100 and _kr (4 shards, 4 nodes).   old: 768 cells = old FL gnn s1-2 + old twin s1-2, _k100 only (4 shards).
#   env: STEM_NEW_GNN / STEM_NEW_TWIN = checkpoint file stems the pinned stager expects (set from the pin; no default, fail loud), NODES_MAX=4.
set -euo pipefail
MODE=${1:?ident|new|old}; PIN=${2:?retest pin}; CK=${3:-}
U=/share/nikola.lukic/uncut_groups; R=$U/own24; G=$U/gate_inputs_own24/inputs_group; SEL=$U/own24_selected.json; mkdir -p $R/logs
GNN_OLD=ra_gnn_eng_combfl_nosplit; GNN_NEW=ra_gnn_eng_combfl_dgu; TWIN_NEW=ra_twin_eng_combfl_dgu; TWIN_OLD=ra_twin_eng_combfl_nosplit
WANT=d4b1fb268c14af65a7c201ef27a97fbfb3a3033a964218cfbf443f060032ac37
T=$(seq -s, 20508 20531); [[ $(echo -n $T | sha256sum | cut -c1-64) == $WANT ]] || { echo "FAIL LOUD: id list sha256"; exit 2; }
[[ $(md5sum < $G/MD5SUMS | cut -c1-32) == 72a7db697771082a14916ad82e540507 ]] || { echo "FAIL LOUD: inputs_group MD5SUMS"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: inputs_group md5sum -c"; exit 2; }
[[ $(md5sum < $SEL | cut -c1-32) == b89d6bf460bba2acb1b17dce8f195d6e ]] || { echo "FAIL LOUD: selection file"; exit 2; }
WL=$HOME/gnn-herosim-wt/s5_own24_${PIN:0:8}
[[ -d $WL && $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WL missing, dirty or not at $PIN (create it before launching)"; exit 2; }
git -C $WL merge-base --is-ancestor c810a270 $PIN 2>/dev/null || echo "WARN: $PIN is not a descendant of c810a270 in this clone's history; check"
CL=$U/comb/md5_comball.json; TW=$U/fl3/md5_twin.json
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
stage() { # stage <IN dir> <md5 file> <kind> <stem> <seeds...>
  python3 - "$@" <<'P'
import hashlib, json, os, shutil, sys
IN, CKF, KIND, STEM, *seeds = sys.argv[1:]
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
d = json.load(open(CKF)); prov = d.get("_provenance", {})
for s in seeds:
    key = f"{KIND}|s{s}"
    if key not in d: raise SystemExit(f"FAIL LOUD: {CKF} has no key {key}")
    src = prov.get(key, {}).get("source") or f"/home/nikola.lukic/gnn-herosim/models/{STEM}-seed{s}-final.pt"
    if m(src) != d[key]: raise SystemExit(f"FAIL LOUD: source {src} md5 {m(src)} != md5 file {d[key]}")
    dst = f"{IN}/models/{STEM}-seed{s}.pt"
    if not os.path.exists(dst):
        shutil.copy2(src, dst); shutil.copy2(src[:-3] + ".contract.json", dst[:-3] + ".contract.json")
    if m(dst) != d[key]: raise SystemExit(f"FAIL LOUD: staged {dst} differs from the md5 file")
    print(f"staged {key} {d[key]}")
P
}
mkin() { # mkin <IN dir> <groups dir>
  [ -e $1 ] && { echo "FAIL LOUD: $1 exists"; exit 2; }
  mkdir -p $1/models; for w in wf1_heavy wf1_moderate; do ln -s $2/$w $1/$w; done
  cp -p $U/conf2/in_conf2/uncut_arm1_split.json $1/
  [[ $(md5sum < $1/uncut_arm1_split.json | cut -c1-32) == cb4f540d4010cba5b83194fae177cf98 ]] || { echo "FAIL LOUD: split md5"; exit 2; }
}
nodes=()
pick() { for n in $(sinfo -p CPU-amd -h -N -t idle -o "%O %N" | sort -n | awk '{print $2}' | awk '!s[$0]++' | grep -v -x -e os-cpu-slurm-21 -e os-cpu-slurm-13 -e os-cpu-slurm-14 -e os-cpu-slurm-15 -e os-cpu-slurm-16 -e os-cpu-slurm-10 -e os-cpu-slurm-17); do
  [ "$(squeue -w $n -h | wc -l)" = 0 ] || continue
  nodes+=($n); [ ${#nodes[@]} -ge ${NODES_MAX:-4} ] && break
done; echo "nodes: ${nodes[*]:-none}"; [ ${#nodes[@]} -ge ${1:-4} ] || { echo "FAIL LOUD: fewer than ${1:-4} idle nodes; not submitting"; exit 2; }; }
free() { [ $(( $(squeue -u nikola.lukic -h | wc -l) + $1 )) -le 44 ] || { echo "FAIL LOUD: $1 more jobs would exceed 44 in the queue"; exit 2; }; }
COMMON_T="R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=60 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped"
shard() { # shard <name> <IN> <OUT> <arms> <seeds> <topos> <select>
  local nm=$1 IN=$2 OUT=$3 ARMS=$4 SEEDS=$5 TOPOS=$6 SELX=$7
  for i in 0 1 2 3; do n=${nodes[$((i % ${#nodes[@]}))]}
    j=$(env IN=$IN SELECT=$SELX R1A_TOPOS=$TOPOS $COMMON_T WT=$WL EXPECT_HEAD=${PIN:0:8} OUT=$OUT R1A_ARMS=$ARMS R1A_SEEDS=$SEEDS R1A_SHARD=$i/4 \
        sbatch --parsable --export=ALL -p CPU-amd -w $n --exclusive --nodes=1 --mem=0 -t 08:00:00 -J s5-own24-$nm-$i -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
    echo "$nm shard $i node $n job $j"; done; }
case $MODE in
ident)
  F=$U/gate_inputs_fl3/inputs_group; IN=$R/in_ident; OUT=$R/ident; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  mkin $IN $F; stage $IN $CL $GNN_OLD uncut-comb-fl-gnn-eng 1; stage $IN $TW $TWIN_OLD uncut-comb-fl-twin 1
  pick 1; free 1
  j=$(env IN=$IN SELECT=$U/fl3_selected.json R1A_TOPOS=30149 R1A_WINDOWS=g0 WF1_RUNGS=heavy PAR=3 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped WT=$WL EXPECT_HEAD=${PIN:0:8} OUT=$OUT \
      R1A_ARMS=cd_ib0_inflightfull_k100,${GNN_OLD}_k100,${TWIN_OLD}_k100 R1A_SEEDS=1 sbatch --parsable --export=ALL -p CPU-amd -w ${nodes[0]} --exclusive --nodes=1 --mem=0 -t 02:00:00 -J s5-own24-ident -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "ident job $j node ${nodes[0]}; then: for k in cd_ib0_inflightfull_k100_s0:cells_bar ra_gnn_eng_combfl_nosplit_k100_s1:cells_learned ra_twin_eng_combfl_nosplit_k100_s1:cells_twin: ident_cells_cmp2.py <fl3 cell> <$OUT cell>" ;;
new)
  [ -s "$CK" ] || { echo "FAIL LOUD: no new-checkpoint md5 file"; exit 2; }
  : "${STEM_NEW_GNN:?FAIL LOUD: set STEM_NEW_GNN}" "${STEM_NEW_TWIN:?FAIL LOUD: set STEM_NEW_TWIN}"
  IN=$R/in_new; OUT=$R/cells_new; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  mkin $IN $G; stage $IN $CK $GNN_NEW $STEM_NEW_GNN 1 2 3 4 5 6; stage $IN $CK $TWIN_NEW $STEM_NEW_TWIN 1 2 3 4 5 6
  pick 4; free 4
  shard new $IN $OUT ${GNN_NEW}_k100,${GNN_NEW}_kr,${TWIN_NEW}_k100,${TWIN_NEW}_kr 1,2,3,4,5,6 $T $SEL
  echo "expected: cells_new 4,608 summaries (23,040 files); sealed: job states and file counts only" ;;
old)
  IN=$R/in_old; OUT=$R/cells_old; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  mkin $IN $G; stage $IN $CL $GNN_OLD uncut-comb-fl-gnn-eng 1 2; stage $IN $TW $TWIN_OLD uncut-comb-fl-twin 1 2
  pick 4; free 4
  shard old $IN $OUT ${GNN_OLD}_k100,${TWIN_OLD}_k100 1,2 $T $SEL
  echo "expected: cells_old 768 summaries (3,840 files); sealed" ;;
*) echo "FAIL LOUD: mode"; exit 2;;
esac
