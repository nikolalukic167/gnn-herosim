#!/bin/bash
# Own-state round retest launch, v2 (registered 6d6e1e54, A1 c810a270; expert's go 2026-10-23). Supersedes own24_retest.sh (md5 e7ddca43...): kinds with _nosplit, the DAgger split for the new arms, checkpoints from the
# best-val snapshot (own24/ckpt_snapshot, md5_dgu.json), non-exclusive packing by free CPUs, GNN arms first (newg) then twin arms (newt).
# usage: own24_retest_v2.sh ident|newg|newt|old <NS shards> [PAR cap, default 30]      pin 05f52abb (worktree s5_own24_05f52abb, clean)
#   ident: 1 job, bar + old FL gnn s1 + old twin s1 on fl3 cc40s30149 g0 heavy at the pin; compare with ident_cells_cmp2.py against fl3/cells_{bar,learned,twin}; a miss stops the study.
#   newg:  new gnn (ra_gnn_eng_combfl_dgu_nosplit) _k100 and _kr, seeds 1-6, 24 topologies x g0-g3 x moderate/heavy = 2,304 cells.   newt: the twin, 2,304 cells.   old: old FL gnn + old twin, seeds 1-2, _k100 = 768 cells.
set -euo pipefail
MODE=${1:?ident|newg|newt|old}; NS=${2:?shards}; PARCAP=${3:-30}; PIN=05f52abb
U=/share/nikola.lukic/uncut_groups; R=$U/own24; G=$U/gate_inputs_own24/inputs_group; SEL=$U/own24_selected.json; mkdir -p $R/logs
GNN_OLD=ra_gnn_eng_combfl_nosplit; GNN_NEW=ra_gnn_eng_combfl_dgu_nosplit; TWIN_NEW=ra_twin_eng_combfl_dgu_nosplit; TWIN_OLD=ra_twin_eng_combfl_nosplit
WANT=d4b1fb268c14af65a7c201ef27a97fbfb3a3033a964218cfbf443f060032ac37
T=$(seq -s, 20508 20531); [[ $(echo -n $T | sha256sum | cut -c1-64) == $WANT ]] || { echo "FAIL LOUD: id list sha256"; exit 2; }
[[ $(md5sum < $G/MD5SUMS | cut -c1-32) == 72a7db697771082a14916ad82e540507 ]] || { echo "FAIL LOUD: inputs_group MD5SUMS"; exit 2; }
(cd $G && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: inputs_group md5sum -c"; exit 2; }
[[ $(md5sum < $SEL | cut -c1-32) == b89d6bf460bba2acb1b17dce8f195d6e ]] || { echo "FAIL LOUD: selection file"; exit 2; }
WL=$HOME/gnn-herosim-wt/s5_own24_$PIN
[[ -d $WL && $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree $WL missing, dirty or not at $PIN"; exit 2; }
git -C $WL merge-base --is-ancestor 211241a7 $PIN || { echo "FAIL LOUD: $PIN does not descend from the retest pin 211241a7"; exit 2; }
CK=$R/ckpt_snapshot/md5_dgu.json; CL=$U/comb/md5_comball.json; TW=$U/fl3/md5_twin.json
DSPLIT_SHA=18e33008abb4bd0443e1bd79e02bb47f38998ac77e05b920bfe213afff2cd232
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
stage() { # stage <IN dir> <md5 json> <kind> <stem> <srcdir or -> <seeds...>   (srcdir "-": use the json's _provenance source)
  python3 - "$@" <<'P'
import hashlib, json, os, shutil, sys
IN, CKF, KIND, STEM, SRCD, *seeds = sys.argv[1:]
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
d = json.load(open(CKF)); prov = d.get("_provenance", {})
for s in seeds:
    key = f"{KIND}|s{s}"
    if key not in d: raise SystemExit(f"FAIL LOUD: {CKF} has no key {key}")
    src = prov[key]["source"] if SRCD == "-" else f"{SRCD}/{STEM}-seed{s}.pt"
    if m(src) != d[key]: raise SystemExit(f"FAIL LOUD: source {src} md5 {m(src)} != md5 file {d[key]}")
    dst = f"{IN}/models/{STEM}-seed{s}.pt"
    if not os.path.exists(dst):
        shutil.copy2(src, dst); shutil.copy2(src[:-3] + ".contract.json", dst[:-3] + ".contract.json")
    if m(dst) != d[key]: raise SystemExit(f"FAIL LOUD: staged {dst} differs from the md5 file")
    print(f"staged {key} {d[key]}")
P
}
mkin() { # mkin <IN dir> <groups dir> <arm1|dagger>
  mkdir -p $1/models; for w in wf1_heavy wf1_moderate; do [ -e $1/$w ] || ln -s $2/$w $1/$w; done
  if [ $3 = arm1 ]; then cp -pn $U/conf2/in_conf2/uncut_arm1_split.json $1/; [[ $(md5sum < $1/uncut_arm1_split.json | cut -c1-32) == cb4f540d4010cba5b83194fae177cf98 ]] || { echo "FAIL LOUD: arm1 split md5"; exit 2; }
  else cp -pn $WL/experiments/uncut_dagger_full_split.json $1/; [[ $(sha256sum < $1/uncut_dagger_full_split.json | cut -c1-64) == $DSPLIT_SHA ]] || { echo "FAIL LOUD: dagger split sha256"; exit 2; }; fi
}
EXCL=" os-cpu-slurm-21 "
declare -A CPUS PARN MEMG
nodes=()
pick() { # pick <count>: the freest nodes by idle CPUs (non-exclusive); PAR = idle/2 capped at PARCAP
  local pairs; pairs=$(sinfo -p CPU-amd -h -N -o "%N %C" | sort -u | awk '{split($2,a,"/"); print a[2], $1}' | sort -rn)
  while read idle n; do [[ "$EXCL" == *" $n "* ]] && continue
    par=$(( idle / 2 )); [ $par -gt $PARCAP ] && par=$PARCAP; [ $par -ge 3 ] || continue
    nodes+=($n); CPUS[$n]=$(( par * 2 )); PARN[$n]=$par; MEMG[$n]=$(( par * 3 + 16 ))
    [ ${#nodes[@]} -ge $1 ] && break
  done <<< "$pairs"
  echo "nodes: ${nodes[*]:-none}"; for n in "${nodes[@]}"; do echo "  $n -c ${CPUS[$n]} PAR ${PARN[$n]} mem ${MEMG[$n]}G"; done
  [ ${#nodes[@]} -ge $1 ] || { echo "FAIL LOUD: fewer than $1 usable nodes; not submitting"; exit 2; }; }
free() { [ $(( $(squeue -u nikola.lukic -h | wc -l) + $1 )) -le 44 ] || { echo "FAIL LOUD: $1 more jobs would exceed 44 in the queue"; exit 2; }; }
shard() { # shard <name> <IN> <OUT> <arms> <seeds> <topos> <select> <split>
  local nm=$1 IN=$2 OUT=$3 ARMS=$4 SEEDS=$5 TOPOS=$6 SELX=$7 SPL=$8
  for i in $(seq 0 $((NS-1))); do n=${nodes[$i]}
    j=$(env IN=$IN SELECT=$SELX R1A_TOPOS=$TOPOS R1A_WINDOWS=g0,g1,g2,g3 WF1_RUNGS=moderate,heavy PAR=${PARN[$n]} CELL_MEM=8G LIMIT=14400 R1A_SPLIT=$SPL R1A_SLATE=declared_pruning_v1_uncapped \
        WT=$WL EXPECT_HEAD=${PIN} OUT=$OUT R1A_ARMS=$ARMS R1A_SEEDS=$SEEDS R1A_SHARD=$i/$NS \
        sbatch --parsable --export=ALL -p CPU-amd -w $n -c ${CPUS[$n]} --mem=${MEMG[$n]}G -t 10:00:00 -J s5-own24-$nm-$i -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
    echo "$nm shard $i/$NS node $n job $j"; done; }
case $MODE in
ident)
  F=$U/gate_inputs_fl3/inputs_group; IN=$R/in_ident; OUT=$R/ident; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  mkin $IN $F arm1; stage $IN $CL $GNN_OLD uncut-comb-fl-gnn-eng $U/comb/in_comball/models 1; stage $IN $TW $TWIN_OLD uncut-comb-fl-twin $U/fl3/in_fl3_twin/models 1
  NS=1; pick 1; free 1
  n=${nodes[0]}
  j=$(env IN=$IN SELECT=$U/fl3_selected.json R1A_TOPOS=30149 R1A_WINDOWS=g0 WF1_RUNGS=heavy PAR=3 CELL_MEM=8G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped WT=$WL EXPECT_HEAD=${PIN} OUT=$OUT \
      R1A_ARMS=cd_ib0_inflightfull_k100,${GNN_OLD}_k100,${TWIN_OLD}_k100 R1A_SEEDS=1 sbatch --parsable --export=ALL -p CPU-amd -w $n -c 8 --mem=32G -t 02:00:00 -J s5-own24-ident -o $R/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "ident job $j node $n" ;;
newg|newt)
  [ -s "$CK" ] || { echo "FAIL LOUD: no checkpoint md5 file"; exit 2; }
  IN=$R/in_new; mkin $IN $G dagger; stage $IN $CK $GNN_NEW uncut-dagger-full-gnn - 1 2 3 4 5 6; stage $IN $CK $TWIN_NEW uncut-dagger-full-twin - 1 2 3 4 5 6
  if [ $MODE = newg ]; then A=${GNN_NEW}_k100,${GNN_NEW}_kr; OUT=$R/cells_new_gnn; else A=${TWIN_NEW}_k100,${TWIN_NEW}_kr; OUT=$R/cells_new_twin; fi
  [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  pick $NS; free $NS
  shard $MODE $IN $OUT $A 1,2,3,4,5,6 $T $SEL uncut_dagger_full_split.json
  echo "expected: $OUT 2,304 summaries (11,520 files); sealed: job states and file counts only" ;;
old)
  IN=$R/in_old; OUT=$R/cells_old; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
  mkin $IN $G arm1; stage $IN $CL $GNN_OLD uncut-comb-fl-gnn-eng $U/comb/in_comball/models 1 2; stage $IN $TW $TWIN_OLD uncut-comb-fl-twin $U/fl3/in_fl3_twin/models 1 2
  pick $NS; free $NS
  shard old $IN $OUT ${GNN_OLD}_k100,${TWIN_OLD}_k100 1,2 $T $SEL uncut_arm1_split.json
  echo "expected: $OUT 768 summaries (3,840 files); sealed" ;;
*) echo "FAIL LOUD: mode"; exit 2;;
esac
