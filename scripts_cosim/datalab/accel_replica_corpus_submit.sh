#!/bin/bash
# accel_replica_v1 corpus (docs/lineages/accel_replica_v1.md; a clone of scale160_corpus_submit.sh on the rp/accel-calib code): the full submit, in two phases. Submit only on the coordinator's GO, after S5's
# go/no-go passed and the rung multipliers are signed into the node. Run on the login node (or elsewhere with REMOTE='ssh datalab').
#
#   phase capture: inputs for the whole POOL (160c x 24s p0.6 cfgs minted per seed, WF1 windows remapped over the clients), then the
#                  capture array in chunks under the 50-queued-jobs account limit, each chunk afterok on the inputs job.
#   phase build:   once every POOL topology has its sentinels, pick the split in seed order from the feasible topologies (no "failed"
#                  sentinel: a seed whose source reaches no platform for a task type fails its captures, StarvedForeverError; a hung
#                  cell is counted, not disqualifying): the first N_TRAIN are train, the next N_HELDOUT held-out. Writes
#                  $ROOT/split.json, then hands over to wf1_corpus_prod_submit_builds.sh (build pieces, then the check).
#
#   PHASE=capture WT=<pin> ROOT=<new root> M_MODERATE=<m> M_HEAVY=<m> scale160_corpus_submit.sh
#   PHASE=build   WT=<pin> ROOT=<same root> scale160_corpus_submit.sh
# Resume phase capture after a partial submit: INPUTS_DONE=1 (the inputs job finished; chunks carry no dependency) and
# START_OFFSET=<first pool index not yet submitted>. Run the driver itself as a 1-CPU sbatch job (scale160_driver.sbatch), never under
# nohup on the login node (datalab-pitfalls 16: the login node reaps background watchers).
# POOL defaults to 16201-16270 (corpus range 16201-16300, confirmed free by S4 and S6; S4 proposes the accel gate at 16301-16320; 9101, 9901-9912
# and 16001-16170 are never used here). N_TRAIN=50 N_HELDOUT=8 BATCHES_TARGET=2500 CAP=44 by default.
set -euo pipefail
PHASE=${PHASE:?FAIL LOUD: set PHASE=capture or PHASE=build}
# accel_replica_v1: the base cfg carries preinit.replica_placement_rule=fastest_compatible (S7's calibration cell cfg); every minted cfg
# inherits it (scale_probe_cfg.py copies the base and changes only counts, p and seed). The rule is read from the cfg, not the environment.
BASECFG=${BASECFG:-/share/nikola.lukic/accel_s7/calib/heavy_root/cfg_src/cc40s9905.json}
WT=${WT:?}; ROOT=${ROOT:?}
REMOTE=${REMOTE:-}; CAP=${CAP:-44}; TAGS="moderate heavy"; NW=4
POOL=${POOL:-$(seq -s ' ' 16201 16270)}
N_TRAIN=${N_TRAIN:-50}; N_HELDOUT=${N_HELDOUT:-8}; BATCHES_TARGET=${BATCHES_TARGET:-2500}
D="$WT/scripts_cosim/datalab"
SUBMIT_USER=${SUBMIT_USER:-$USER}
sub() { ${REMOTE:-bash} ${REMOTE:+bash -s}; }
q() { sub <<<"squeue -h -r -u $SUBMIT_USER | wc -l"; }
for t in $POOL; do
  [[ $t -ge 16201 && $t -le 16300 ]] || { echo "FAIL LOUD: seed $t is outside the accel_replica_v1 corpus range 16201-16300"; exit 1; }
done

if [[ $PHASE == capture ]]; then
  M_MODERATE=${M_MODERATE:?FAIL LOUD: set M_MODERATE (signed final multiplier, factor = 1/m)}
  M_HEAVY=${M_HEAVY:?FAIL LOUD: set M_HEAVY (signed final multiplier, factor = 1/m)}
  if [[ "${INPUTS_DONE:-0}" == 1 ]]; then
    [[ -d "$ROOT/inputs/wf1_moderate" && -d "$ROOT/inputs/wf1_heavy" ]] || { echo "FAIL LOUD: INPUTS_DONE=1 but $ROOT/inputs is incomplete"; exit 1; }
    dep=""
  else
    inp=$(sub <<<"export WT='$WT' ROOT='$ROOT' TOPOS='$POOL' SCALE=160:24:0.6 RUNGS='moderate=$M_MODERATE,heavy=$M_HEAVY' \
      SPLIT_SOURCE='accel_replica_v1' BASECFG='$BASECFG'; sbatch --parsable --export=ALL '$D/wf1_corpus_prod_inputs.sbatch'")
    echo "inputs -> $inp"; dep="--dependency=afterok:$inp"
  fi
  N=$(echo $POOL | wc -w); off=${START_OFFSET:-0}
  while [[ $off -lt $N ]]; do
    room=$(( CAP - $(q) ))
    if [[ $room -lt 1 ]]; then echo "$(date +%H:%M) queue full"; sleep 120; continue; fi
    n=$(( N - off < room ? N - off : room ))
    id=$(sub <<<"export WT='$WT' ROOT='$ROOT' TOPOS='$POOL' TAGS='$TAGS' TOPO_OFFSET=$off; \
      sbatch --parsable $dep --array=0-$((n-1)) --export=ALL '$D/wf1_corpus_prod_capture.sbatch'")
    echo "capture chunk offset=$off n=$n -> $id"; off=$((off+n))
  done
  exit 0
fi

[[ $PHASE == build ]] || { echo "FAIL LOUD: PHASE=$PHASE"; exit 1; }
NCELL=$(( $(echo $TAGS | wc -w) * NW ))
feasible=(); infeasible=(); pending=()
for t in $POOL; do
  have=$(sub <<<"ls $ROOT/snapshots/cc40s${t}_*.done 2>/dev/null | wc -l")
  if [[ $have -lt $NCELL ]]; then pending+=("$t"); continue; fi
  bad=$(sub <<<"grep -l '\"status\": \"failed\"' $ROOT/snapshots/cc40s${t}_*.done 2>/dev/null | wc -l")
  [[ $bad -eq 0 ]] && feasible+=("$t") || infeasible+=("$t")
done
[[ ${#pending[@]} -eq 0 ]] || { echo "FAIL LOUD: captures not finished for ${pending[*]}"; exit 1; }
need=$(( N_TRAIN + N_HELDOUT ))
[[ ${#feasible[@]} -ge $need ]] || { echo "FAIL LOUD: ${#feasible[@]} feasible topologies, need $need (infeasible: ${infeasible[*]:-none})"; exit 1; }
TRAIN="${feasible[*]:0:$N_TRAIN}"; HELD="${feasible[*]:$N_TRAIN:$N_HELDOUT}"
sub <<<"python3 - '$ROOT/split.json' '$TRAIN' '$HELD' '${infeasible[*]:-}' <<'PY'
import json, sys
out, train, held, bad = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4].split()
json.dump({'source': 'accel_replica_v1: first feasible seeds from 16201 in order (no failed capture sentinel), train then held-out',
           'train': train, 'heldout': held, 'infeasible': bad}, open(out, 'w'), indent=1)
PY"
echo "split: ${#feasible[@]} feasible, infeasible: ${infeasible[*]:-none}; train $N_TRAIN, held-out: $HELD"
PER_CELL=$(( (BATCHES_TARGET + need * NCELL - 1) / (need * NCELL) ))
WT="$WT" ROOT="$ROOT" TOPOS="$TRAIN $HELD" HELDOUT_TOPOS="$HELD" BATCHES_TARGET="$BATCHES_TARGET" TAGS="$TAGS" NW=$NW \
  CHECK_PER_CELL=$PER_CELL REMOTE="$REMOTE" CAP=$CAP bash "$(dirname "$0")/wf1_corpus_prod_submit_builds.sh"
