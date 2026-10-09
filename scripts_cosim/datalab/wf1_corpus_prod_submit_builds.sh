#!/bin/bash
# Submit the production BUILD array in pieces under the 50-queued-jobs account limit (array tasks count; other sessions share it), then the check.
# Run on the login node, or elsewhere with REMOTE='ssh datalab' (one squeue per poll, every 2 min). The captures must be finished: with every sentinel present the pieces carry no
# dependency (aftercorr on a finished capture is a no-op), and each build task still fails loud if its topology lacks sentinels.
#   WT=<pin> ROOT=<root> TOPOS="<all topologies>" HELDOUT_TOPOS="..." BATCHES_TARGET=6180 [PIECE=20] [CAP=44] wf1_corpus_prod_submit_builds.sh
# Stop it with Ctrl-C / kill; nothing it has already submitted is touched (scancel the pieces if needed).
set -euo pipefail
WT=${WT:?}; ROOT=${ROOT:?}; TOPOS=${TOPOS:?}; HELDOUT_TOPOS=${HELDOUT_TOPOS:?}; BATCHES_TARGET=${BATCHES_TARGET:?}
REMOTE=${REMOTE:-}; PIECE=${PIECE:-20}; CAP=${CAP:-44}; TAGS=${TAGS:-"light moderate heavy"}
D="$WT/scripts_cosim/datalab"
N=$(echo $TOPOS | wc -w); ids=(); off=0; TOPO_ARR=($TOPOS); NCELL=$(( $(echo $TAGS | wc -w) * ${NW:-4} ))
SUBMIT_USER=${SUBMIT_USER:-$USER}
sub() { ${REMOTE:-bash} ${REMOTE:+bash -s}; }   # run the heredoc on the cluster (REMOTE='ssh datalab') or here
q() { sub <<<"squeue -h -r -u $SUBMIT_USER | wc -l"; }
while [[ $off -lt $N ]]; do
  n=$(( N - off < PIECE ? N - off : PIECE ))
  # a piece is submitted only when every topology in it has all its capture sentinels (a topology still capturing must not fail loud)
  while true; do
    missing=0
    for ((i=off; i<off+n; i++)); do
      have=$(sub <<<"ls $ROOT/snapshots/cc40s${TOPO_ARR[$i]}_*.done 2>/dev/null | wc -l")
      [[ "$have" -ge "$NCELL" ]] || missing=$((missing+1))
    done
    [[ $missing -eq 0 && $(( $(q) + n )) -le $CAP ]] && break
    echo "$(date +%H:%M) piece offset=$off: $missing topologies still capturing, or queue full"; sleep 120
  done
  id=$(sub <<<"sbatch --parsable --array=0-$((n-1)) --export=ALL,WT='$WT',ROOT='$ROOT',TOPOS='$TOPOS',TOPO_OFFSET=$off,TAGS='$TAGS',HELDOUT_TOPOS='$HELDOUT_TOPOS',BATCHES_TARGET=$BATCHES_TARGET '$D/wf1_corpus_prod_build.sbatch'")
  echo "build piece offset=$off n=$n -> $id"; ids+=("$id"); off=$((off+n))
done
while [[ $(( $(q) + 1 )) -gt $CAP ]]; do sleep 120; done
dep=$(IFS=:; echo "${ids[*]}")
# the check's volume target excludes cells the capture marked hung or failed: CHECK_PER_CELL x (cells with an ok sentinel)
CHECK_TARGET=$BATCHES_TARGET
if [[ -n "${CHECK_PER_CELL:-}" ]]; then
  ok=$(sub <<<"cat $ROOT/snapshots/*.done | grep -c '\"status\": \"ok\"'"); CHECK_TARGET=$(( CHECK_PER_CELL * ok ))
  echo "check target: $CHECK_PER_CELL x $ok ok cells = $CHECK_TARGET"
fi
chk=$(sub <<<"sbatch --parsable --dependency=afterany:$dep --export=ALL,WT='$WT',ROOT='$ROOT',BATCHES_TARGET=$CHECK_TARGET '$D/wf1_corpus_prod_check.sbatch'")
echo "check -> $chk (afterany:$dep)"
