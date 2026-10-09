#!/bin/bash
# Submit the production BUILD array in pieces under the 50-queued-jobs account limit (array tasks count; other sessions share it), then the check.
# Run on the login node (one ssh-free squeue per poll, every 2 min). The captures must be finished: with every sentinel present the pieces carry no
# dependency (aftercorr on a finished capture is a no-op), and each build task still fails loud if its topology lacks sentinels.
#   WT=<pin> ROOT=<root> TOPOS="<all topologies>" HELDOUT_TOPOS="..." BATCHES_TARGET=6180 [PIECE=20] [CAP=44] wf1_corpus_prod_submit_builds.sh
# Stop it with Ctrl-C / kill; nothing it has already submitted is touched (scancel the pieces if needed).
set -euo pipefail
WT=${WT:?}; ROOT=${ROOT:?}; TOPOS=${TOPOS:?}; HELDOUT_TOPOS=${HELDOUT_TOPOS:?}; BATCHES_TARGET=${BATCHES_TARGET:?}
PIECE=${PIECE:-20}; CAP=${CAP:-44}; TAGS=${TAGS:-"light moderate heavy"}
D="$WT/scripts_cosim/datalab"
N=$(echo $TOPOS | wc -w); ids=(); off=0
while [[ $off -lt $N ]]; do
  n=$(( N - off < PIECE ? N - off : PIECE ))
  while [[ $(( $(squeue -h -r -u "$USER" | wc -l) + n )) -gt $CAP ]]; do sleep 120; done
  id=$(sbatch --parsable --array=0-$((n-1)) --export=ALL,WT="$WT",ROOT="$ROOT",TOPOS="$TOPOS",TOPO_OFFSET=$off,TAGS="$TAGS",HELDOUT_TOPOS="$HELDOUT_TOPOS",BATCHES_TARGET=$BATCHES_TARGET "$D/wf1_corpus_prod_build.sbatch")
  echo "build piece offset=$off n=$n -> $id"; ids+=("$id"); off=$((off+n))
done
while [[ $(( $(squeue -h -r -u "$USER" | wc -l) + 1 )) -gt $CAP ]]; do sleep 120; done
dep=$(IFS=:; echo "${ids[*]}")
chk=$(sbatch --parsable --dependency=afterany:$dep --export=ALL,WT="$WT",ROOT="$ROOT",BATCHES_TARGET=$BATCHES_TARGET "$D/wf1_corpus_prod_check.sbatch")
echo "check -> $chk (afterany:$dep)"
