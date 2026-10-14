#!/bin/bash
# Submit the Arm 2 / Arm 2-K0 training chains: 4 runs per TAG, LINKS dependent 7-day jobs per run (afterany; each link refuses to start unless the
# previous one ended TIMEOUT, see uncut_arm2_train.sbatch). Usage: uncut_arm2_chain.sh <worktree> <sha> [tags: "arm2 arm2k0"] [links: 3]
set -euo pipefail
WT=${1:?worktree}; SHA=${2:?sha}; TAGS=${3:-"arm2 arm2k0"}; LINKS=${4:-3}
cd "$WT"
for TAG in $TAGS; do
  for T in 0 1 2 3; do
    prev=""
    for L in $(seq 1 "$LINKS"); do
      dep=(); exp="ALL,TAG=$TAG,T=$T,WT=$WT,EXPECT_HEAD=$SHA"
      [[ -n "$prev" ]] && { dep=(--dependency=afterany:"$prev"); exp="$exp,PREV=$prev"; }
      prev=$(sbatch --parsable ${dep[@]+"${dep[@]}"} --export="$exp" scripts_cosim/datalab/uncut_arm2_train.sbatch)
      echo "$TAG T=$T link $L job $prev"
    done
  done
done
