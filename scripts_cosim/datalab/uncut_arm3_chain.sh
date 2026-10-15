#!/bin/bash
# Submit resume-capable chains for the expected move-regret lever's wave-1 runs (the safety net; used only for a run that was lost or would pass the time limit).
# One chain per run in TARGETS (array indices of uncut_arm3_train.sbatch: 0 = l0p5 seed 1, 1 = l0p5 seed 2, 2 = l2p0 seed 1, 3 = l2p0 seed 2), LINKS dependent 14 h jobs per run (afterany; each link
# refuses to start unless the previous one ended TIMEOUT, see uncut_arm3_train.sbatch). Usage: uncut_arm3_chain.sh <worktree> <sha> <table sha256> [targets: "0 1 2 3"] [links: 2] [extra sbatch args]
set -euo pipefail
WT=${1:?worktree}; SHA=${2:?sha}; TSHA=${3:?table sha256}; TARGETS=${4:-"0 1 2 3"}; LINKS=${5:-2}; EXTRA=${6:---exclude=os-cpu-slurm-6}
cd "$WT"
for I in $TARGETS; do
  prev=""
  for L in $(seq 1 "$LINKS"); do
    dep=(); exp="ALL,RESUME=1,WT=$WT,EXPECT_HEAD=$SHA,TABLE_SHA=$TSHA"
    [[ -n "$prev" ]] && { dep=(--dependency=afterany:"$prev"); exp="$exp,PREV=$prev"; }
    prev=$(sbatch --parsable --array="$I" $EXTRA ${dep[@]+"${dep[@]}"} --export="$exp" scripts_cosim/datalab/uncut_arm3_train.sbatch)
    echo "target $I link $L job $prev"
  done
done
