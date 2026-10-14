#!/bin/bash
# Submit the two K=2 GNN Arm 2 runs (T=0 seed 1, T=2 seed 2) on one L40S each, LINKS dependent 7-day links per run (afterany; each link refuses to start
# unless the previous one ended TIMEOUT, see uncut_arm2_gpu_train.sbatch). Usage: uncut_arm2_gpu_chain.sh <worktree> <sha> [links: 2]
set -euo pipefail
WT=${1:?worktree}; SHA=${2:?sha}; LINKS=${3:-2}
cd "$WT"
for T in 0 2; do
  prev=""
  for L in $(seq 1 "$LINKS"); do
    dep=(); exp="ALL,T=$T,WT=$WT,EXPECT_HEAD=$SHA"
    [[ -n "$prev" ]] && { dep=(--dependency=afterany:"$prev"); exp="$exp,PREV=$prev"; }
    prev=$(sbatch --parsable ${dep[@]+"${dep[@]}"} --export="$exp" scripts_cosim/datalab/uncut_arm2_gpu_train.sbatch)
    echo "gpu T=$T link $L job $prev"
  done
done
