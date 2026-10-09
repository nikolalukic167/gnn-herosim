#!/bin/bash
# r1_attribution_v1: submit the production chain -- cache -> training array (afterok cache) -> selection (afterok array). Run on the datalab login node.
#   CHECK_JOB=853822 WT=<pinned worktree> bash r1a_prod_submit.sh [dry]     (CHECK_JOB: the corpus check the cache must wait for; unset = no dependency)
# THROTTLE caps concurrent array tasks (3 runs each); the shared limit is 44 jobs.
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}; THROTTLE=${THROTTLE:-12}
R=/share/nikola.lukic/r1a_prod; STAMP=$(date +%Y%m%d_%H%M%S)
INPUTS=${INPUTS:-/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/r1a_gate_inputs}
cd "$WT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
echo "worktree $(git rev-parse HEAD)"
DEP=""; [[ -n "${CHECK_JOB:-}" ]] && DEP="--dependency=afterok:$CHECK_JOB"
CMD_CACHE="sbatch --parsable $DEP --export=ALL,WT=$WT,OUT=$R/$STAMP scripts_cosim/datalab/r1a_prod_cache.sbatch"
echo "$CMD_CACHE"; [[ "${1:-}" == dry ]] && exit 0
mkdir -p "$R"
CACHE_JOB=$($CMD_CACHE)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$CACHE_JOB --array=0-41%$THROTTLE --export=ALL,WT=$WT scripts_cosim/datalab/r1a_prod_train.sbatch)
SEL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB --export=ALL,WT=$WT,INPUTS=$INPUTS,OUT=$R/$STAMP/selection.json scripts_cosim/datalab/r1a_prod_select.sbatch)
echo "cache $CACHE_JOB -> train array $TRAIN_JOB (throttle $THROTTLE) -> select $SEL_JOB; outputs $R/$STAMP"
