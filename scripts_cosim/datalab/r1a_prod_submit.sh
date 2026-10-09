#!/bin/bash
# r1_attribution_v1: submit the production chain on the datalab login node, in two stages so the shared 44-job limit is never exceeded.
#   stage 1 (now):   cache (afterok the corpus check) -> training array indices 0-20 (the 21 slowest tasks, afterok cache) -> a 1-CPU waiter
#   stage 2 (waiter): training indices 21-41 and the selection job (afterok both halves), submitted when queued jobs + 21 <= 44
# Fails loud when stage 1 would not fit. Usage:
#   CHECK_JOB=853822 WT=<pinned worktree> EXPECT_HEAD=<sha> bash r1a_prod_submit.sh [dry]
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
THROTTLE=${THROTTLE:-12}; LIMIT=${LIMIT:-44}
R=/share/nikola.lukic/r1a_prod; STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
INPUTS=${INPUTS:-/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/r1a_gate_inputs}
cd "$WT"
HEAD=$(git rev-parse HEAD)
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink into ~/gnn-herosim/models"; exit 1; }
queued=$(squeue -u "$USER" -r -h | wc -l)
need=$((1 + 21 + 1))   # cache + 21 array elements + the waiter
(( queued + need <= LIMIT )) || { echo "FAIL LOUD: $queued jobs queued + $need > $LIMIT; not submitting a partial chain"; exit 1; }
echo "worktree $HEAD; queued $queued; stage 1 adds $need"
DEP=""; [[ -n "${CHECK_JOB:-}" ]] && DEP="--dependency=afterok:$CHECK_JOB"
[[ "${1:-}" == dry ]] && { echo "dry: would submit under $R/$STAMP"; exit 0; }
mkdir -p "$R/$STAMP"; echo 150G > "$R/$STAMP/mem2"
CACHE_JOB=$(sbatch --parsable $DEP --export=ALL,WT="$WT",OUT="$R/$STAMP/cache_run" scripts_cosim/datalab/r1a_prod_cache.sbatch)
CHUNK1=$(sbatch --parsable --dependency=afterok:$CACHE_JOB --array=0-20%$THROTTLE --export=ALL,WT="$WT" scripts_cosim/datalab/r1a_prod_train.sbatch)
WAIT=$(sbatch --parsable --export=ALL,WT="$WT",CACHE_JOB=$CACHE_JOB,CHUNK1_JOB=$CHUNK1,STATE="$R/$STAMP",INPUTS="$INPUTS",SEL_OUT="$R/$STAMP/selection.json",THROTTLE=$THROTTLE,LIMIT=$LIMIT scripts_cosim/datalab/r1a_prod_chunk2.sbatch)
for j in "$CACHE_JOB" "$CHUNK1" "$WAIT"; do [[ "$j" =~ ^[0-9]+$ ]] || { echo "FAIL LOUD: a submission was refused (got '$j'); cancel the others: scancel $CACHE_JOB $CHUNK1 $WAIT"; exit 1; }; done
echo "cache $CACHE_JOB -> train indices 0-20 $CHUNK1 (throttle $THROTTLE) ; chunk-2 waiter $WAIT (submits 21-41 + selection) ; outputs $R/$STAMP"
