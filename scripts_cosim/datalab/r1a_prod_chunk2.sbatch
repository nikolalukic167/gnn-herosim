#!/bin/bash
# r1_attribution_v1 production chain, stage 2: a 1-CPU job that (a) submits the second half of the training array (indices 21-41, the engineered trio
# first, see scripts_cosim/r1a_task_order.py) and the all-arms selection as soon as the account has room under the shared 44-job limit (pending
# array elements count; it never submits a partial array: sbatch takes the whole 21-element array or refuses it, and any refusal that is not a limit
# refusal fails this job loudly), then (b) stays alive and runs the per-arm selection (r1a_select_checkpoints.py --arm X --if-complete) for each arm the
# moment all 18 of its runs have a validation record, staging that arm's checkpoints into $INPUTS/models.
#   sbatch --export=ALL,WT=<pinned worktree>,SCRIPTS=<dir holding this script's siblings>,CACHE_JOB=..,CHUNK1_JOB=..,STATE=<dir with mem2>,INPUTS=..,SEL_OUT=.. r1a_prod_chunk2.sbatch
# $STATE/mem2 holds the --mem for the second half (default 150G); edit it before this job submits to lower it.
#SBATCH --job-name=r1a-chunk2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-chunk2-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
set -uo pipefail
WT=${WT:?}; SCRIPTS=${SCRIPTS:?}; CACHE_JOB=${CACHE_JOB:?}; CHUNK1_JOB=${CHUNK1_JOB:?}; STATE=${STATE:?}; INPUTS=${INPUTS:?}; SEL_OUT=${SEL_OUT:?}
LIMIT=${LIMIT:-44}; THROTTLE=${THROTTLE:-12}; N2=21
ARMS="gnn_eng twin_eng mlp_same gnn_raw twin_raw gnn_eng_physmp set_transformer"
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
echo "waiter for chunk 2 | worktree $(git rev-parse HEAD) | scripts $SCRIPTS"
NEXT=21; LAST=41; SUBS=""; SEL_SUBMITTED=0; DONE=""
export PYTHONPATH="$WT"
while true; do
  # (a) the second half goes in as explicit, logged sub-arrays in index order, each as large as the room under the limit allows (>= 3 elements,
  #     or whatever is left when that is fewer); selection (afterok on the first half and every sub-array) follows the last one
  if (( NEXT <= LAST )); then
    st=$(sacct -j "$CACHE_JOB" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*) echo "FAIL LOUD: the cache job $CACHE_JOB is $st; the rest of chunk 2 is not submitted"; exit 1;; esac
    n=$(squeue -u "$USER" -r -h | wc -l)   # includes this job
    room=$((LIMIT - n)); left=$((LAST - NEXT + 1)); size=$(( room < left ? room : left ))
    if (( size >= 3 || (size >= left && size >= 1) )); then
      END=$((NEXT + size - 1)); MEM=$(cat "$STATE/mem2" 2>/dev/null || echo 150G); TH=$(( size < THROTTLE ? size : THROTTLE ))
      out=$(sbatch --parsable --dependency=afterok:"$CACHE_JOB" --array="$NEXT-$END%$TH" --mem="$MEM" --export=ALL,WT="$WT",ORDER_PY="$SCRIPTS/../r1a_task_order.py" "$SCRIPTS/r1a_prod_train.sbatch" 2>&1)
      if [[ "$out" =~ ^[0-9]+$ ]]; then
        SUBS="$SUBS:$out"; echo "$(date +%T) SUB-ARRAY indices $NEXT-$END -> job $out (mem $MEM, $n jobs queued before, room $room)"; NEXT=$((END + 1))
      elif echo "$out" | grep -qiE "limit|QOS|too many"; then echo "$(date +%T) sub-array $NEXT-$END refused at the limit ($out); retrying"
      else echo "FAIL LOUD: sbatch refused sub-array $NEXT-$END: $out"; exit 1; fi
    else
      echo "$(date +%T) $n jobs queued, room $room < 3 for indices $NEXT-$LAST; waiting"
    fi
  elif (( SEL_SUBMITTED == 0 )); then
    SEL=$(sbatch --parsable --dependency=afterok:"$CHUNK1_JOB$SUBS" --export=ALL,WT="$WT",INPUTS="$INPUTS",OUT="$SEL_OUT" "$SCRIPTS/r1a_prod_select.sbatch" 2>&1)
    [[ "$SEL" =~ ^[0-9]+$ ]] || { echo "FAIL LOUD: selection submission refused: $SEL (sub-arrays$SUBS)"; exit 1; }
    echo "all of 21-41 submitted as sub-arrays$SUBS; selection = $SEL (afterok first half $CHUNK1_JOB$SUBS)"; SEL_SUBMITTED=1
  fi
  # (b) per-arm selection as arms complete
  left_arms=0
  for A in $ARMS; do
    [[ " $DONE " == *" $A "* ]] && continue
    if [[ -f "$STATE/selection_$A.json" ]]; then DONE="$DONE $A"; continue; fi
    python3 "$SCRIPTS/../r1a_select_checkpoints.py" --models-dir /home/nikola.lukic/gnn-herosim/models --inputs-dir "$INPUTS" \
      --split "$WT/experiments/r1_attribution_v1_split.json" --out "$STATE/selection_$A.json" --arm "$A" --if-complete; rc=$?
    case $rc in
      0) echo "$(date +%T) ARM SELECTED: $A -> $INPUTS/models ($STATE/selection_$A.json)"; DONE="$DONE $A";;
      3) left_arms=$((left_arms+1));;
      *) echo "FAIL LOUD: per-arm selection of $A failed (rc $rc)"; exit 1;;
    esac
  done
  if (( NEXT > LAST && SEL_SUBMITTED == 1 && left_arms == 0 )); then echo "all arms selected"; exit 0; fi
  sleep 300
done
