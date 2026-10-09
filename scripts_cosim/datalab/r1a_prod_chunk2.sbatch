#!/bin/bash
# r1_attribution_v1 production chain, stage 2: a 1-CPU waiter that submits the second half of the training array (indices 21-41) and the selection job
# as soon as the account has room under the shared 44-job limit (pending array elements count). It never submits a partial array: sbatch either takes
# the whole 21-element array or refuses it, and any refusal that is not a limit refusal fails this job loudly.
#   sbatch --export=ALL,WT=...,CACHE_JOB=...,CHUNK1_JOB=...,STATE=<dir with mem2>,INPUTS=...,SEL_OUT=... r1a_prod_chunk2.sbatch
# $STATE/mem2 holds the --mem for the second half (default 150G); edit it before this job submits to lower it.
#SBATCH --job-name=r1a-chunk2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-chunk2-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=72:00:00
set -uo pipefail
WT=${WT:?}; CACHE_JOB=${CACHE_JOB:?}; CHUNK1_JOB=${CHUNK1_JOB:?}; STATE=${STATE:?}; INPUTS=${INPUTS:?}; SEL_OUT=${SEL_OUT:?}
LIMIT=${LIMIT:-44}; THROTTLE=${THROTTLE:-12}; N2=21
cd "$WT"
echo "waiter for chunk 2 | worktree $(git rev-parse HEAD)"
while true; do
  st=$(sacct -j "$CACHE_JOB" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*) echo "FAIL LOUD: the cache job $CACHE_JOB is $st; chunk 2 is not submitted"; exit 1;; esac
  n=$(squeue -u "$USER" -r -h | wc -l)   # includes this job
  if (( n + N2 <= LIMIT )); then
    MEM=$(cat "$STATE/mem2" 2>/dev/null || echo 150G)
    out=$(sbatch --parsable --dependency=afterok:"$CACHE_JOB" --array=21-41%"$THROTTLE" --mem="$MEM" --export=ALL,WT="$WT" scripts_cosim/datalab/r1a_prod_train.sbatch 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      CHUNK2=$out
      SEL=$(sbatch --parsable --dependency=afterok:"$CHUNK1_JOB":"$CHUNK2" --export=ALL,WT="$WT",INPUTS="$INPUTS",OUT="$SEL_OUT" scripts_cosim/datalab/r1a_prod_select.sbatch 2>&1)
      [[ "$SEL" =~ ^[0-9]+$ ]] || { echo "FAIL LOUD: selection submission refused: $SEL (chunk 2 is $CHUNK2)"; exit 1; }
      echo "chunk 2 = $CHUNK2 (mem $MEM, $((n)) jobs queued before), selection = $SEL"; exit 0
    fi
    if echo "$out" | grep -qiE "limit|QOS|too many"; then echo "$(date +%T) refused at the limit ($out); retrying"; else echo "FAIL LOUD: sbatch refused chunk 2: $out"; exit 1; fi
  else
    echo "$(date +%T) $n jobs queued + $N2 > $LIMIT; waiting"
  fi
  sleep 300
done
