#!/usr/bin/env bash
# Stop shard 3; split ds_00458–499 between two jobs; fill ds_00263 on idle shard-0 slot.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100}"
CPUS="${CPUS:-64}"

sed -i 's/\r$//' \
  scripts_cosim/datalab/run_warmth_regen_range.sh \
  scripts_cosim/datalab/warmth_regen_tail_split.sbatch

echo "Cancelling shard 3 job 479788 (if running)..."
scancel 479788 2>/dev/null || true
sleep 2

submit_one() {
  local label=$1 start=$2 count=$3
  export SHARD_LABEL="$label"
  export START_FROM="$start"
  export MAX_DATASETS="$count"
  echo "Submit ${label}: ds_$(printf '%05d' "${start}").. ds_$(printf '%05d' "$((start + count - 1))") (${count})"
  sbatch --test-only --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
    --job-name="warmth-${label}" \
    --export=ALL,SHARD_LABEL,START_FROM,MAX_DATASETS \
    scripts_cosim/datalab/warmth_regen_tail_split.sbatch
  sbatch --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
    --job-name="warmth-${label}" \
    --export=ALL,SHARD_LABEL,START_FROM,MAX_DATASETS \
    scripts_cosim/datalab/warmth_regen_tail_split.sbatch
}

# ds_00263 gap (array0), then shard3 lower/upper halves
submit_one "0gap" 263 1
submit_one "3a" 458 21
submit_one "3b" 479 21

echo
squeue -u "$USER" | grep -E "warmth|JOBID"
