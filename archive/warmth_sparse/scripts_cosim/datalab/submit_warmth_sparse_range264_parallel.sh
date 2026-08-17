#!/usr/bin/env bash
# Submit 4 parallel shards for sparse_warmth_v2 ds_00264–00350 (87 ds).
# Uses 48 CPUs/shard on GPU-a100s so all four can start immediately.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-48}"
CANCEL_STALE="${CANCEL_STALE:-1}"

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_sparse_range264.sbatch \
  scripts_cosim/datalab/run_warmth_regen_range.sh

if [[ "${CANCEL_STALE}" == "1" ]]; then
  echo "=== Cancel stale jobs on range 264-350 ==="
  for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/warmth-sparse-s3|warmth-sp264/ {print $1}'); do
    echo "scancel ${j}"
    scancel "${j}" || true
  done
  scancel 480288 2>/dev/null || true
  sleep 2
fi

echo "=== Partition landscape ==="
sinfo -p "${PARTITION}" -o "%P %C"
sinfo -N -p "${PARTITION}" -t idle,mix -o "%N %t %C" | head -8

submit_one() {
  local label=$1 start=$2 count=$3
  export SHARD_LABEL="${label}"
  export START_FROM="${start}"
  export MAX_DATASETS="${count}"
  local end=$((start + count - 1))
  echo
  echo "Submit ${label}: ds_$(printf '%05d' "${start}")..ds_$(printf '%05d' "${end}") (${count} ds, ${CPUS} CPU)"
  sbatch --test-only \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS}" \
    --job-name="warmth-sp264-${label}" \
    --export=ALL,SHARD_LABEL,START_FROM,MAX_DATASETS \
    scripts_cosim/datalab/warmth_sparse_range264.sbatch
  sbatch \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS}" \
    --job-name="warmth-sp264-${label}" \
    --export=ALL,SHARD_LABEL,START_FROM,MAX_DATASETS \
    scripts_cosim/datalab/warmth_sparse_range264.sbatch
}

# 87 datasets: 22 + 22 + 22 + 21
submit_one "s0" 264 22
submit_one "s1" 286 22
submit_one "s2" 308 22
submit_one "s3" 330 21

echo
echo "=== Queue ==="
squeue -u "$USER" -o "%.18i %.9P %.12j %.2t %.10M %C %R" | grep -E "warmth-sp264|JOBID"
