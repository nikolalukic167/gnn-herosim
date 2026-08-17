#!/usr/bin/env bash
# Submit warmth_v2 tail finisher only (ds_00488..499, full BF + placements.jsonl).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100}"
CPUS="${CPUS:-64}"

sed -i 's/\r$//' \
  scripts_cosim/datalab/run_warmth_regen_range.sh \
  scripts_cosim/datalab/warmth_regen_tail.sbatch

echo "=== Cancel running warmth-tail jobs ==="
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/warmth-tail/ {print $1}'); do
  echo "scancel ${j}"
  scancel "${j}" || true
done
sleep 2

job_id=$(sbatch --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_regen_tail.sbatch | awk '{print $NF}')
echo "Submitted warmth tail: ${job_id}"
echo "TAIL_JOB_ID=${job_id}"
squeue -u "$USER" | grep -E "warmth|JOBID" || true
