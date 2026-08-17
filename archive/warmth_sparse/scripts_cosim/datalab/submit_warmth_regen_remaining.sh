#!/usr/bin/env bash
# Submit 4-shard SLURM array for remaining warmth_v2 regen (ds_00235+).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-128}"
EXCLUSIVE="${EXCLUSIVE:-}"
TIME="${TIME:-24:00:00}"

for f in \
  scripts_cosim/datalab/warmth_regen_remaining.sbatch \
  scripts_cosim/datalab/run_warmth_regen_shard.sh \
  scripts_cosim/generate_gnn_datasets_fast.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_regen_remaining.sbatch \
  scripts_cosim/datalab/run_warmth_regen_shard.sh

echo "Test-only schedule (${PARTITION}, ${CPUS} CPUs × 4 array tasks, exclusive=${EXCLUSIVE}):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  ${EXCLUSIVE} \
  --array=0-3 \
  scripts_cosim/datalab/warmth_regen_remaining.sbatch

job_id=$(sbatch \
  --job-name=warmth-v2 \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  ${EXCLUSIVE} \
  --array=0-3 \
  scripts_cosim/datalab/warmth_regen_remaining.sbatch | awk '{print $NF}')

echo "Submitted warmth regen array: ${job_id} (tasks 0-3, ds_00235+, ${CPUS} CPUs/task, no GPU)"
echo "Monitor: squeue -u \$USER · tail -f logs/warmth-v2-${job_id}_*.out"
echo "Progress: find simulation_data/gnn_datasets_4tasks_1060_warmth_v2 -name best.json | wc -l"
