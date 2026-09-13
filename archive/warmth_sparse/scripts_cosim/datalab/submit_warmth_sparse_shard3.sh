#!/usr/bin/env bash
# Cancel stuck array task 3; submit standalone 64-CPU shard for ds_00264–00350.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-64}"
ARRAY_JOB="${ARRAY_JOB:-480275}"

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_sparse_shard3.sbatch \
  scripts_cosim/datalab/run_warmth_regen_range.sh

echo "=== Cancel array task ${ARRAY_JOB}_3 (if pending) ==="
scancel "${ARRAY_JOB}_3" 2>/dev/null || true
sleep 2

echo
echo "=== best.json shard 3 before submit ==="
cnt=0
for i in $(seq 264 350); do
  [ -f "simulation_data/gnn_datasets_4tasks_sparse_warmth_v2/ds_$(printf '%05d' "$i")/best.json" ] && cnt=$((cnt + 1))
done
echo "shard 3: ${cnt}/87"

echo
echo "Test-only (${PARTITION}, ${CPUS} CPUs, ds_00264–00350):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_sparse_shard3.sbatch

job_id=$(sbatch \
  --job-name=warmth-sparse-s3 \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_sparse_shard3.sbatch | awk '{print $NF}')

echo "Submitted standalone shard 3: ${job_id} (${CPUS} CPUs, no GPU)"
echo "Monitor: squeue -u \$USER · tail -f logs/warmth-sparse-s3-${job_id}.out"
