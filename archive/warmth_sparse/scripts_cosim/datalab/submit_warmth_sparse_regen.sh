#!/usr/bin/env bash
# Submit 4-shard SLURM array for sparse_warmth_v2 co-sim regen (351 ds, Option A).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-96}"
TIME="${TIME:-24:00:00}"

for f in \
  scripts_cosim/datalab/warmth_sparse_regen.sbatch \
  scripts_cosim/datalab/run_warmth_regen_shard.sh \
  scripts_cosim/generate_gnn_datasets_fast.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_sparse_regen.sbatch \
  scripts_cosim/datalab/run_warmth_regen_shard.sh

echo "=== Cancel pending/running warmth-sparse jobs ==="
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/warmth-sparse/ {print $1}'); do
  echo "scancel ${j}"
  scancel "${j}" || true
done
sleep 2

echo
echo "=== best.json before submit ==="
find "simulation_data/gnn_datasets_4tasks_sparse_warmth_v2" -name best.json 2>/dev/null | wc -l

echo
echo "Test-only schedule (${PARTITION}, ${CPUS} CPUs × 4 array tasks):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --array=0-3 \
  scripts_cosim/datalab/warmth_sparse_regen.sbatch

job_id=$(sbatch \
  --job-name=warmth-sparse \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --array=0-3 \
  scripts_cosim/datalab/warmth_sparse_regen.sbatch | awk '{print $NF}')

echo "Submitted sparse regen array: ${job_id} (tasks 0-3, 351 ds, ${CPUS} CPUs/task, no GPU)"
echo "Monitor: squeue -u \$USER · tail -f logs/warmth-sparse-${job_id}_*.out"
echo "Progress: find simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 -name best.json | wc -l"
