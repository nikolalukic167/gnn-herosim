#!/usr/bin/env bash
# Submit 4-shard SLURM arrays for --repair --force on warmth_v2 + sparse_warmth_v2 (824 ds).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-32}"
TIME="${TIME:-48:00:00}"

for f in \
  scripts_cosim/datalab/warmth_repair_warmth.sbatch \
  scripts_cosim/datalab/warmth_repair_sparse.sbatch \
  scripts_cosim/datalab/run_warmth_repair_shard.sh \
  scripts_cosim/refresh_optimal_full_stats.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_repair_warmth.sbatch \
  scripts_cosim/datalab/warmth_repair_sparse.sbatch \
  scripts_cosim/datalab/run_warmth_repair_shard.sh

echo "=== optimal_result.json counts before submit ==="
echo "warmth: $(find simulation_data/gnn_datasets_4tasks_1060_warmth_v2 -name optimal_result.json 2>/dev/null | wc -l)"
echo "sparse: $(find simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 -name optimal_result.json 2>/dev/null | wc -l)"

echo
echo "Test-only warmth repair (${PARTITION}, ${CPUS} CPUs × 4 array tasks):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --array=0-3 \
  scripts_cosim/datalab/warmth_repair_warmth.sbatch

warmth_job=$(sbatch \
  --job-name=warmth-repair \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --array=0-3 \
  scripts_cosim/datalab/warmth_repair_warmth.sbatch | awk '{print $NF}')

echo "Submitted warmth repair array: ${warmth_job} (500 ds, 4 shards)"

sparse_job=$(sbatch \
  --job-name=warmth-repair-sparse \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --array=0-3 \
  scripts_cosim/datalab/warmth_repair_sparse.sbatch | awk '{print $NF}')

echo "Submitted sparse repair array: ${sparse_job} (351 ds, 4 shards)"
echo "Monitor: squeue -u \$USER · tail -f logs/warmth-repair-${warmth_job}_*.out"
echo "After completion: recache graphs_cache_warmth_v2_features_v1"
