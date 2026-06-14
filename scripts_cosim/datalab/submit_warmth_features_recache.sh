#!/usr/bin/env bash
# Submit single-job SLURM recache for graphs_cache_warmth_v2_features_v1.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-64}"
TIME="${TIME:-24:00:00}"

for f in \
  scripts_cosim/datalab/warmth_features_recache.sbatch \
  scripts_cosim/datalab/run_warmth_features_recache.sh \
  src/notebooks/prepare_graphs_cache_seq.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_features_recache.sbatch \
  scripts_cosim/datalab/run_warmth_features_recache.sh

echo "=== optimal_result counts ==="
echo "warmth: $(find simulation_data/gnn_datasets_4tasks_1060_warmth_v2 -name optimal_result.json 2>/dev/null | wc -l)"
echo "sparse: $(find simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 -name optimal_result.json 2>/dev/null | wc -l)"

echo
echo "Test-only (${PARTITION}, ${CPUS} CPUs):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  scripts_cosim/datalab/warmth_features_recache.sbatch

job_id=$(sbatch \
  --job-name=warmth-recache \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  scripts_cosim/datalab/warmth_features_recache.sbatch | awk '{print $NF}')

echo "Submitted features_v1 recache: ${job_id} (${CPUS} CPUs, ${TIME})"
echo "Monitor: squeue -u \$USER · tail -f logs/warmth-recache-${job_id}.out"
echo "Output: simulation_data/graphs_cache_warmth_v2_features_v1"
