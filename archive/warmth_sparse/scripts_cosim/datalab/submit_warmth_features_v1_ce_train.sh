#!/usr/bin/env bash
# Submit CE-only GNN train on graphs_cache_warmth_v2_features_v1 (cache must exist on datalab).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a40}"
TIME="${TIME:-08:00:00}"

CACHE="simulation_data/graphs_cache_warmth_v2_features_v1"
if [[ ! -f "${CACHE}/graphs.pkl" ]]; then
  echo "ERROR: missing ${CACHE}/graphs.pkl" >&2
  exit 1
fi

sbatch --test-only \
  --partition="${PARTITION}" \
  --gres=gpu:a40:1 \
  --cpus-per-task=16 \
  --time="${TIME}" \
  scripts_cosim/datalab/warmth_features_v1_ce_train.sbatch

job_id=$(sbatch \
  --job-name=wfv1-ce \
  --partition="${PARTITION}" \
  --gres=gpu:a40:1 \
  --cpus-per-task=16 \
  --time="${TIME}" \
  scripts_cosim/datalab/warmth_features_v1_ce_train.sbatch | awk '{print $NF}')

echo "Submitted features_v1 CE-only train: ${job_id}"
echo "Monitor: squeue -u \$USER · tail -f logs/wfv1-ce-${job_id}.out"
