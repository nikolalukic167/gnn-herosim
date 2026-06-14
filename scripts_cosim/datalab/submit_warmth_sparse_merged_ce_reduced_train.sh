#!/usr/bin/env bash
# Submit merged warmth+sparse CE-reduced GNN from-scratch training on datalab GPU.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a40}"
CPUS="${CPUS:-16}"
MEM="${MEM:-64G}"
TIME="${TIME:-08:00:00}"

for f in \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_train.sbatch \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_train.sh \
  src/notebooks/train_near_rtt_v2_warmth_sparse_merged_ce_reduced.py \
  src/notebooks/train_near_rtt_ce_reduced_features.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_train.sbatch \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_train.sh \
  scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_train.sh

echo "=== Cancel pending/running ws-merged-cr jobs ==="
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/ws-merged-cr/ {print $1}'); do
  echo "scancel ${j}"
  scancel "${j}" || true
done
sleep 2

echo
echo "=== Dataset counts ==="
echo "warmth: $(find simulation_data/gnn_datasets_4tasks_1060_warmth_v2 -name best.json 2>/dev/null | wc -l)"
echo "sparse: $(find simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 -name best.json 2>/dev/null | wc -l)/351"

echo
echo "Test-only schedule (${PARTITION}, ${CPUS} CPUs, gpu:a40:1):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --gres=gpu:a40:1 \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_train.sbatch

job_id=$(sbatch \
  --job-name=ws-merged-cr \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --gres=gpu:a40:1 \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_train.sbatch | awk '{print $NF}')

echo "Submitted warmth+sparse merged CE-reduced train: ${job_id}"
echo "Monitor: squeue -u \$USER · tail -f logs/ws-merged-cr-${job_id}.out"
echo "Checkpoint (when done): models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt"
