#!/usr/bin/env bash
# Push merged warmth+sparse CE-reduced GNN training pipeline to datalab and submit.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"
CACHE_DIR="simulation_data/graphs_cache_warmth_v2_sparse_merged"
LOCAL_CACHE="${ROOT}/${CACHE_DIR}"

if [[ ! -f "${LOCAL_CACHE}/graphs.pkl" ]]; then
  echo "ERROR: missing merged cache ${LOCAL_CACHE}/graphs.pkl — build on mitrix first" >&2
  exit 1
fi

for f in \
  src/notebooks/train_near_rtt_v2_warmth_sparse_merged_ce_reduced.py \
  src/notebooks/train_near_rtt_ce_reduced_features.py \
  src/notebooks/prepare_graphs_cache.py \
  scripts_cosim/refresh_optimal_full_stats.py \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_train.sh \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_train.sbatch \
  scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_train.sh; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/logs \
  ${REPO}/models \
  ${REPO}/${CACHE_DIR} \
  ${REPO}/src/notebooks/models \
  ${REPO}/src/notebooks/non_unique_lib \
  ${REPO}/src/policy/tabular \
  ${REPO}/scripts_cosim/datalab"

echo "[1/6] rsync merged graph cache (~290M, 824 graphs)..."
rsync -avP -e "$RSYNC_SSH" \
  "${LOCAL_CACHE}/" \
  "${REMOTE}:${REPO}/${CACHE_DIR}/"

echo "[2/6] rsync training notebooks + non_unique_lib..."
rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/train_near_rtt_v2_warmth_sparse_merged_ce_reduced.py \
  src/notebooks/train_near_rtt_ce_reduced_features.py \
  src/notebooks/prepare_graphs_cache.py \
  "${REMOTE}:${REPO}/src/notebooks/"
rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/non_unique_lib/ \
  "${REMOTE}:${REPO}/src/notebooks/non_unique_lib/"

echo "[3/6] rsync cache refresh + cosim deps..."
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/refresh_optimal_full_stats.py \
  "${REMOTE}:${REPO}/scripts_cosim/"
rsync -avP -e "$RSYNC_SSH" \
  src/executecosimulation.py \
  src/executesimulation.py \
  "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" src/placement/ "${REMOTE}:${REPO}/src/placement/"

echo "[4/6] rsync inference layout (ce_reduced)..."
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/reduced_features.py \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/gnn/gnn_model.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"

echo "[5/6] rsync datalab scripts..."
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_train.sh \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_train.sbatch \
  scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_train.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

ssh -i "$SSH_KEY" "$REMOTE" "chmod +x \
  ${REPO}/scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_train.sh \
  ${REPO}/scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_train.sh"

echo "[6/6] submit SLURM job on datalab (train-only, prebuilt cache)..."
ssh -i "$SSH_KEY" "$REMOTE" "cd ${REPO} && bash scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_train.sh"

echo
echo "Transfer + submit complete -> ${REMOTE}:${REPO}"
echo "Monitor: ssh -i ${SSH_KEY} ${REMOTE} 'squeue -u nikola.lukic; tail -f ${REPO}/logs/ws-merged-cr-*.out'"
echo "Pull checkpoint when done:"
echo "  rsync -avP -e \"ssh -i ${SSH_KEY}\" \\"
echo "    ${REMOTE}:${REPO}/models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt \\"
echo "    models/"
