#!/usr/bin/env bash
# Push warmth-v2 ce-reduced skew3 sweep scripts + v2-trained inference artifacts to datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_v2_ce_reduced_skew3_20260611}"
GNN_MODEL="models/near-rtt-v2-warmth-ce-reduced.pt"
MLP_MODEL="models/tabular/batch_edge_mlp_warmth_ce_reduced.pt"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
BASE_CONFIG="simulation_data/space_with_network.json"
SKEW_CFG_SRC="simulation_data/normal_sim_sweeps/atomic21_skew_configs"

for f in "$GNN_MODEL" "$MLP_MODEL" "$WORKLOAD" "$BASE_CONFIG" \
  "${SKEW_CFG_SRC}/default_20_20_degree_skew.json" \
  "${SKEW_CFG_SRC}/05_sparse_40_40_p25_degree_skew.json"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/models/tabular \
  ${REPO}/data/nofs-ids/traces \
  ${REPO}/simulation_data \
  ${REPO}/${SWEEP_DIR}/results \
  ${REPO}/${SKEW_CFG_SRC} \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/logs"

echo "[1/3] rsync ce-reduced models..."
rsync -avP -e "$RSYNC_SSH" "$GNN_MODEL" "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" "$MLP_MODEL" "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "${MLP_MODEL}.meta.json" "${REMOTE}:${REPO}/models/tabular/"

echo "[2/3] rsync workload + configs..."
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "$BASE_CONFIG" "${REMOTE}:${REPO}/simulation_data/"
rsync -avP -e "$RSYNC_SSH" "${SKEW_CFG_SRC}/" "${REMOTE}:${REPO}/${SKEW_CFG_SRC}/"

echo "[3/3] rsync inference code + datalab scripts..."
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/run_warmth_v2_ce_reduced_skew3_one.sh \
  scripts_cosim/datalab/warmth_v2_ce_reduced_skew3_gpu.sbatch \
  scripts_cosim/datalab/submit_warmth_v2_ce_reduced_skew3.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/run_simulation.py "${REMOTE}:${REPO}/scripts_cosim/"
rsync -avP -e "$RSYNC_SSH" src/placement/warmth.py src/executesimulation.py "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" src/placement/simulation.py "${REMOTE}:${REPO}/src/placement/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  src/policy/tabular/mlp_model.py \
  src/policy/tabular/reduced_features.py \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/gnn/seq_decode.py \
  src/policy/gnn/gnn_model.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"

ssh -i "$SSH_KEY" "$REMOTE" "chmod +x \
  ${REPO}/scripts_cosim/datalab/run_warmth_v2_ce_reduced_skew3_one.sh \
  ${REPO}/scripts_cosim/datalab/submit_warmth_v2_ce_reduced_skew3.sh"

echo "Transfer complete -> ${REMOTE}:${REPO}"
echo "Submit on datalab: bash scripts_cosim/datalab/submit_warmth_v2_ce_reduced_skew3.sh"
