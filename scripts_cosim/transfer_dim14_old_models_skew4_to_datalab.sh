#!/usr/bin/env bash
# Push skew-4 legacy comparison artifacts + dim22 inference code to datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/dim14_old_models_skew4_20260610}"
GNN_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp.pt"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
BASE_CONFIG="simulation_data/space_with_network.json"
SPARSE_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
SKEW_CFG_SRC="simulation_data/normal_sim_sweeps/atomic21_skew_configs"

for f in "$GNN_MODEL" "$MLP_MODEL" "$WORKLOAD" "$BASE_CONFIG" "$SPARSE_CFG" \
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
  ${REPO}/simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/scripts_cosim/important \
  ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" "$GNN_MODEL" "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" "$MLP_MODEL" "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "${MLP_MODEL}.meta.json" "${REMOTE}:${REPO}/models/tabular/" 2>/dev/null || true
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "$BASE_CONFIG" "${REMOTE}:${REPO}/simulation_data/"
rsync -avP -e "$RSYNC_SSH" "$SPARSE_CFG" "${REMOTE}:${REPO}/simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/"
rsync -avP -e "$RSYNC_SSH" "${SKEW_CFG_SRC}/" "${REMOTE}:${REPO}/${SKEW_CFG_SRC}/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/ "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/important/run_dim14_old_models_skew4_sweep.sh \
  scripts_cosim/important/compare_atomic21_gnn_mlp_sweep.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"

# dim22 legacy inference (not yet on remote main)
rsync -avP -e "$RSYNC_SSH" src/executesimulation.py "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  src/policy/tabular/mlp_model.py \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/gnn_snapshot_inference.py "${REMOTE}:${REPO}/scripts_cosim/"

# Partial local results (skip completed configs on cluster)
if [[ -d "${SWEEP_DIR}/results" ]]; then
  rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/results/" "${REMOTE}:${REPO}/${SWEEP_DIR}/results/"
fi

echo "[+] Transfer complete."
echo "    Workload:  ${REPO}/${WORKLOAD}"
echo "    Sweep dir: ${REPO}/${SWEEP_DIR}/results/"
echo "    Submit:    SWEEP_DIR=${SWEEP_DIR} WORKLOAD=${WORKLOAD} ssh -i ${SSH_KEY} ${REMOTE} 'cd ${REPO} && sbatch --export=ALL scripts_cosim/datalab/dim14_old_models_skew4.sbatch'"
