#!/usr/bin/env bash
# Push tiered-hub dim22 125-225 sweep (11 configs, 22 GPU jobs) to datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
BASE_CONFIG="simulation_data/space_with_network.json"
SPARSE_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
SKEW_BASE="simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"

for f in \
  models/near-rtt-v2-dim14-ce-only.pt \
  models/tabular/batch_edge_mlp.pt \
  "$WORKLOAD" \
  "$BASE_CONFIG" \
  "$SPARSE_CFG" \
  "$SKEW_BASE"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

pipenv run python3 scripts_cosim/important/generate_tiered_hub_configs.py \
  --out-dir "${SWEEP_DIR}/configs" \
  --policies dim22 \
  --with-controls

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/models/tabular \
  ${REPO}/data/nofs-ids/traces \
  ${REPO}/simulation_data \
  ${REPO}/${SWEEP_DIR}/configs \
  ${REPO}/${SWEEP_DIR}/results \
  ${REPO}/simulation_data/normal_sim_sweeps/atomic21_skew_configs \
  ${REPO}/simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/scripts_cosim/important \
  ${REPO}/src/policy/tabular \
  ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" models/near-rtt-v2-dim14-ce-only.pt "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" models/tabular/batch_edge_mlp.pt "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "$BASE_CONFIG" "${REMOTE}:${REPO}/simulation_data/"
rsync -avP -e "$RSYNC_SSH" "$SPARSE_CFG" "${REMOTE}:${REPO}/simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/"
rsync -avP -e "$RSYNC_SSH" "$SKEW_BASE" "${REMOTE}:${REPO}/simulation_data/normal_sim_sweeps/atomic21_skew_configs/"
rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/configs/" "${REMOTE}:${REPO}/${SWEEP_DIR}/configs/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/ "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/important/generate_tiered_hub_configs.py \
  scripts_cosim/important/compare_tiered_hub_gnn_mlp_sweep.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"
rsync -avP -e "$RSYNC_SSH" src/executesimulation.py "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  src/policy/tabular/mlp_model.py \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"

if [[ -d "${SWEEP_DIR}/results" ]]; then
  rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/results/" "${REMOTE}:${REPO}/${SWEEP_DIR}/results/"
fi

echo "[+] Transfer complete (22 GPU jobs: gnn_dim22 + mlp_dim22 × 11 configs)."
echo "Submit on datalab:"
echo "  SWEEP_DIR=${SWEEP_DIR} WORKLOAD=${WORKLOAD} ssh -i ${SSH_KEY} ${REMOTE} \\"
echo "    'cd ${REPO} && sed -i \"s/\\r\$//\" scripts_cosim/datalab/*.sh scripts_cosim/datalab/*.sbatch && bash scripts_cosim/datalab/submit_tiered_hub_dim22_125225.sh'"
