#!/usr/bin/env bash
# Push tiered-hub 45-job sweep to datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_20260610}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"

for f in \
  models/near-rtt-v2-dim14-ce-only.pt \
  models/tabular/batch_edge_mlp.pt \
  models/tabular/batch_edge_mlp_atomic21.pt \
  "$WORKLOAD"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

pipenv run python3 scripts_cosim/important/generate_tiered_hub_configs.py --out-dir "${SWEEP_DIR}/configs"

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/models/tabular \
  ${REPO}/data/nofs-ids/traces \
  ${REPO}/${SWEEP_DIR}/configs \
  ${REPO}/${SWEEP_DIR}/results \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/scripts_cosim/important \
  ${REPO}/src/policy/tabular \
  ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" models/near-rtt-v2-dim14-ce-only.pt "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" models/tabular/batch_edge_mlp.pt models/tabular/batch_edge_mlp_atomic21.pt \
  "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/configs/" "${REMOTE}:${REPO}/${SWEEP_DIR}/configs/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/ "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/important/generate_tiered_hub_configs.py \
  scripts_cosim/important/compare_tiered_hub_gnn_mlp_sweep.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"
rsync -avP -e "$RSYNC_SSH" src/executesimulation.py src/generate_infrastructure.py "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  src/policy/tabular/mlp_model.py \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"

if [[ -d "${SWEEP_DIR}/results" ]]; then
  rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/results/" "${REMOTE}:${REPO}/${SWEEP_DIR}/results/"
fi

echo "[+] Transfer complete (45 jobs: 36 GPU + 9 CPU)."
echo "Submit on datalab:"
echo "  SWEEP_DIR=${SWEEP_DIR} WORKLOAD=${WORKLOAD} ssh -i ${SSH_KEY} ${REMOTE} \\"
echo "    'cd ${REPO} && bash scripts_cosim/datalab/submit_tiered_hub_all.sh'"
