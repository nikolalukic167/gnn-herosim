#!/usr/bin/env bash
# Push dim14-ce checkpoint + sweep configs needed for GNN all7 on datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

MODEL="models/near-rtt-v2-dim14-ce-only.pt"
WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"

for f in "$MODEL" "$WORKLOAD"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p ${REPO}/models ${REPO}/data/nofs-ids/traces \
  ${REPO}/simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs \
  ${REPO}/simulation_data/normal_sim_sweeps/reviewer_triangle_all7_20260609/results \
  ${REPO}/scripts_cosim/datalab ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" "$MODEL" "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "${CFG_SRC}/" "${REMOTE}:${REPO}/${CFG_SRC}/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/ "${REMOTE}:${REPO}/scripts_cosim/datalab/"

echo "[+] Transfer complete."
echo "    Model:  ${REPO}/${MODEL}"
echo "    Submit: ssh ... 'cd ${REPO} && sbatch scripts_cosim/datalab/reviewer_triangle_gnn_dim14_ce_all7.sbatch'"
