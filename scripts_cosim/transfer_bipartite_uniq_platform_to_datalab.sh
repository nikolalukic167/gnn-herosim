#!/usr/bin/env bash
# Push argmax_uniq probe code + sweep artifacts to datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"

for f in \
  models/near-rtt-v2-dim14-ce-only.pt \
  "$WORKLOAD"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/models \
  ${REPO}/data/nofs-ids/traces \
  ${REPO}/${SWEEP_DIR}/configs \
  ${REPO}/${SWEEP_DIR}/results \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/scripts_cosim \
  ${REPO}/src/policy/gnn \
  ${REPO}/src \
  ${REPO}/logs/bipartite_uniq_platform \
  ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" models/near-rtt-v2-dim14-ce-only.pt "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/configs/" "${REMOTE}:${REPO}/${SWEEP_DIR}/configs/"
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/run_bipartite_uniq_platform_one.sh \
  scripts_cosim/run_simulation.py \
  scripts_cosim/datalab/uniq_platform_probe.sbatch \
  "${REMOTE}:${REPO}/scripts_cosim/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/uniq_platform_probe.sbatch \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/gnn/seq_decode.py \
  src/policy/gnn/scheduler.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"
rsync -avP -e "$RSYNC_SSH" src/executesimulation.py "${REMOTE}:${REPO}/src/"

if [[ -d "${SWEEP_DIR}/results" ]] && [[ -n "$(ls -A "${SWEEP_DIR}/results" 2>/dev/null)" ]]; then
  rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/results/" "${REMOTE}:${REPO}/${SWEEP_DIR}/results/"
fi

echo "[+] Transfer complete."
echo "Submit on datalab:"
echo "  ssh -i ${SSH_KEY} ${REMOTE} 'cd ${REPO} && sed -i \"s/\\r\$//\" scripts_cosim/run_bipartite_uniq_platform_one.sh scripts_cosim/datalab/uniq_platform_probe.sbatch && sbatch scripts_cosim/datalab/uniq_platform_probe.sbatch'"
