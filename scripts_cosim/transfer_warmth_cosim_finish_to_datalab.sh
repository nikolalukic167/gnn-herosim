#!/usr/bin/env bash
# Push warmth co-sim finish fixes (tail + JSONL BF repair) to datalab and submit.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

for f in \
  simulation_data/space_with_network.json \
  simulation_data/sample_simple.json \
  simulation_data/lhs_samples_simple.npy \
  simulation_data/lhs_samples_simple_mapping.pkl \
  data/nofs-ids/traces/workload-10.json; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/logs \
  ${REPO}/simulation_data \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/src/placement \
  ${REPO}/src/policy \
  ${REPO}/data/nofs-ids/traces"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/generate_gnn_datasets_fast.py \
  scripts_cosim/recover_placements_jsonl_from_scratch.sh \
  "${REMOTE}:${REPO}/scripts_cosim/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_warmth_regen_range.sh \
  scripts_cosim/datalab/warmth_regen_tail.sbatch \
  scripts_cosim/datalab/submit_warmth_regen_tail.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

rsync -avP -e "$RSYNC_SSH" \
  src/executecosimulation.py \
  src/generate_infrastructure.py \
  "${REMOTE}:${REPO}/src/"

rsync -avP -e "$RSYNC_SSH" src/placement/ "${REMOTE}:${REPO}/src/placement/"
rsync -avP -e "$RSYNC_SSH" src/policy/ "${REMOTE}:${REPO}/src/policy/"

rsync -avP -e "$RSYNC_SSH" \
  simulation_data/space_with_network.json \
  simulation_data/sample_simple.json \
  simulation_data/lhs_samples_simple.npy \
  simulation_data/lhs_samples_simple_mapping.pkl \
  "${REMOTE}:${REPO}/simulation_data/"

rsync -avP -e "$RSYNC_SSH" \
  data/nofs-ids/traces/workload-10.json \
  "${REMOTE}:${REPO}/data/nofs-ids/traces/"

echo "[+] Warmth tail scripts synced."
echo "    Submitting tail finisher on datalab..."
ssh -i "$SSH_KEY" "$REMOTE" \
  "cd ${REPO} && sed -i 's/\r$//' scripts_cosim/datalab/*.sh scripts_cosim/datalab/*.sbatch scripts_cosim/*.sh && \
  bash scripts_cosim/datalab/submit_warmth_regen_tail.sh"
