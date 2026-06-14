#!/usr/bin/env bash
# Push warmth_v2 + sparse_warmth_v2 repair scripts to datalab and submit SLURM arrays.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

for f in \
  scripts_cosim/refresh_optimal_full_stats.py \
  scripts_cosim/datalab/run_warmth_repair_shard.sh \
  scripts_cosim/datalab/warmth_repair_warmth.sbatch \
  scripts_cosim/datalab/warmth_repair_sparse.sbatch \
  scripts_cosim/datalab/submit_warmth_repair_merged.sh; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/logs \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/simulation_data/gnn_datasets_4tasks_1060_warmth_v2 \
  ${REPO}/simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 \
  ${REPO}/src/policy"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/refresh_optimal_full_stats.py \
  "${REMOTE}:${REPO}/scripts_cosim/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_warmth_repair_shard.sh \
  scripts_cosim/datalab/warmth_repair_warmth.sbatch \
  scripts_cosim/datalab/warmth_repair_sparse.sbatch \
  scripts_cosim/datalab/submit_warmth_repair_merged.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

rsync -avP -e "$RSYNC_SSH" \
  src/executecosimulation.py \
  "${REMOTE}:${REPO}/src/"

rsync -avP -e "$RSYNC_SSH" \
  src/policy/state_capture.py \
  "${REMOTE}:${REPO}/src/policy/"

ssh -i "$SSH_KEY" "$REMOTE" \
  "cd ${REPO} && sed -i 's/\r$//' scripts_cosim/datalab/*.sh scripts_cosim/datalab/*.sbatch && \
  bash scripts_cosim/datalab/submit_warmth_repair_merged.sh"

echo "Repair jobs submitted on datalab (warmth 500 + sparse 351 = 824 ds)"
