#!/usr/bin/env bash
set -euo pipefail

DATALAB_USER="${DATALAB_USER:-nikola.lukic}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
SOURCE_PATH="${SOURCE_PATH:-/share/nikola.lukic/herosim/data/gnn_datasets_4tasks_1500/}"
DEST_PATH="${DEST_PATH:-/root/projects/my-herosim/simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks_1060}"
DEST_HOST="${DEST_HOST:-cluster.datalab.tuwien.ac.at}"

echo "Source: ${DATALAB_USER}@${DEST_HOST}:${SOURCE_PATH}"
echo "Dest:   ${DEST_PATH}"
echo "SSH key: ${SSH_KEY}"
echo ""
echo "Starting incremental rsync (safe to re-run)..."
sleep 2

mkdir -p "${DEST_PATH}"

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${DATALAB_USER}@${DEST_HOST}:${SOURCE_PATH}" \
  "${DEST_PATH}/"

echo ""
echo "Transfer complete."
echo "Quick check:"
find "${DEST_PATH}" -mindepth 1 -maxdepth 1 -type d -name 'ds_*' -exec test -f '{}/best.json' \; -print | wc -l
