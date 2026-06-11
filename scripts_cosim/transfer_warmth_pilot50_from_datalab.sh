#!/usr/bin/env bash
# Pull datalab warmth pilot second-half datasets back to mitrix.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"

mkdir -p "simulation_data/${OUTPUT_SUBDIR}" logs

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/simulation_data/${OUTPUT_SUBDIR}/" \
  "simulation_data/${OUTPUT_SUBDIR}/"

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/logs/progress_${OUTPUT_SUBDIR}.txt" \
  logs/ 2>/dev/null || true

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/logs/warmth_regen_pilot50_datalab_"*.log \
  logs/ 2>/dev/null || true

echo "[+] Pull complete -> simulation_data/${OUTPUT_SUBDIR}/"
