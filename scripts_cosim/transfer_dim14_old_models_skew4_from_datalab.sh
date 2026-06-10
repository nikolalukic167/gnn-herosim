#!/usr/bin/env bash
# Pull skew-4 legacy comparison results from datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/dim14_old_models_skew4_20260610}"
LOCAL_DIR="${ROOT}/${SWEEP_DIR}"

mkdir -p "$LOCAL_DIR"

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/${SWEEP_DIR}/" \
  "$LOCAL_DIR/"

echo "[+] Pulled to ${LOCAL_DIR}"
ls "$LOCAL_DIR/results/"*.json 2>/dev/null | wc -l | xargs -I{} echo "  Result JSONs: {}"
