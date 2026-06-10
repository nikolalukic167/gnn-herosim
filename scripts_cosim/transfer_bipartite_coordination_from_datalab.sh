#!/usr/bin/env bash
# Pull sweep_bipartite_coordination_v1 results from datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
LOCAL_DIR="${ROOT}/${SWEEP_DIR}"

mkdir -p "$LOCAL_DIR"

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/${SWEEP_DIR}/" \
  "$LOCAL_DIR/"

echo "[+] Pulled to ${LOCAL_DIR}"
pipenv run python3 scripts_cosim/important/compare_bipartite_coordination_sweep.py --sweep-dir "$SWEEP_DIR"
