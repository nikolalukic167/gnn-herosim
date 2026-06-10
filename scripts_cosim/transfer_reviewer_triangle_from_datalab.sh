#!/usr/bin/env bash
# Pull reviewer_triangle_all7 sweep results from datalab to mitrix/local.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
LOCAL_DIR="${ROOT}/simulation_data/normal_sim_sweeps/reviewer_triangle_all7_20260609"

mkdir -p "$LOCAL_DIR"

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/simulation_data/normal_sim_sweeps/reviewer_triangle_all7_20260609/" \
  "$LOCAL_DIR/"

echo "[+] Pulled to ${LOCAL_DIR}"
ls "$LOCAL_DIR/results/"*_mlp_batch.json 2>/dev/null | wc -l | xargs -I{} echo "  MLP JSONs: {}"
ls "$LOCAL_DIR/results/"*_xgboost_batch.json 2>/dev/null | wc -l | xargs -I{} echo "  XGB JSONs: {}"
ls "$LOCAL_DIR/results/"*_gnn_dim14_ce.json 2>/dev/null | wc -l | xargs -I{} echo "  GNN JSONs: {}"
