#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="${SSH_KEY:-~/.ssh/nikolalukic167}"
REMOTE="nikola.lukic@cluster.datalab.tuwien.ac.at"
REMOTE_ROOT="/home/nikola.lukic/gnn-herosim"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP="bipartite_v2_mlp_dim22_batchcache_20260614"

mkdir -p "${LOCAL_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/results"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${REMOTE}:${REMOTE_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/" \
  "${LOCAL_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/"

echo "Results:"
ls "${LOCAL_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/results/"*.json 2>/dev/null | grep -v decode_stats || echo "(none yet)"
