#!/usr/bin/env bash
# Pull fixed MLP bipartite v2 results from datalab.
set -euo pipefail

SSH_KEY="${SSH_KEY:-~/.ssh/nikolalukic167}"
REMOTE="nikola.lukic@datalab.ijs.si"
REMOTE_ROOT="/home/nikola.lukic/gnn-herosim"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP="bipartite_v2_mlp_dim22_fix_20260614"

mkdir -p "${LOCAL_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/results"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${REMOTE}:${REMOTE_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/" \
  "${LOCAL_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/"

echo "Results in: simulation_data/normal_sim_sweeps/${SWEEP}/results/"
ls "${LOCAL_ROOT}/simulation_data/normal_sim_sweeps/${SWEEP}/results/" 2>/dev/null || echo "(empty — jobs may not have completed yet)"
