#!/usr/bin/env bash
# Push completed strategic-merge + weighted-merge live gate results to datalab (no SLURM submit).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEPS=(
  simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616
  simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616
  simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500
)

MODELS=(
  models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt
  models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt
  models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt
  models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt
)

for sweep in "${SWEEPS[@]}"; do
  ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p ${REPO}/${sweep}/results"
done
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p ${REPO}/models/tabular"

for m in "${MODELS[@]}"; do
  rsync -avP -e "$RSYNC_SSH" "$m" "${REMOTE}:${REPO}/$(dirname "$m")/"
done

for sweep in "${SWEEPS[@]}"; do
  echo "[+] rsync ${sweep}/results/"
  rsync -avP -e "$RSYNC_SSH" "${sweep}/results/" "${REMOTE}:${REPO}/${sweep}/results/"
done

echo "[+] Results rsync complete (3 sweeps, 30 JSONs)."
