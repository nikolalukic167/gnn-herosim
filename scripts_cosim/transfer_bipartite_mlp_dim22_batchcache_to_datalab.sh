#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="${SSH_KEY:-~/.ssh/nikolalukic167}"
REMOTE="nikola.lukic@cluster.datalab.tuwien.ac.at"
REMOTE_ROOT="/home/nikola.lukic/gnn-herosim"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${LOCAL_ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt" \
  "${LOCAL_ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt.meta.json" \
  "${REMOTE}:${REMOTE_ROOT}/models/tabular/"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${LOCAL_ROOT}/scripts_cosim/datalab/run_bipartite_mlp_dim22_batchcache_one.sh" \
  "${LOCAL_ROOT}/scripts_cosim/datalab/bipartite_mlp_dim22_batchcache.sbatch" \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/datalab/"

echo "On datalab:"
echo "  chmod +x scripts_cosim/datalab/run_bipartite_mlp_dim22_batchcache_one.sh"
echo "  sbatch scripts_cosim/datalab/bipartite_mlp_dim22_batchcache.sbatch"
