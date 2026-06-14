#!/usr/bin/env bash
# Transfer fixed dim22 MLP model + scripts to datalab for Group 2 rerun.
set -euo pipefail

SSH_KEY="${SSH_KEY:-~/.ssh/nikolalukic167}"
REMOTE="nikola.lukic@datalab.ijs.si"
REMOTE_ROOT="/home/nikola.lukic/gnn-herosim"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${LOCAL_ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22.pt" \
  "${LOCAL_ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22.pt.meta.json" \
  "${REMOTE}:${REMOTE_ROOT}/models/tabular/"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${LOCAL_ROOT}/scripts_cosim/datalab/run_bipartite_mlp_dim22_fix_one.sh" \
  "${LOCAL_ROOT}/scripts_cosim/datalab/bipartite_mlp_dim22_fix.sbatch" \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/datalab/"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${LOCAL_ROOT}/src/policy/tabular/reduced_features.py" \
  "${LOCAL_ROOT}/src/policy/tabular/train_mlp_dim22_from_seq.py" \
  "${REMOTE}:${REMOTE_ROOT}/src/policy/tabular/"

echo ""
echo "=== Post-transfer on datalab ==="
echo "  cd ${REMOTE_ROOT}"
echo "  sed -i 's/\r\$//' scripts_cosim/datalab/run_bipartite_mlp_dim22_fix_one.sh"
echo "  sed -i 's/\r\$//' scripts_cosim/datalab/bipartite_mlp_dim22_fix.sbatch"
echo "  chmod +x scripts_cosim/datalab/run_bipartite_mlp_dim22_fix_one.sh"
echo "  sbatch scripts_cosim/datalab/bipartite_mlp_dim22_fix.sbatch"
