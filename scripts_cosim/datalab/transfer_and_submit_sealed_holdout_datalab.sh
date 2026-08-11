#!/usr/bin/env bash
# Transfer 873/v5.5 sealed-holdout artifacts to datalab and submit GNN array.
# Knative+MLP results already done locally; GNN runs on datalab GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATALAB_USER="${DATALAB_USER:-nikola.lukic}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
DEST_HOST="${DEST_HOST:-cluster.datalab.tuwien.ac.at}"
REMOTE="${DATALAB_USER}@${DEST_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
SWEEP_REL="simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806"

SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes)
RSYNC=(rsync -avP -e "ssh -i ${SSH_KEY}")

[[ -f "$SSH_KEY" ]] || { echo "ERROR: SSH key missing: $SSH_KEY" >&2; exit 1; }
[[ -f models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt ]] || {
  echo "ERROR: GNN ckpt missing" >&2; exit 1
}

echo "=== 1) remote mkdir ==="
"${SSH[@]}" "$REMOTE" "mkdir -p \
  ${REMOTE_ROOT}/models/tabular \
  ${REMOTE_ROOT}/logs \
  ${REMOTE_ROOT}/${SWEEP_REL}/results \
  ${REMOTE_ROOT}/${SWEEP_REL}/configs \
  ${REMOTE_ROOT}/scripts_cosim/datalab \
  ${REMOTE_ROOT}/scripts_cosim/important"

echo "=== 2) rsync models ==="
"${RSYNC[@]}" \
  models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt \
  models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt.meta.json \
  "${REMOTE}:${REMOTE_ROOT}/models/"

"${RSYNC[@]}" \
  models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt \
  models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt.meta.json \
  "${REMOTE}:${REMOTE_ROOT}/models/tabular/"

echo "=== 3) rsync scripts ==="
"${RSYNC[@]}" \
  scripts_cosim/datalab/run_sealed_holdout_one.sh \
  scripts_cosim/datalab/sealed_holdout_gpu.sbatch \
  scripts_cosim/datalab/submit_sealed_holdout.sh \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/datalab/"

"${RSYNC[@]}" \
  scripts_cosim/important/compare_sealed_live_holdout.py \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/important/"

echo "=== 4) rsync sweep manifest + completed GNN only (SKIP s42) ==="
"${RSYNC[@]}" \
  "${SWEEP_REL}/manifest.json" \
  "${REMOTE}:${REMOTE_ROOT}/${SWEEP_REL}/"
"${RSYNC[@]}" \
  "${SWEEP_REL}/configs/" \
  "${REMOTE}:${REMOTE_ROOT}/${SWEEP_REL}/configs/"

# Do not ship 40×140MB Kn/MLP JSONs — GNN array only needs existing GNN for SKIP.
if compgen -G "${SWEEP_REL}/results/*_gnn.json" > /dev/null; then
  "${RSYNC[@]}" \
    "${SWEEP_REL}/results/"*_gnn.json \
    "${SWEEP_REL}/results/"*_gnn.decode_stats.json \
    "${REMOTE}:${REMOTE_ROOT}/${SWEEP_REL}/results/" 2>/dev/null || \
  "${RSYNC[@]}" \
    "${SWEEP_REL}/results/"*_gnn.json \
    "${REMOTE}:${REMOTE_ROOT}/${SWEEP_REL}/results/"
fi

echo "=== 5) CRLF heal + submit on datalab ==="
"${SSH[@]}" "$REMOTE" "bash -lc '
  set -euo pipefail
  cd ${REMOTE_ROOT}
  sed -i \"s/\\r\$//\" \
    scripts_cosim/datalab/run_sealed_holdout_one.sh \
    scripts_cosim/datalab/sealed_holdout_gpu.sbatch \
    scripts_cosim/datalab/submit_sealed_holdout.sh
  chmod +x scripts_cosim/datalab/run_sealed_holdout_one.sh \
           scripts_cosim/datalab/submit_sealed_holdout.sh
  POLICY=gnn ARRAY=0-19 bash scripts_cosim/datalab/submit_sealed_holdout.sh
'"

echo "=== done ==="
echo "Pull later:"
echo "  rsync -avP -e \"ssh -i ${SSH_KEY}\" \\"
echo "    ${REMOTE}:${REMOTE_ROOT}/${SWEEP_REL}/results/ \\"
echo "    ${SWEEP_REL}/results/"
