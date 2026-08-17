#!/usr/bin/env bash
# From-scratch dim14 CE-only training on merged warmth_v2 + sparse_warmth_v2 cache.
set -euo pipefail
cd "$(dirname "$0")/../.."

CACHE_DIR="simulation_data/graphs_cache_warmth_v2_sparse_merged"
OUT_CKPT="models/near-rtt-v2-warmth-sparse-merged-dim14-ce-only.pt"
NB_MODEL="src/notebooks/models/near-rtt-v2-warmth-sparse-merged-dim14-ce-only.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_warmth_sparse_merged_dim14_ce_only_${TS}.log"

mkdir -p logs models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== warmth+sparse merged from-scratch train ${TS} ==="
echo "Cache: ${CACHE_DIR}"
echo "Output: ${OUT_CKPT}"

if [[ ! -f "${CACHE_DIR}/graphs.pkl" ]]; then
  echo "ERROR: merged cache missing — run prepare_graphs_cache first" >&2
  exit 1
fi

unset TRAIN_INIT_CHECKPOINT
export PYTHONPATH="$(pwd)"
export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE_DIR}"

cd src/notebooks
pipenv run python3 train_near_rtt_v2_warmth_sparse_merged_dim14_ce_only.py
cd ../..

if [[ ! -f "$NB_MODEL" ]]; then
  echo "ERROR: training finished but checkpoint missing: $NB_MODEL" >&2
  exit 1
fi

cp -f "$NB_MODEL" "$OUT_CKPT"
echo "=== Done ==="
echo "Checkpoint: ${OUT_CKPT}"
echo "Log: ${LOG}"
