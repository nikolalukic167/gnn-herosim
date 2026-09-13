#!/usr/bin/env bash
# SSC refresh (sparse) -> merged graph cache -> finetune dim14-ce from warmth checkpoint.
set -euo pipefail
cd "$(dirname "$0")/../.."

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
CACHE_DIR="simulation_data/graphs_cache_warmth_v2_sparse_merged"
INIT_CKPT="models/near-rtt-v2-warmth-dim14-ce-only.pt"
OUT_CKPT="models/near-rtt-v2-warmth-sparse-finetune-dim14-ce-only.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/warmth_sparse_recache_finetune_${TS}.log"

mkdir -p logs models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== warmth+sparse pipeline ${TS} ==="

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: missing init checkpoint $INIT_CKPT" >&2
  exit 1
fi

sparse_done=$(find "$SPARSE_DIR" -name best.json 2>/dev/null | wc -l)
if [[ "$sparse_done" -lt 351 ]]; then
  echo "ERROR: sparse dir has ${sparse_done}/351 best.json — run transfer_warmth_sparse_from_datalab.sh first" >&2
  exit 1
fi

echo "=== refresh_optimal_full_stats (sparse, --rewrite-ssc) ==="
pipenv run python3 scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir "$SPARSE_DIR" \
  --rewrite-ssc

echo "=== prepare_graphs_cache (merged warmth_v2 + sparse) ==="
PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
  --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" \
  --cache-dir "$CACHE_DIR"

echo "=== finetune dim14-ce (init=$INIT_CKPT) ==="
export PYTHONPATH="$(pwd)"
export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE_DIR}"
export TRAIN_INIT_CHECKPOINT="$(pwd)/${INIT_CKPT}"
cd src/notebooks
pipenv run python3 train_near_rtt_v2_warmth_sparse_finetune.py
cd ../..

NB_MODEL="src/notebooks/models/near-rtt-v2-warmth-sparse-finetune-dim14-ce-only.pt"
if [[ ! -f "$NB_MODEL" ]]; then
  echo "ERROR: finetune finished but checkpoint missing: $NB_MODEL" >&2
  exit 1
fi
cp -f "$NB_MODEL" "$OUT_CKPT"
echo "=== Done ==="
echo "Cache: ${CACHE_DIR}"
echo "Checkpoint: ${OUT_CKPT}"
echo "Log: ${LOG}"
