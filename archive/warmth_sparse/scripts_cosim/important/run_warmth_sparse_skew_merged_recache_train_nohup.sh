#!/usr/bin/env bash
# warmth+sparse+skew merged cache -> dim14 CE-only + CE-reduced GNN (100ep each, from scratch).
set -euo pipefail
cd "$(dirname "$0")/../.."

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
SKEW_DIR="simulation_data/gnn_datasets_4tasks_skew_warmth_v2"
CACHE_DIR="simulation_data/graphs_cache_warmth_v2_sparse_skew_merged"
DIM14_CKPT="models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt"
CE_RED_CKPT="models/near-rtt-v2-warmth-sparse-skew-merged-ce-reduced.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/warmth_sparse_skew_merged_recache_train_${TS}.log"

mkdir -p logs models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== warmth+sparse+skew merged pipeline ${TS} ==="
echo "Cache: ${CACHE_DIR}"

skew_done=$(find "$SKEW_DIR" -name best.json 2>/dev/null | wc -l)
if [[ "$skew_done" -lt 1 ]]; then
  echo "ERROR: skew dir has ${skew_done} best.json — sync from datalab first" >&2
  exit 1
fi
echo "Datasets with best.json: warmth=$(find "$WARMTH_DIR" -name best.json | wc -l), sparse=$(find "$SPARSE_DIR" -name best.json | wc -l), skew=${skew_done}"

for dir in "$WARMTH_DIR" "$SPARSE_DIR" "$SKEW_DIR"; do
  echo "=== refresh_optimal_full_stats (${dir##*/}, --rewrite-ssc) ==="
  pipenv run python3 scripts_cosim/refresh_optimal_full_stats.py \
    --base-dir "$dir" \
    --rewrite-ssc
done

echo "=== prepare_graphs_cache (warmth + sparse + skew) ==="
PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
  --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" "$SKEW_DIR" \
  --cache-dir "$CACHE_DIR"

n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Merged cache: ${n_graphs} graphs"

export PYTHONPATH="$(pwd)"
export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE_DIR}"

echo "=== dim14 CE-only GNN (100ep, from scratch) ==="
cd src/notebooks
NB_DIM14="models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt"
pipenv run python3 train_near_rtt_v2_warmth_sparse_skew_merged_dim14_ce_only.py
if [[ ! -f "$NB_DIM14" ]]; then
  echo "ERROR: dim14 checkpoint missing: $NB_DIM14" >&2
  exit 1
fi
cp -f "$NB_DIM14" "../../${DIM14_CKPT}"
echo "Saved ${DIM14_CKPT}"

echo "=== CE-reduced GNN (100ep, from scratch) ==="
NB_CE="models/near-rtt-v2-warmth-sparse-skew-merged-ce-reduced.pt"
pipenv run python3 train_near_rtt_v2_warmth_sparse_skew_merged_ce_reduced.py
if [[ ! -f "$NB_CE" ]]; then
  echo "ERROR: ce-reduced checkpoint missing: $NB_CE" >&2
  exit 1
fi
cp -f "$NB_CE" "../../${CE_RED_CKPT}"
cd ../..

echo "=== Done ==="
echo "Cache: ${CACHE_DIR} (${n_graphs} graphs)"
echo "Checkpoints: ${DIM14_CKPT}, ${CE_RED_CKPT}"
echo "Log: ${LOG}"
