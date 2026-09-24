#!/usr/bin/env bash
# CE-reduced GNN from scratch on merged warmth+sparse cache.
# Default: train only if graphs.pkl already present (e.g. rsynced from mitrix).
# Set FORCE_RECACHE=1 to rebuild cache locally (SSC refresh + prepare_graphs_cache).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
CACHE_DIR="simulation_data/graphs_cache_warmth_v2_sparse_merged"
OUT_CKPT="models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt"
NB_MODEL="src/notebooks/models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/warmth_sparse_merged_ce_reduced_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== warmth+sparse merged CE-reduced from-scratch train ${TS} ==="
echo "Host: $(hostname)"
echo "Cache: ${CACHE_DIR}"
echo "Output: ${OUT_CKPT}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

python3 -c 'import importlib.util, sys; req=["torch","torch_geometric"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
  || { echo "ERROR: micromamba env ${ENV_NAME} missing torch/torch_geometric" >&2; exit 1; }

sparse_done=$(find "$SPARSE_DIR" -name best.json 2>/dev/null | wc -l)
if [[ -f "${CACHE_DIR}/graphs.pkl" && "${FORCE_RECACHE:-0}" != "1" ]]; then
  n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
  echo "Using prebuilt cache: ${CACHE_DIR} (${n_graphs} graphs)"
else
  if [[ "$sparse_done" -lt 351 ]]; then
    echo "ERROR: sparse dir has ${sparse_done}/351 best.json" >&2
    exit 1
  fi
  warmth_done=$(find "$WARMTH_DIR" -name best.json 2>/dev/null | wc -l)
  echo "Datasets: warmth=${warmth_done}, sparse=${sparse_done}/351"

  echo "=== refresh_optimal_full_stats (sparse, --rewrite-ssc) ==="
  export PYTHONPATH="${PROJECT_ROOT}"
  python3 scripts_cosim/refresh_optimal_full_stats.py \
    --base-dir "$SPARSE_DIR" \
    --rewrite-ssc

  echo "=== prepare_graphs_cache (merged warmth_v2 + sparse) ==="
  python3 src/notebooks/prepare_graphs_cache.py \
    --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" \
    --cache-dir "$CACHE_DIR"
fi

echo "=== CE-reduced GNN train from scratch (100ep, no init checkpoint) ==="
unset TRAIN_INIT_CHECKPOINT
export NEAR_RTT_DATALOADER_WORKERS="${NEAR_RTT_DATALOADER_WORKERS:-0}"
export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
cd src/notebooks
python3 train_near_rtt_v2_warmth_sparse_merged_ce_reduced.py
cd "$PROJECT_ROOT"

if [[ ! -f "$NB_MODEL" ]]; then
  echo "ERROR: training finished but checkpoint missing: $NB_MODEL" >&2
  exit 1
fi

cp -f "$NB_MODEL" "$OUT_CKPT"
echo "=== Done ==="
echo "Checkpoint: ${OUT_CKPT}"
echo "Log: ${LOG}"
