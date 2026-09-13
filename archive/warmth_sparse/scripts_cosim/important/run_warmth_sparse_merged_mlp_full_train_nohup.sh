#!/usr/bin/env bash
# warmth+sparse merged Regime A MLP (full 21-d): merged seq cache -> parquet -> grouped CE train (100ep).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
SEQ_CACHE="simulation_data/graphs_cache_warmth_v2_sparse_merged_seq"
PARQUET="simulation_data/artifacts/tabular/batch_edges_warmth_sparse_merged.parquet"
MODEL="models/tabular/batch_edge_mlp_warmth_sparse_merged.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_mlp_warmth_sparse_merged_full_${TS}.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

mkdir -p logs simulation_data/artifacts/tabular models/tabular

log "=== warmth+sparse merged MLP full (21-d) pipeline start ==="
log "Warmth datasets: ${WARMTH_DIR}"
log "Sparse datasets: ${SPARSE_DIR}"
log "Seq cache: ${SEQ_CACHE}"
log "Parquet: ${PARQUET}"
log "Model: ${MODEL}"
log "Init: random (no checkpoint)"

sparse_done=$(find "$SPARSE_DIR" -name best.json 2>/dev/null | wc -l)
if [[ "$sparse_done" -lt 351 ]]; then
  log "ERROR: sparse dir has ${sparse_done}/351 best.json — run transfer first"
  exit 1
fi

log "Phase 1: merged sequential graph cache"
if [[ ! -f "${SEQ_CACHE}/graphs.pkl" ]]; then
  pipenv run python3 -u src/notebooks/prepare_graphs_cache_seq.py \
    --base-dirs "${WARMTH_DIR}" "${SPARSE_DIR}" \
    --cache-dir "${SEQ_CACHE}" \
    2>&1 | tee -a "$LOG"
else
  log "SKIP seq cache build (graphs.pkl exists)"
fi

log "Phase 2: tabular parquet extraction (21-d)"
pipenv run python3 -u src/notebooks/prepare_tabular_dataset.py \
  --cache-dir "${SEQ_CACHE}" \
  --output "${PARQUET}" \
  --regime batch \
  2>&1 | tee -a "$LOG"

log "Phase 3: MLP train (100 epochs, patience 10, grouped CE)"
pipenv run python3 -u -m src.policy.tabular.train_mlp \
  --input "${PARQUET}" \
  --output "${MODEL}" \
  --epochs 100 \
  --patience 10 \
  --hidden-dim 64 \
  --lr 1e-3 \
  --random-state 42 \
  --test-size 0.2 \
  2>&1 | tee -a "$LOG"

if [[ ! -f "${MODEL}" ]]; then
  log "ERROR: training finished but checkpoint missing: ${MODEL}"
  exit 1
fi

log "=== warmth+sparse merged MLP full pipeline complete ==="
log "Checkpoint: ${MODEL}"
log "Meta: ${MODEL}.meta.json"
log "Log: ${LOG}"
