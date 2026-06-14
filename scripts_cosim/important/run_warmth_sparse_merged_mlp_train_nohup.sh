#!/usr/bin/env bash
# warmth+sparse merged CE-reduced MLP: merged seq cache -> grouped CE train (100ep, from scratch).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
SEQ_CACHE="simulation_data/graphs_cache_warmth_v2_sparse_merged_seq"
MODEL="models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_mlp_warmth_sparse_merged_ce_reduced_${TS}.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

mkdir -p logs models/tabular

log "=== warmth+sparse merged MLP CE-reduced pipeline start ==="
log "Warmth datasets: ${WARMTH_DIR}"
log "Sparse datasets: ${SPARSE_DIR}"
log "Seq cache: ${SEQ_CACHE}"
log "Layout: task=3 platform=6 edge=2 (11-d)"
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

log "Phase 2: CE-reduced MLP train (100 epochs, patience 10, grouped CE)"
pipenv run python3 -u src/notebooks/train_mlp_v2_warmth_sparse_merged_ce_reduced.py 2>&1 | tee -a "$LOG"

if [[ ! -f "${MODEL}" ]]; then
  log "ERROR: training finished but checkpoint missing: ${MODEL}"
  exit 1
fi

log "=== warmth+sparse merged MLP CE-reduced pipeline complete ==="
log "Checkpoint: ${MODEL}"
log "Meta: ${MODEL}.meta.json"
log "Log: ${LOG}"
