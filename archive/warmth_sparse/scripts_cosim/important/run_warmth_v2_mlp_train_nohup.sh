#!/usr/bin/env bash
# warmth_v2 Regime A MLP: seq cache -> parquet -> grouped CE train (100ep).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_mlp_warmth_v2_${TS}.log"
BASE_DIRS="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SEQ_CACHE="simulation_data/graphs_cache_gnn_datasets_4tasks_1060_warmth_v2_seq"
PARQUET="simulation_data/artifacts/tabular/batch_edges_warmth_v2.parquet"
MODEL="models/tabular/batch_edge_mlp_warmth_v2.pt"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

mkdir -p logs simulation_data/artifacts/tabular models/tabular

log "=== warmth_v2 MLP pipeline start ==="
log "Base datasets: ${BASE_DIRS}"
log "Seq cache: ${SEQ_CACHE}"
log "Parquet: ${PARQUET}"
log "Model: ${MODEL}"

log "Phase 1: sequential graph cache"
if [[ ! -f "${SEQ_CACHE}/graphs.pkl" ]]; then
  pipenv run python3 -u src/notebooks/prepare_graphs_cache_seq.py \
    --base-dirs "${BASE_DIRS}" \
    --cache-dir "${SEQ_CACHE}" \
    2>&1 | tee -a "$LOG"
else
  log "SKIP seq cache build (graphs.pkl exists)"
fi

log "Phase 2: tabular parquet extraction"
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

log "=== warmth_v2 MLP pipeline complete ==="
log "Checkpoint: ${MODEL}"
log "Meta: ${MODEL}.meta.json"
