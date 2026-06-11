#!/usr/bin/env bash
# warmth_v2 CE-reduced MLP: slice seq cache in-process -> grouped CE (100ep).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_mlp_warmth_ce_reduced_${TS}.log"
SEQ_CACHE="simulation_data/graphs_cache_gnn_datasets_4tasks_1060_warmth_v2_seq"
MODEL="models/tabular/batch_edge_mlp_warmth_ce_reduced.pt"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

mkdir -p logs models/tabular

log "=== warmth_v2 MLP CE-reduced pipeline start ==="
log "Seq cache: ${SEQ_CACHE} (slice in-process, no regen)"
log "Layout: task=3 platform=6 edge=2 (11-d)"
log "Model: ${MODEL}"

if [[ ! -f "${SEQ_CACHE}/graphs.pkl" ]]; then
  log "ERROR: seq cache missing — run run_warmth_v2_mlp_train_nohup.sh first"
  exit 1
fi

pipenv run python3 -u src/notebooks/train_mlp_v2_warmth_ce_reduced.py 2>&1 | tee -a "$LOG"

log "=== warmth_v2 MLP CE-reduced pipeline complete ==="
log "Checkpoint: ${MODEL}"
log "Meta: ${MODEL}.meta.json"
