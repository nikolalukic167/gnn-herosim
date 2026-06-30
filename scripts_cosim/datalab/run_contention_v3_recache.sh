#!/usr/bin/env bash
# Build graphs_cache_contention_v3 from 900-ds contention_v3 corpus on datalab.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CORPUS_DIR="simulation_data/gnn_datasets_4tasks_contention_v3"
CACHE_DIR="simulation_data/graphs_cache_contention_v3"
PHASE_DIR="logs/contention_v3_pipeline"
MIN_JSONL="${MIN_JSONL:-850}"
MIN_GRAPHS="${MIN_GRAPHS:-850}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/contention_v3_recache_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs "$PHASE_DIR" "$CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== contention_v3 recache ${TS} ==="
echo "Host: $(hostname)"
echo "Corpus: ${CORPUS_DIR}"
echo "Cache: ${CACHE_DIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

jsonl_count=$(find "$CORPUS_DIR" -path '*/placements/placements.jsonl' -size +0 2>/dev/null | wc -l)
best_count=$(find "$CORPUS_DIR" -name best.json 2>/dev/null | wc -l)
echo "Corpus: jsonl=${jsonl_count} best.json=${best_count}"
if [[ "$jsonl_count" -lt "$MIN_JSONL" ]]; then
  echo "ERROR: expected >= ${MIN_JSONL} placements.jsonl, got ${jsonl_count}" >&2
  exit 1
fi

if [[ -f "${CACHE_DIR}/graphs.pkl" && "${FORCE_RECACHE:-0}" != "1" ]]; then
  n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
  if [[ "$n_graphs" -ge "$MIN_GRAPHS" ]]; then
    echo "SKIP recache: ${CACHE_DIR} already has ${n_graphs} graphs"
    touch "${PHASE_DIR}/phase_recache.done"
    exit 0
  fi
  echo "WARN: cache exists but only ${n_graphs} graphs (< ${MIN_GRAPHS}); rebuilding"
fi

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
echo "=== refresh_optimal_full_stats (--rewrite-ssc) ==="
python3 -u scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir "$CORPUS_DIR" \
  --rewrite-ssc

ssc_count=$(find "$CORPUS_DIR" -name system_state_captured_unique.json -size +0 2>/dev/null | wc -l)
echo "SSC files: ${ssc_count}/${jsonl_count}"
if [[ "$ssc_count" -lt "$MIN_JSONL" ]]; then
  echo "ERROR: SSC refresh incomplete (${ssc_count} < ${MIN_JSONL})" >&2
  exit 1
fi

echo "=== prepare_graphs_cache (contention_v3, ${jsonl_count} ds) ==="
python3 -u src/notebooks/prepare_graphs_cache.py \
  --base-dirs "$CORPUS_DIR" \
  --cache-dir "$CACHE_DIR"

n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Graphs in cache: ${n_graphs}"
if [[ "$n_graphs" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: cache too small (${n_graphs} < ${MIN_GRAPHS})" >&2
  exit 1
fi

touch "${PHASE_DIR}/phase_recache.done"
echo "=== recache complete === log=${LOG}"
