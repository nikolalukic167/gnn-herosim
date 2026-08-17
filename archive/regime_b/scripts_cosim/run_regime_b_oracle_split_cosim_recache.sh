#!/usr/bin/env bash
# Local: CACHE 5.6 recache for oracle_split_cosim corpus.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CORPUS_DIR="simulation_data/gnn_datasets_4tasks_regime_b_cold_burst_v1_oracle_split_cosim"
CACHE_DIR="simulation_data/graphs_cache_regime_b_oracle_split_cosim"
MIN_JSONL="${MIN_JSONL:-40}"
MIN_GRAPHS="${MIN_GRAPHS:-40}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/regime_b_oracle_split_cosim_recache_${TS}.log"
mkdir -p logs "$CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== oracle_split_cosim recache ${TS} ==="

jsonl_count=$(find "$CORPUS_DIR" -path '*/placements/placements.jsonl' -size +0 2>/dev/null | wc -l)
echo "Corpus: jsonl=${jsonl_count}"
if [[ "$jsonl_count" -lt "$MIN_JSONL" ]]; then
  echo "ERROR: expected >= ${MIN_JSONL} placements.jsonl, got ${jsonl_count}" >&2
  exit 1
fi

if [[ -f "${CACHE_DIR}/graphs.pkl" && "${FORCE_RECACHE:-0}" != "1" ]]; then
  n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
  if [[ "$n_graphs" -ge "$MIN_GRAPHS" ]]; then
    echo "SKIP recache: ${CACHE_DIR} already has ${n_graphs} graphs"
    exit 0
  fi
fi

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
echo "=== refresh_optimal_full_stats (--rewrite-ssc) ==="
pipenv run python3 -u scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir "$CORPUS_DIR" \
  --rewrite-ssc

ssc_count=$(find "$CORPUS_DIR" -name system_state_captured_unique.json -size +0 2>/dev/null | wc -l)
echo "SSC files: ${ssc_count}/${jsonl_count}"
if [[ "$ssc_count" -lt "$MIN_JSONL" ]]; then
  echo "ERROR: SSC incomplete (${ssc_count} < ${MIN_JSONL})" >&2
  exit 1
fi

echo "=== prepare_graphs_cache ==="
pipenv run python3 -u src/notebooks/prepare_graphs_cache.py \
  --base-dirs "$CORPUS_DIR" \
  --cache-dir "$CACHE_DIR"

n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
plat_dim=$(pipenv run python3 -c "import pickle; g=pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))[0]; print(int(g.platform_features.size(-1)))")
cache_ver=$(pipenv run python3 -c "import json; print(json.load(open('${CACHE_DIR}/metadata.json'))['version'])")
echo "graphs=${n_graphs} plat_dim=${plat_dim} version=${cache_ver}"
if [[ "$n_graphs" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: cache too small (${n_graphs} < ${MIN_GRAPHS})" >&2
  exit 1
fi
if [[ "$plat_dim" -ne 16 ]]; then
  echo "ERROR: CACHE 5.6 requires plat_dim=16, got ${plat_dim}" >&2
  exit 1
fi
if [[ "$cache_ver" != "5.6" ]]; then
  echo "ERROR: expected cache version 5.6, got ${cache_ver}" >&2
  exit 1
fi

pipenv run python3 -u scripts_cosim/validate_training_cache_contract.py \
  --cache-dir "$CACHE_DIR"

echo "=== recache complete === log=${LOG}"
