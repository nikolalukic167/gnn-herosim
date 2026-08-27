#!/usr/bin/env bash
# Build graphs_cache_warmth_v2_features_v1 from repaired warmth+sparse co-sim (B1 seq features).
#
# Sanity: rtt_parent_dataset_ids.txt count vs datasets with placements/placements.jsonl.
# Parents without JSONL get graphs but no counterfactual RTT rows — repair does not fix that.
# docs/notes/placements_jsonl_required.md
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
CACHE_DIR="simulation_data/graphs_cache_warmth_v2_features_v1"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/warmth_features_recache_${TS}.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"

mkdir -p logs "simulation_data"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

python3 -c 'import importlib.util, sys; req=["torch","torch_geometric","pandas"]; miss=[x for x in req if importlib.util.find_spec(x) is None]; sys.exit(1 if miss else 0)' \
  || { echo "ERROR: env ${ENV_NAME} missing torch/torch_geometric/pandas" >&2; exit 1; }

sparse_opt=$(find "$SPARSE_DIR" -name optimal_result.json 2>/dev/null | wc -l)
warmth_opt=$(find "$WARMTH_DIR" -name optimal_result.json 2>/dev/null | wc -l)
echo "=== warmth_features_recache ${TS} ==="
echo "Host: $(hostname)"
echo "Warmth optimal_result: ${warmth_opt}"
echo "Sparse optimal_result: ${sparse_opt}"
echo "Output cache: ${CACHE_DIR}"
echo "Log: ${LOG}"

if [[ "$sparse_opt" -lt 351 ]]; then
  echo "ERROR: sparse has ${sparse_opt}/351 optimal_result.json" >&2
  exit 1
fi

if [[ "${FORCE_RECACHE:-0}" == "1" && -d "$CACHE_DIR" ]]; then
  echo "FORCE_RECACHE=1: removing ${CACHE_DIR}"
  rm -rf "$CACHE_DIR"
fi

exec > >(tee -a "$LOG") 2>&1

echo "=== prepare_graphs_cache_seq (B1: src_norm, dim13 disk, is_warm) ==="
python3 -u src/notebooks/prepare_graphs_cache_seq.py \
  --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" \
  --cache-dir "$CACHE_DIR"

if [[ ! -f "${CACHE_DIR}/graphs.pkl" ]]; then
  echo "ERROR: recache finished but ${CACHE_DIR}/graphs.pkl missing" >&2
  exit 1
fi

n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "=== Recache complete: ${n_graphs} graphs in ${CACHE_DIR} ==="
ls -lh "${CACHE_DIR}/graphs.pkl" "${CACHE_DIR}/metadata.json" 2>/dev/null || true
