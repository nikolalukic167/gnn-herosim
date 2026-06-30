#!/usr/bin/env bash
# MLP dim22 batchcache train from contention_v3 cache.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_contention_v3"
OUT_CKPT="models/tabular/batch_edge_mlp_contention_v3_dim22_batchcache.pt"
PHASE_DIR="logs/contention_v3_pipeline"
MIN_GRAPHS="${MIN_GRAPHS:-850}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/contention_v3_mlp_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models/tabular "$PHASE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== contention_v3 MLP train ${TS} ==="

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing ${CACHE_DIR}/graphs.pkl" >&2; exit 1; }
n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
if [[ "$n_graphs" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: cache too small (${n_graphs})" >&2
  exit 1
fi

if [[ -f "$OUT_CKPT" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  echo "SKIP MLP train: ${OUT_CKPT} exists"
  touch "${PHASE_DIR}/phase_train_mlp.done"
  exit 0
fi

export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
export MLP_MODEL_PATH="${PROJECT_ROOT}/${OUT_CKPT}"
export MLP_EPOCHS="${MLP_EPOCHS:-100}"
export MLP_PATIENCE="${MLP_PATIENCE:-10}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"

python3 -u src/notebooks/train_mlp_contention_v3_dim22_batchcache.py

[[ -f "$OUT_CKPT" ]] || { echo "ERROR: MLP checkpoint missing: ${OUT_CKPT}" >&2; exit 1; }

touch "${PHASE_DIR}/phase_train_mlp.done"
echo "MLP checkpoint: ${OUT_CKPT}"
echo "=== MLP train complete === log=${LOG}"
