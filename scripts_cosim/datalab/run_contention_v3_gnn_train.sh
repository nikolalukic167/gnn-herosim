#!/usr/bin/env bash
# GNN dim14 CE-only train from contention_v3 cache.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_contention_v3"
OUT_CKPT="models/near-rtt-v2-contention-v3-dim14-ce-only.pt"
NB_CKPT="src/notebooks/models/near-rtt-v2-contention-v3-dim14-ce-only.pt"
PHASE_DIR="logs/contention_v3_pipeline"
MIN_GRAPHS="${MIN_GRAPHS:-850}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/contention_v3_gnn_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models models/tabular src/notebooks/models "$PHASE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== contention_v3 GNN train ${TS} ==="

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing ${CACHE_DIR}/graphs.pkl" >&2; exit 1; }
n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Cache graphs: ${n_graphs}"
if [[ "$n_graphs" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: cache too small (${n_graphs})" >&2
  exit 1
fi

if [[ -f "$OUT_CKPT" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  echo "SKIP GNN train: ${OUT_CKPT} exists"
  touch "${PHASE_DIR}/phase_train_gnn.done"
  exit 0
fi

export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
export NEAR_RTT_DATALOADER_WORKERS="${NEAR_RTT_DATALOADER_WORKERS:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export WANDB_MODE="${WANDB_MODE:-online}"
export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-100}"
unset TRAIN_INIT_CHECKPOINT

cd src/notebooks
python3 -u train_near_rtt_v2_contention_v3_dim14_ce_only.py
cd "$PROJECT_ROOT"

if [[ -f "$NB_CKPT" ]]; then
  cp -f "$NB_CKPT" "$OUT_CKPT"
elif [[ -f "$OUT_CKPT" ]]; then
  :
else
  echo "ERROR: GNN checkpoint missing after train" >&2
  exit 1
fi

touch "${PHASE_DIR}/phase_train_gnn.done"
echo "GNN checkpoint: ${OUT_CKPT}"
echo "=== GNN train complete === log=${LOG}"
