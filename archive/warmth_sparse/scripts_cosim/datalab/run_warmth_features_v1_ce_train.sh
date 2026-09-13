#!/usr/bin/env bash
# CE-only GNN train on graphs_cache_warmth_v2_features_v1 (B1 seq cache, 4144 graphs).
# CE loss uses graph y labels only — placements.jsonl gaps do not block training.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_warmth_v2_features_v1"
OUT_CKPT="models/near-rtt-v2-warmth-features-v1-ce-only.pt"
NB_MODEL="src/notebooks/models/near-rtt-v2-warmth-features-v1-ce-only.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/warmth_features_v1_ce_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== warmth features_v1 CE-only train ${TS} ==="
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

if [[ ! -f "${CACHE_DIR}/graphs.pkl" ]]; then
  echo "ERROR: missing ${CACHE_DIR}/graphs.pkl — run recache first" >&2
  exit 1
fi

n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Training on ${n_graphs} graphs (CE-only, regret_weight=0)"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
export NEAR_RTT_DATALOADER_WORKERS="${NEAR_RTT_DATALOADER_WORKERS:-0}"
unset TRAIN_INIT_CHECKPOINT

cd src/notebooks
python3 train_near_rtt_v2_warmth_features_v1_ce_only.py
cd "$PROJECT_ROOT"

if [[ ! -f "$NB_MODEL" ]]; then
  echo "ERROR: training finished but checkpoint missing: $NB_MODEL" >&2
  exit 1
fi

cp -f "$NB_MODEL" "$OUT_CKPT"
echo "=== Done ==="
echo "Checkpoint: ${OUT_CKPT}"
echo "Log: ${LOG}"
