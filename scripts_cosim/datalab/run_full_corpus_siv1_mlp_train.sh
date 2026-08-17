#!/usr/bin/env bash
# MLP dim22 batchcache train on the full corpus (2,816 ds), scale_invariant_v1 queue features.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_full_corpus_siv1_dim14"
OUT_CKPT="models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt"
PHASE_DIR="logs/full_corpus_siv1_pipeline"
MIN_GRAPHS="${MIN_GRAPHS:-2700}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/full_corpus_siv1_mlp_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models/tabular "$PHASE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== full_corpus siv1 MLP train ${TS} ==="

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing ${CACHE_DIR}/graphs.pkl -- run recache first" >&2; exit 1; }
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

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"

python3 -u src/policy/tabular/train_mlp_dim22_from_batch.py \
  --cache-dir "$CACHE_DIR" \
  --output "$OUT_CKPT" \
  --epochs "${MLP_EPOCHS:-100}" \
  --patience "${MLP_PATIENCE:-10}" \
  --hidden-dim 64 \
  --lr 1e-3 \
  --random-state 42 \
  --test-size 0.2 \
  --wandb-project "${WANDB_PROJECT:-gnn-full-corpus-siv1-aug2026}" \
  --wandb-run-name "batch-edge-mlp-full-corpus-siv1-dim22" \
  --wandb-entity "${WANDB_ENTITY:-nikolalukic167-tu-wien}"

[[ -f "$OUT_CKPT" ]] || { echo "ERROR: MLP checkpoint missing: ${OUT_CKPT}" >&2; exit 1; }

touch "${PHASE_DIR}/phase_train_mlp.done"
echo "MLP checkpoint: ${OUT_CKPT}"
echo "=== MLP train complete === log=${LOG}"
