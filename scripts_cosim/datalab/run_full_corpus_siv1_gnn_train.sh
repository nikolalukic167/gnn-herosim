#!/usr/bin/env bash
# GNN dim14 CE-only train on the full corpus (2,816 ds), scale_invariant_v1 queue features.
# Same loss/margin hyperparams as the working near-rtt-v2-contention-v2-dim14-ce-only recipe.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

# train_near_rtt.py saves to models/{wandb.run.name}.pt (train_near_rtt.py:1141), so the
# checkpoint name IS the wandb run name -- they cannot be set independently, and OUT_CKPT is
# derived from WANDB_RUN_NAME below rather than repeated. Both are overridable so a retrain on
# a corrected cache lands beside the deployed checkpoint instead of on top of it.
CACHE_DIR="${CACHE_DIR:-simulation_data/graphs_cache_full_corpus_siv1_dim14}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-near-rtt-v2-full-corpus-siv1-dim14-ce-only}"
OUT_CKPT="models/${WANDB_RUN_NAME}.pt"
PHASE_DIR="logs/full_corpus_siv1_pipeline"
MIN_GRAPHS="${MIN_GRAPHS:-2700}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/full_corpus_siv1_gnn_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models "$PHASE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== full_corpus siv1 GNN train ${TS} ==="

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing ${CACHE_DIR}/graphs.pkl -- run recache first" >&2; exit 1; }
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

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export WANDB_MODE="${WANDB_MODE:-online}"
export NEAR_RTT_LOSS_VARIANT="${NEAR_RTT_LOSS_VARIANT:-near-rtt-v2-trash-exp}"
export NEAR_RTT_SIDECAR_NAME="${NEAR_RTT_SIDECAR_NAME:-valid_combos_near_rtt_v2_capped.pkl}"
export NEAR_RTT_MARGIN_MODE="${NEAR_RTT_MARGIN_MODE:-exp}"
export NEAR_RTT_MARGIN_CAP="${NEAR_RTT_MARGIN_CAP:-8.0}"
export NEAR_RTT_MARGIN_EXP_SCALE="${NEAR_RTT_MARGIN_EXP_SCALE:-0.75}"
export NEAR_RTT_MARGIN_EXP_CLIP="${NEAR_RTT_MARGIN_EXP_CLIP:-4.0}"
export NEAR_RTT_TRASH_DELTA="${NEAR_RTT_TRASH_DELTA:-5.0}"
export NEAR_RTT_TRASH_WEIGHT="${NEAR_RTT_TRASH_WEIGHT:-1.0}"
export NEAR_RTT_FAR_WEIGHT="${NEAR_RTT_FAR_WEIGHT:-0.75}"
export NEAR_RTT_UNMAPPED_PENALTY="${NEAR_RTT_UNMAPPED_PENALTY:-8.0}"
export WANDB_TAGS="near-rtt,ce-only,dim14,full-corpus,scale-invariant-v1,from-scratch"
unset TRAIN_INIT_CHECKPOINT

python3 -u src/notebooks/train_near_rtt.py \
  --cache-dir "$CACHE_DIR" \
  --regret-loss-weight 0 \
  --ce-loss-weight 1 \
  --epochs "${NEAR_RTT_TRAIN_EPOCHS:-100}" \
  --wandb-project "${WANDB_PROJECT:-gnn-full-corpus-siv1-aug2026}"

[[ -f "$OUT_CKPT" ]] || { echo "ERROR: GNN checkpoint missing after train: ${OUT_CKPT}" >&2; exit 1; }

touch "${PHASE_DIR}/phase_train_gnn.done"
echo "GNN checkpoint: ${OUT_CKPT}"
echo "=== GNN train complete === log=${LOG}"
