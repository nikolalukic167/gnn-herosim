#!/usr/bin/env bash
# Message-passing restoration arms on the full corpus (see LINEAGES.md: siv1_full_corpus).
#
# Same corpus, loss and margin hyperparameters as run_full_corpus_siv1_gnn_train.sh -- the
# ONLY differences are the NEAR_RTT_MP_* variables, so a delta is attributable to message
# passing rather than to the recipe.
#
#   ARM=residual              gated GIN residual, bipartite message passing only
#   ARM=residual_node_edges   residual + candidate-restricted same-node edges
#
# Baseline for both: near-rtt-v2-full-corpus-siv1-dim14-ce-only (0.88x Knative live,
# still losing to MLP at 0.78x).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ARM="${ARM:?set ARM=residual or ARM=residual_node_edges}"
case "$ARM" in
  residual)
    MP_RESIDUAL=1; MP_NODE_EDGES=0
    SUFFIX="mp-residual"
    ;;
  residual_node_edges)
    MP_RESIDUAL=1; MP_NODE_EDGES=1
    SUFFIX="mp-residual-node-edges"
    ;;
  *)
    echo "ERROR: unknown ARM=${ARM} (use residual | residual_node_edges)" >&2; exit 1 ;;
esac

CACHE_DIR="${CACHE_DIR:-simulation_data/graphs_cache_full_corpus_siv1_dim14}"
RUN_NAME="near-rtt-v2-full-corpus-siv1-dim14-ce-only-${SUFFIX}"
OUT_CKPT="models/${RUN_NAME}.pt"
MIN_GRAPHS="${MIN_GRAPHS:-2600}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/mp_arm_${ARM}_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs models

exec > >(tee -a "$LOG") 2>&1
echo "=== mp arm '${ARM}' train ${TS} ==="

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing ${CACHE_DIR}/graphs.pkl" >&2; exit 1; }
n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Cache graphs: ${n_graphs}"
[[ "$n_graphs" -ge "$MIN_GRAPHS" ]] || { echo "ERROR: cache too small (${n_graphs} < ${MIN_GRAPHS})" >&2; exit 1; }

if [[ -f "$OUT_CKPT" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  echo "SKIP: ${OUT_CKPT} exists (FORCE_RETRAIN=1 to override)"
  exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export WANDB_MODE="${WANDB_MODE:-online}"
# --- shared recipe, identical to the 0.88x baseline run ---
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
# --- the variables under test ---
export NEAR_RTT_MP_RESIDUAL="${MP_RESIDUAL}"
export NEAR_RTT_MP_NODE_EDGES="${MP_NODE_EDGES}"
export NEAR_RTT_MP_NODE_EDGES_CANDIDATES_ONLY="${NEAR_RTT_MP_NODE_EDGES_CANDIDATES_ONLY:-1}"

export WANDB_RUN_NAME="${RUN_NAME}"
export WANDB_TAGS="near-rtt,ce-only,dim14,full-corpus,scale-invariant-v1,from-scratch,${SUFFIX}"
unset TRAIN_INIT_CHECKPOINT

python3 -u src/notebooks/train_near_rtt.py \
  --cache-dir "$CACHE_DIR" \
  --regret-loss-weight 0 \
  --ce-loss-weight 1 \
  --epochs "${NEAR_RTT_TRAIN_EPOCHS:-100}" \
  --wandb-project "${WANDB_PROJECT:-gnn-full-corpus-siv1-aug2026}"

[[ -f "$OUT_CKPT" ]] || { echo "ERROR: checkpoint missing after train: ${OUT_CKPT}" >&2; exit 1; }
[[ -f "models/${RUN_NAME}.contract.json" ]] || { echo "ERROR: contract sidecar missing -- serving cannot recover mp_node_edges" >&2; exit 1; }

echo "checkpoint: ${OUT_CKPT}"
echo "=== mp arm '${ARM}' complete === log=${LOG}"
