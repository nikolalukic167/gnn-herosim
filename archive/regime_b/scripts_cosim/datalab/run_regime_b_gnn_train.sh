#!/usr/bin/env bash
# Regime B: train GNN dim16 CE-only (CACHE 5.6 pull observables) on regime_b cache.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_regime_b_cold_burst_v1"
PHASE_DIR="logs/regime_b_pipeline"
GNN_CKPT="models/near-rtt-v2-regime-b-cold-burst-v1-dim16-ce-only.pt"
MIN_GRAPHS="${MIN_GRAPHS:-440}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/regime_b_gnn_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs "$PHASE_DIR" models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== regime_b GNN dim16 train ${TS} ==="
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

if [[ ! -f "${CACHE_DIR}/graphs.pkl" ]]; then
  echo "ERROR: missing ${CACHE_DIR}/graphs.pkl — run recache first" >&2
  exit 1
fi
n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Graphs: ${n_graphs}"
if [[ "$n_graphs" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: cache too small (${n_graphs} < ${MIN_GRAPHS})" >&2
  exit 1
fi

plat_dim=$(python3 -c "import pickle; g=pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))[0]; print(int(g.platform_features.size(-1)))")
echo "platform_feature_dim=${plat_dim}"
if [[ "$plat_dim" -ne 16 ]]; then
  echo "ERROR: expected CACHE 5.6 platform_feature_dim=16, got ${plat_dim} — FORCE_RECACHE=1" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-100}"
export NEAR_RTT_DATALOADER_WORKERS="${NEAR_RTT_DATALOADER_WORKERS:-0}"

cd src/notebooks
python3 -u train_near_rtt_v2_regime_b_cold_burst_v1_dim16_ce_only.py
cd "$PROJECT_ROOT"

NB_GNN="src/notebooks/models/near-rtt-v2-regime-b-cold-burst-v1-dim16-ce-only.pt"
if [[ -f "$NB_GNN" ]]; then
  cp -f "$NB_GNN" "$GNN_CKPT"
elif [[ -f "$GNN_CKPT" ]]; then
  echo "WARN: notebook ckpt missing; using existing ${GNN_CKPT}"
else
  alt=$(ls -1t src/notebooks/models/*regime-b*dim16*.pt 2>/dev/null | head -1 || true)
  if [[ -n "${alt}" ]]; then
    cp -f "$alt" "$GNN_CKPT"
  else
    echo "ERROR: GNN checkpoint missing after training" >&2
    exit 1
  fi
fi

python3 - <<'PY'
import hashlib
from pathlib import Path
p = Path("models/near-rtt-v2-regime-b-cold-burst-v1-dim16-ce-only.pt")
h = hashlib.md5(p.read_bytes()).hexdigest()
meta = p.with_suffix(p.suffix + ".meta.json")
print(f"GNN ckpt={p} md5={h} size={p.stat().st_size}")
meta.write_text('{"md5":"%s","path":"%s","platform_feature_dim":16}\n' % (h, p.as_posix()))
print(f"Wrote {meta}")
PY

touch "${PHASE_DIR}/phase_gnn_train.done"
echo "=== GNN train complete === ${GNN_CKPT} log=${LOG}"
