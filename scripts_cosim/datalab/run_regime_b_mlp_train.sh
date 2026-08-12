#!/usr/bin/env bash
# Regime B: train dim24 MLP from CACHE 5.6 batch cache (pull observables).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_regime_b_cold_burst_v1"
PHASE_DIR="logs/regime_b_pipeline"
MLP_CKPT="models/tabular/batch_edge_mlp_regime_b_cold_burst_v1_dim24_batchcache.pt"
MIN_GRAPHS="${MIN_GRAPHS:-440}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/regime_b_mlp_train_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs "$PHASE_DIR" models/tabular

exec > >(tee -a "$LOG") 2>&1
echo "=== regime_b MLP dim24 train ${TS} ==="
echo "Host: $(hostname)"

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
export MLP_MODEL_PATH="${PROJECT_ROOT}/${MLP_CKPT}"
export MLP_EPOCHS="${MLP_EPOCHS:-100}"
export WANDB_MODE="${WANDB_MODE:-offline}"

python3 -u src/notebooks/train_mlp_regime_b_cold_burst_v1_dim24_batchcache.py

[[ -f "$MLP_CKPT" ]] || { echo "ERROR: MLP ckpt missing: ${MLP_CKPT}" >&2; exit 1; }

python3 - <<'PY'
import hashlib
from pathlib import Path
p = Path("models/tabular/batch_edge_mlp_regime_b_cold_burst_v1_dim24_batchcache.pt")
h = hashlib.md5(p.read_bytes()).hexdigest()
meta = Path(str(p) + ".meta.json")
print(f"MLP ckpt={p} md5={h} size={p.stat().st_size}")
meta.write_text('{"md5":"%s","path":"%s","input_dim":24}\n' % (h, p.as_posix()))
print(f"Wrote {meta}")
PY

touch "${PHASE_DIR}/phase_mlp_train.done"
echo "=== MLP train complete === ${MLP_CKPT} log=${LOG}"
