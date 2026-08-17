#!/usr/bin/env bash
# Phase 2: capacity-matched soft_combo_conc train on oracle_split_cosim CACHE 5.6 / dim16.
# Same arch + cache as CE dim16; objective = soft_combo + concentration (τ=0.25, γ=0.02).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_regime_b_oracle_split_cosim"
GNN_CKPT="models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-soft-combo-conc.pt"
MIN_GRAPHS="${MIN_GRAPHS:-40}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/regime_b_oracle_split_soft_combo_conc_train_${TS}.log"
mkdir -p logs models src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== oracle_split soft_combo_conc train ${TS} ==="

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing cache" >&2; exit 1; }
[[ -f "${CACHE_DIR}/valid_combos_near_rtt_v2_capped.pkl" ]] || {
  echo "ERROR: missing valid_combos sidecar (required for soft_combo)" >&2
  exit 1
}

n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
plat_dim=$(pipenv run python3 -c "import pickle; g=pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))[0]; print(int(g.platform_features.size(-1)))")
has_qmeta=$(pipenv run python3 -c "import pickle; g=pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))[0]; print(int(isinstance(getattr(g,'queue_key_to_platform_meta',None), dict) and len(g.queue_key_to_platform_meta)>0))")
echo "graphs=${n_graphs} plat_dim=${plat_dim} queue_key_meta=${has_qmeta}"
[[ "$n_graphs" -ge "$MIN_GRAPHS" ]] || { echo "ERROR: too few graphs" >&2; exit 1; }
[[ "$plat_dim" -eq 16 ]] || { echo "ERROR: need plat_dim=16" >&2; exit 1; }
[[ "$has_qmeta" -eq 1 ]] || { echo "ERROR: queue_key_to_platform_meta missing (soft_combo_conc)" >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-80}"
export NEAR_RTT_DATALOADER_WORKERS="${NEAR_RTT_DATALOADER_WORKERS:-0}"

echo "=== GNN dim16 soft_combo_conc ==="
cd src/notebooks
pipenv run python3 -u <<'PY'
import os, sys, runpy
from pathlib import Path
cache = Path(os.environ["NEAR_RTT_CACHE_DIR"])
os.environ.pop("TRAIN_INIT_CHECKPOINT", None)
os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
os.environ.setdefault("NEAR_RTT_TRAIN_OBJECTIVE", "soft_combo_conc")
os.environ.setdefault("NEAR_RTT_SOFT_COMBO_TAU", "0.25")
os.environ.setdefault("NEAR_RTT_SOFT_COMBO_MAX_COMBOS", "4096")
os.environ.setdefault("NEAR_RTT_CONC_GAMMA", "0.02")
os.environ.setdefault("NEAR_RTT_CONC_CAP", "1.5")
os.environ.setdefault("NEAR_RTT_MARGIN_MODE", "exp")
os.environ.setdefault("NEAR_RTT_MARGIN_CAP", "8.0")
os.environ.setdefault("NEAR_RTT_MARGIN_EXP_SCALE", "0.75")
os.environ.setdefault("NEAR_RTT_MARGIN_EXP_CLIP", "4.0")
os.environ.setdefault("NEAR_RTT_TRASH_DELTA", "5.0")
os.environ.setdefault("NEAR_RTT_TRASH_WEIGHT", "1.0")
os.environ.setdefault("NEAR_RTT_FAR_WEIGHT", "0.75")
os.environ.setdefault("NEAR_RTT_UNMAPPED_PENALTY", "8.0")
os.environ["WANDB_RUN_NAME"] = "near-rtt-v2-regime-b-oracle-split-cosim-dim16-soft-combo-conc"
os.environ["WANDB_TAGS"] = (
    "near-rtt,soft-combo-conc,dim16,pull-obs,regime-b,oracle-split-cosim,phase2"
)
trainer = Path("train_near_rtt.py")
sys.argv = [
    str(trainer), "--cache-dir", str(cache),
    "--regret-loss-weight", "0", "--ce-loss-weight", "1",
    "--epochs", os.environ.get("NEAR_RTT_TRAIN_EPOCHS", "80"),
    "--wandb-project", os.environ.get("WANDB_PROJECT", "gnn-near-rtt-regime-b-aug2026"),
]
runpy.run_path(str(trainer), run_name="__main__")
PY
cd "$PROJECT_ROOT"

NB_GNN="src/notebooks/models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-soft-combo-conc.pt"
if [[ ! -f "$NB_GNN" ]]; then
  NB_GNN=$(ls -1t src/notebooks/models/*oracle-split-cosim*soft-combo-conc*.pt 2>/dev/null | head -1 || true)
fi
[[ -n "${NB_GNN}" && -f "${NB_GNN}" ]] || { echo "ERROR: GNN ckpt missing after train" >&2; exit 1; }
cp -f "$NB_GNN" "$GNN_CKPT"
pipenv run python3 - <<PY
import hashlib, json
from pathlib import Path
p = Path("${GNN_CKPT}")
h = hashlib.md5(p.read_bytes()).hexdigest()
meta = {
    "md5": h,
    "path": p.as_posix(),
    "platform_feature_dim": 16,
    "corpus": "oracle_split_cosim",
    "train_objective": "soft_combo_conc",
    "soft_combo_tau": 0.25,
    "conc_gamma": 0.02,
    "phase": "phase2",
}
p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps(meta) + "\n")
print(f"GNN md5={h} path={p}")
PY

echo "=== soft_combo_conc train complete === log=${LOG} ckpt=${GNN_CKPT}"
