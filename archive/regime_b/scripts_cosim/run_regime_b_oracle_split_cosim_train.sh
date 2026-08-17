#!/usr/bin/env bash
# Local: train GNN dim16 + MLP dim24 on oracle_split_cosim CACHE 5.6.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CACHE_DIR="simulation_data/graphs_cache_regime_b_oracle_split_cosim"
GNN_CKPT="models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt"
MLP_CKPT="models/tabular/batch_edge_mlp_regime_b_oracle_split_cosim_dim24_batchcache.pt"
MIN_GRAPHS="${MIN_GRAPHS:-40}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/regime_b_oracle_split_cosim_train_${TS}.log"
mkdir -p logs models/tabular src/notebooks/models

exec > >(tee -a "$LOG") 2>&1
echo "=== oracle_split_cosim train ${TS} ==="

[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "ERROR: missing cache" >&2; exit 1; }
n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
plat_dim=$(pipenv run python3 -c "import pickle; g=pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))[0]; print(int(g.platform_features.size(-1)))")
echo "graphs=${n_graphs} plat_dim=${plat_dim}"
[[ "$n_graphs" -ge "$MIN_GRAPHS" ]] || { echo "ERROR: too few graphs" >&2; exit 1; }
[[ "$plat_dim" -eq 16 ]] || { echo "ERROR: need plat_dim=16" >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"
export NEAR_RTT_CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-80}"
export NEAR_RTT_DATALOADER_WORKERS="${NEAR_RTT_DATALOADER_WORKERS:-0}"
export MLP_EPOCHS="${MLP_EPOCHS:-80}"
export MLP_PATIENCE="${MLP_PATIENCE:-12}"

echo "=== GNN dim16 ==="
cd src/notebooks
pipenv run python3 -u <<'PY'
import os, sys, runpy
from pathlib import Path
cache = Path(os.environ["NEAR_RTT_CACHE_DIR"])
os.environ.pop("TRAIN_INIT_CHECKPOINT", None)
os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
os.environ.setdefault("NEAR_RTT_MARGIN_MODE", "exp")
os.environ.setdefault("NEAR_RTT_MARGIN_CAP", "8.0")
os.environ.setdefault("NEAR_RTT_MARGIN_EXP_SCALE", "0.75")
os.environ.setdefault("NEAR_RTT_MARGIN_EXP_CLIP", "4.0")
os.environ.setdefault("NEAR_RTT_TRASH_DELTA", "5.0")
os.environ.setdefault("NEAR_RTT_TRASH_WEIGHT", "1.0")
os.environ.setdefault("NEAR_RTT_FAR_WEIGHT", "0.75")
os.environ.setdefault("NEAR_RTT_UNMAPPED_PENALTY", "8.0")
os.environ["WANDB_RUN_NAME"] = "near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only"
os.environ["WANDB_TAGS"] = "near-rtt,ce-only,dim16,pull-obs,regime-b,oracle-split-cosim"
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

NB_GNN="src/notebooks/models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt"
if [[ ! -f "$NB_GNN" ]]; then
  NB_GNN=$(ls -1t src/notebooks/models/*oracle-split-cosim*.pt 2>/dev/null | head -1 || true)
fi
[[ -n "${NB_GNN}" && -f "${NB_GNN}" ]] || { echo "ERROR: GNN ckpt missing after train" >&2; exit 1; }
cp -f "$NB_GNN" "$GNN_CKPT"
pipenv run python3 - <<PY
import hashlib
from pathlib import Path
p = Path("${GNN_CKPT}")
h = hashlib.md5(p.read_bytes()).hexdigest()
p.with_suffix(p.suffix + ".meta.json").write_text(
    '{"md5":"%s","path":"%s","platform_feature_dim":16,"corpus":"oracle_split_cosim"}\n' % (h, p.as_posix())
)
print(f"GNN md5={h} path={p}")
PY

echo "=== MLP dim24 ==="
export MLP_MODEL_PATH="${PROJECT_ROOT}/${MLP_CKPT}"
pipenv run python3 -u <<'PY'
import os, sys, runpy
from pathlib import Path
repo = Path(".").resolve()
cache = Path(os.environ["NEAR_RTT_CACHE_DIR"])
model = Path(os.environ["MLP_MODEL_PATH"])
trainer = repo / "src/policy/tabular/train_mlp_dim22_from_batch.py"
sys.argv = [
    str(trainer), "--cache-dir", str(cache), "--output", str(model),
    "--epochs", os.environ.get("MLP_EPOCHS", "80"),
    "--patience", os.environ.get("MLP_PATIENCE", "12"),
    "--hidden-dim", "64", "--lr", "1e-3", "--random-state", "42",
    "--test-size", "0.2",
]
runpy.run_path(str(trainer), run_name="__main__")
PY
[[ -f "$MLP_CKPT" ]] || { echo "ERROR: MLP missing" >&2; exit 1; }
pipenv run python3 - <<PY
import hashlib
from pathlib import Path
p = Path("${MLP_CKPT}")
h = hashlib.md5(p.read_bytes()).hexdigest()
Path(str(p) + ".meta.json").write_text(
    '{"md5":"%s","path":"%s","input_dim":24,"corpus":"oracle_split_cosim"}\n' % (h, p.as_posix())
)
print(f"MLP md5={h} path={p}")
PY

echo "=== train complete === log=${LOG}"
