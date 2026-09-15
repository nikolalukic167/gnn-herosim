#!/bin/bash
# peer_affinity_v1 -- training registration T1 (docs/lineages/peer_affinity_v1.md), GNN arms.
# 6 configs (gnn / mpoff x lr {5e-4, 1e-3, 2e-3}) x 8 seeds = 48 tasks:
#   task = 8*cfg_index + (seed-1); cfg order: gnn-lr5e4, gnn-lr1e3, gnn-lr2e3, mpoff-lr5e4, mpoff-lr1e3, mpoff-lr2e3
# Cache and split are built LOCALLY and rsynced (never race on them here). The MLP arms run in
# peer_affinity_v1_t1_mlp_train.sbatch on CPU-amd.
#
#SBATCH --job-name=pa-t1-gnn
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-t1-gnn-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-t1-gnn-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-47

set -euo pipefail
# datalab-pitfalls #11: without this every GNN task dies in the first epochs with
# `RuntimeError: received 0 items of ancdata`.
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

T=${SLURM_ARRAY_TASK_ID}
CFGI=$(( T / 8 )); SEED=$(( T % 8 + 1 ))
CFGS=(gnn_lr5e4 gnn_lr1e3 gnn_lr2e3 mpoff_lr5e4 mpoff_lr1e3 mpoff_lr2e3)
NAME=${CFGS[$CFGI]}
ARM=${NAME%%_*}; LR=${NAME##*_}
CFG=experiments/peer_affinity_v1_t1_${NAME}.yaml
CK="models/peer-affinity-v1-t1-${ARM}-${LR}-seed${SEED}.pt"
CACHE=simulation_data/graphs_cache_peer_affinity_v1_t1
SPLIT=experiments/peer_affinity_v1_t1_split.json
echo "[TASK $T] arm=$ARM lr=$LR seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks valid_combos_near_rtt_capped.pkl" >&2; exit 1; }
[[ -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks rtt_chunk_*.pkl -- NEAR_RTT_VAL_EXACT_REGRET needs the full sweep" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
python3 - "$CACHE" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v2"), ("dag_primary_alpha_key", "2.5"), ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, T1 requires {v!r}")
print("[cache] contract ok")
EOF2
mkdir -p logs models

if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
python3 - "$SC" "$ARM" "$SPLIT" <<'EOF2'
import hashlib, json, sys
sc, arm, split = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
checks = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
          "partial_state_contract": "partial_state_v2", "dag_alpha_key": "2.5",
          "decode_replica_reuse": True, "decode_relax_on_stuck": True, "peer_mass": True}
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
sa = sc.get("split_artifact") or {}
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if sa.get("sha256") != sha:
    raise SystemExit(f"FAIL LOUD: sidecar split sha {sa.get('sha256')} != {split} {sha}")
print("[sidecar] ok")
EOF2
echo "=== task $T done: $CK ==="
