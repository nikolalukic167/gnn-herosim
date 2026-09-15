#!/bin/bash
# peer_affinity_v1 -- training registration T1 (docs/lineages/peer_affinity_v1.md), MLP arms.
# 6 configs (mlp_t1 / mlp_t1x x lr {5e-4, 1e-3, 2e-3}) x 8 seeds = 48 tasks:
#   task = 8*cfg_index + (seed-1); cfg order: mlp_t1-lr5e4, mlp_t1-lr1e3, mlp_t1-lr2e3, mlp_t1x-lr5e4, mlp_t1x-lr1e3, mlp_t1x-lr2e3
#
#SBATCH --job-name=pa-t1b-mlp
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-t1b-mlp-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-t1b-mlp-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-95

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

T=${SLURM_ARRAY_TASK_ID}
CFGI=$(( T / 16 )); SEED=$(( T % 16 + 1 ))
CFGS=(mlp_t1_lr5e4 mlp_t1_lr1e3 mlp_t1_lr2e3 mlp_t1x_lr5e4 mlp_t1x_lr1e3 mlp_t1x_lr2e3)
NAME=${CFGS[$CFGI]}
ARM=${NAME%_lr*}; LR=lr${NAME##*_lr}
CFG=experiments/peer_affinity_v1_t1b_${NAME}.yaml
# run_experiment.py suffixes the mlp output stem with _seed{N}
CK="models/tabular/peer-affinity-v1-t1b-${ARM}-${LR}_seed${SEED}.pt"
CACHE=simulation_data/graphs_cache_peer_affinity_v1_t1b
SPLIT=experiments/peer_affinity_v1_t1b_split.json
echo "[TASK $T] arm=$ARM lr=$LR seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
mkdir -p logs models/tabular

if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK" >&2; exit 1; }
python3 - "$CK" "$ARM" <<'EOF2'
import sys, torch
pl = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
want_pm = sys.argv[2] == "mlp_t1x"
for k, want in (("partial_state", True), ("partial_state_contract", "partial_state_v2"), ("peer_mass", want_pm), ("inference_feature_layout", "dim63crk")):
    if pl.get(k) != want:
        raise SystemExit(f"FAIL LOUD: payload {k}={pl.get(k)!r}, arm {sys.argv[2]} expects {want!r}")
print("[payload] ok")
EOF2
echo "=== task $T done: $CK ==="
