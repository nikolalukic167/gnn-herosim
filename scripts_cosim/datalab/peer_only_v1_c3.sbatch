#!/bin/bash
# peer_only_v1 C3 (AMENDMENT 8) -- the over-smoothing probe. CPU, single task, minutes.
#SBATCH --job-name=po-c3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/po-c3-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/po-c3-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export HEROSIM_GNN_DEVICE=cpu
export PYTHONHASHSEED=0
# These checkpoints are prefix-conditioned (partial_state_edge_features), so
# load_gnn_model refuses any decode mode but masked_topo -- correctly: any other would
# score against an all-zero prefix block. C3 never decodes (it probes _encode, where the
# partial-state columns provably do not enter), but the loader guard is a LOADER guard
# and it is right to fire before it knows that. Set what the gate sets.
export GNN_DECODE_MODE=masked_topo
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

python3 scripts_cosim/peer_only_v1_oversmoothing.py \
  --cache simulation_data/graphs_cache_peer_only_v1_1670_psv3 \
  --models models --tag peer-only-v1-1670-gnn-lr2e3 --seeds 1-16 \
  --out simulation_data/peer_affinity_live_gate/results/peer_only_v1_c3.json
