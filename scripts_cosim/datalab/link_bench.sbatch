#!/bin/bash
#SBATCH --job-name=link-bench
#SBATCH --output=/share/nikola.lukic/uncut_groups/flowcap/link_corpus/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
# live_link_features_v1 variant (b): training-loss timing, as-built vs re-encode-per-placement, CPU or GPU. sbatch --export=ALL,WT=<worktree>,DEVICE=cpu|cuda [--partition=GPU-a40 --gres=gpu:1] ...
set -euo pipefail
WT=${WT:?}; DEVICE=${DEVICE:?}; SIZES=${SIZES:-"12 24 60 120 240"}
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
echo "HEAD $(git rev-parse HEAD) node $(hostname) $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
python3 scripts_cosim/bench_link_conv.py --device "$DEVICE" --sizes $SIZES
