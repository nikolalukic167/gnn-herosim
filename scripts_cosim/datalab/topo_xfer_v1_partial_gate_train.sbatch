#!/bin/bash
# topology_transfer_v1 partial live gate, training step (LINEAGES 2026-08-21 unblocked entry).
# Same pre-registered split as job 706415; adds --save-checkpoints so arms are deployable.
# Tracked 2026-08-23: datalab is synced post-merge, so the pre-merge pin no longer applies.
#SBATCH --job-name=topo-xfer-ckpt
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/topo-xfer-ckpt-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/topo-xfer-ckpt-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
CKPT_DIR="models/topo_transfer_v1_ckpts"
mkdir -p "$CKPT_DIR" logs
python3 scripts_cosim/gnn_necessity_ablation.py \
  --cache simulation_data/graphs_cache_topo_transfer_v1 \
  --corpus-root simulation_data \
  --split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80 \
  --epochs 120 --seed "$SEED" \
  --models pointwise gnn_base \
  --power-tier tier_launch \
  --save-checkpoints "$CKPT_DIR" \
  --output "simulation_data/topo_transfer_v1_phase4_ckpt_seed${SEED}.json"
ls -la "$CKPT_DIR"/*_seed${SEED}.pt "$CKPT_DIR"/*_seed${SEED}.contract.json
echo "=== seed=${SEED} complete with checkpoints ==="
