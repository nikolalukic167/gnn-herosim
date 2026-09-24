#!/bin/bash
# Matched MLP retrain on the corrected (_tempfix) cache — fair-comparison arm for the
# GNN corrected-cache retrain (LINEAGES siv1_full_corpus). Mirrors the deployed MLP
# training exactly (run_full_corpus_siv1_mlp_train.sh) except cache/output/run-name.
# Tracked 2026-08-23: datalab is synced post-merge, so the pre-merge pin no longer applies.
#SBATCH --job-name=fc-siv1-mlp-tfix
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/fc-siv1-mlp-tempfix-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/fc-siv1-mlp-tempfix-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p logs models/tabular
python3 -u src/policy/tabular/train_mlp_dim22_from_batch.py \
  --cache-dir simulation_data/graphs_cache_full_corpus_siv1_dim14_tempfix \
  --output models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt \
  --epochs 100 --patience 10 --hidden-dim 64 --lr 1e-3 --random-state 42 --test-size 0.2 \
  --wandb-project gnn-full-corpus-siv1-aug2026 \
  --wandb-run-name batch-edge-mlp-full-corpus-siv1-dim22-tempfix \
  --wandb-entity nikolalukic167-tu-wien
ls -la models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt*
echo "=== MLP tempfix train complete ==="
