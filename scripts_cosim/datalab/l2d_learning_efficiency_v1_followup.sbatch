#!/bin/bash
# Post-hoc midpoint gate, evaluated on fresh seed-303 instances.
# GNN: 2000 PPO updates; MP-OFF and typed MLP: existing 3000-update checkpoints.
#SBATCH --job-name=l2d-eff-v1b
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/l2d-eff-v1b-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/l2d-eff-v1b-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-47%24

set -euo pipefail
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export L2D_ROOT="${HOME}/L2D"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

ARMS=(gnn mpoff mlp_t1)
I="$SLURM_ARRAY_TASK_ID"
ARM="${ARMS[$(( I % 3 ))]}"
SEED=$(( (I / 3) % 8 ))
if [ "$I" -lt 24 ]; then
  N=6
  LR=0.0005
else
  N=10
  if [ "$ARM" = mlp_t1 ]; then LR=0.0001; else LR=0.0005; fi
fi
BASE=simulation_data/l2d_learning_efficiency_v1
if [ "$ARM" = gnn ]; then
  OUT="$BASE/followup_runs"
  NAME="l2d_${N}x${N}_${ARM}_lr${LR}_up2000_s${SEED}"
  RUN_DIR="$OUT/$NAME"
  mkdir -p "$OUT"
  if [ ! -f "$RUN_DIR/final.pth" ]; then
    python3 scripts_reeval/l2d/train_arm.py --arm "$ARM" --n_j "$N" --n_m "$N" \
      --seed "$SEED" --lr "$LR" --max_updates 2000 --save_updates 2000 \
      --out "$OUT" --threads 2 --wandb_project literature-reeval-l2d
  fi
  UPDATE=2000
else
  OUT="$BASE/runs"
  NAME="l2d_${N}x${N}_${ARM}_lr${LR}_up3000_s${SEED}"
  RUN_DIR="$OUT/$NAME"
  UPDATE=3000
  test -f "$RUN_DIR/update_3000.pth"
fi
python3 scripts_reeval/l2d/evaluate_fixed_budget.py --run_dir "$RUN_DIR" \
  --test_seed 303 --n_test 100 --updates "$UPDATE" --threads 2
