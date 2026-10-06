#!/bin/bash
#SBATCH --job-name=workflow-v1
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/workflow-%A_%a.log
set -euo pipefail
ulimit -n 65536
cd "$SLURM_SUBMIT_DIR"
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=offline
export WANDB_DIR="$PWD/wandb" WANDB_CACHE_DIR="$PWD/.wandb-cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
python3 scripts_cosim/workflow_gpu_preflight.py
arms=(gnn mpoff mlp_hand)
arm="${arms[$((SLURM_ARRAY_TASK_ID / 3))]}"
seed=$((101 + SLURM_ARRAY_TASK_ID % 3))
python3 run_experiment.py "experiments/workflow_amortized_v1_${arm}.yaml" --seed "$seed"
