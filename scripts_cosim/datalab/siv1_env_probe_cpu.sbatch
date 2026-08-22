#!/bin/bash
#SBATCH --job-name=siv1-envprobe-cpu
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/siv1-envprobe-cpu-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/siv1-envprobe-cpu-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# Environment-sensitivity probe (2026-08-21): the recorded siv1 gate ran GNN
# inference on CUDA (see logs/siv1-gate-708549_10.out "on cuda"). This re-runs
# gnn/cell01 CPU-forced with the same thread pinning the local box used.
# Local CPU values for this cell: 50407465.33164156 / 50358532.7523122
# (two runs, 0.097% apart). If this lands there rather than at the recorded
# 65822323.77859402, the datalab-vs-local divergence is CUDA-vs-CPU numerics.
set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn

# REQUIRED. `pipenv run python3` resolves its own venv and shells straight past the
# activation above, so without this every runner script below would silently execute in
# an undeclared venv (torch 2.12.0+cu130) instead of the `gnn` env this job just
# activated. That is what happened to every cluster gate through 2026-08-21; see
# PARITY.md rule 1. The runners read ${HEROSIM_PY:-pipenv run python3}, so locally
# (where this is unset) nothing changes.
export HEROSIM_PY=python3
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONHASHSEED=0
export SWEEP_DIR="simulation_data/normal_sim_sweeps/probe_cpu_forced"
export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh gnn cell01_p25_s9001
