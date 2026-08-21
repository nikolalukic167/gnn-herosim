#!/bin/bash
#SBATCH --job-name=siv1-envprobe-gpu
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/siv1-envprobe-gpu-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/siv1-envprobe-gpu-%A_%a.err
#SBATCH --partition=GPU-a40
#SBATCH --nodelist=avm-a40-o-5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1

# Environment-sensitivity probe (2026-08-21): re-run the recorded siv1 gate's
# gnn/cell01 on the SAME node (avm-a40-o-5) twice. Recorded value (job 708549_10):
# total_rtt = 65822323.77859402. If the two runs here disagree with each other,
# GPU inference is not run-to-run reproducible (CUDA scatter-add atomics in GIN).
set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export PYTHONHASHSEED=0
export SWEEP_DIR="simulation_data/normal_sim_sweeps/probe_gpu_repro_${SLURM_ARRAY_TASK_ID}"
export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh gnn cell01_p25_s9001
