#!/bin/bash
# objective_pivot_v1 Phase 3, Amendment E — the powered CL-GNN sample.
#
# 120 FRESH training seeds, 17..136. The 16 seeds of the original pilot are deliberately
# NOT reused: the decision to extend was made after looking at their gate, and pooling
# them into the primary would inflate Type I error by exactly what that look was worth.
# They are reported as a pooled secondary only.
#
# Hyperparameters carry over UNCHANGED from the Amendment D3 selection -- lr 1e-4, T 0.1,
# 20 steps, reservoir_k 64, 4 episodes/cell, same three training cells. D3's tuning was
# under-powered at one seed per configuration and that is a named limitation of this
# result; re-selecting a learning rate now would be choosing it with knowledge of the
# outcome, which is what the kill criterion forbids.
#
# CL-MLP is NOT re-run (E3): its measured across-run sd is 0.0325 pp, so the sizing rule
# returns n >= 1 and the existing 16 seeds are already sixteen times oversized.
#
# %30 concurrency against 8+ idle 128-core nodes: 120 runs at ~1 h each => ~4 h wall,
# ~720 CPU-h total.
#
#SBATCH --job-name=p3-n120
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-n120-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-n120-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=17-136%30

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

SEED="$SLURM_ARRAY_TASK_ID"
if [[ "$SEED" -lt 17 || "$SEED" -gt 136 ]]; then
  echo "FAIL LOUD: seed $SEED is outside the registered 17..136 range" >&2
  exit 1
fi

ARM=gnn
INIT=models/gnn-linkmp-lgon-s8.pt   # Amendment A
LR=1e-4                             # Amendment D3 selection, carried over unchanged (E2)
T=0.1
[[ -f "$INIT" ]] || { echo "FAIL LOUD: missing $INIT" >&2; exit 1; }
[[ -f "${INIT%.pt}.contract.json" ]] || { echo "FAIL LOUD: no sidecar for $INIT" >&2; exit 1; }

OUT="simulation_data/p3_n120/${ARM}_s${SEED}"
echo "=== n120 seed $SEED: arm=$ARM lr=$LR T=$T ==="

python3 scripts_cosim/closed_loop/train_closed_loop.py \
  --arm "$ARM" --init "$INIT" \
  --sweep-dir simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core4_bw0p5 \
  --cells cell01_p25_s9001 cell02_p35_s9002 cell04_p50_s9004 \
  --workload data/nofs-ids/traces/workload-150-100-30k.json \
  --out-dir "$OUT" \
  --steps 20 --episodes-per-cell 4 \
  --temperature "$T" --lr "$LR" --reservoir-k 64 --seed "$SEED" \
  --episode-workers 8 \
  --wandb-project herosim-closed-loop \
  --wandb-run-name "n120-${ARM}-s${SEED}"

echo "=== n120 seed $SEED COMPLETE ==="
