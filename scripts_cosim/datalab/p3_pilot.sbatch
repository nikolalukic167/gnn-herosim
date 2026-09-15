#!/bin/bash
# objective_pivot_v1 Phase 3 — the P1 pilot. 16 paired training seeds per trained arm.
#
# Amendment D2 sets the floor at 16 seeds per arm; the replication unit is the TRAINING
# RUN, not the episode and not the cell (D1). Array indices 0-15 are CL-GNN seeds 1-16,
# 16-31 are CL-MLP seeds 1-16. Both arms draw the same episode seeds at a given --seed,
# which is the common-random-numbers pairing the registered primary statistic is computed
# over.
#
# HYPERPARAMETERS ARE FROZEN HERE, from the Amendment D3 dev-cell selection (2026-09-02),
# and are NOT revisited after any pilot or gate number exists:
#   CL-GNN  lr = 1e-4, T = 0.1   (dev +8.50% vs frozen)
#   CL-MLP  lr = 1e-6, T = 0.1   (dev +0.03% vs frozen -- the least-damaging rung; every
#                                 config that actually moved the MLP made it worse)
# Each arm runs at ITS OWN dev optimum rather than a shared value: the registration asks
# for the same loop, objective, budget and seeds, not the same learning rate, and giving
# both arms their best shot is the fairer comparison.
#
# %8 caps concurrency so 8 x 8 = 64 cores are in flight; 32 jobs at ~1 h each => ~4 h.
#
#SBATCH --job-name=p3-pilot
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-pilot-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-pilot-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-31%8

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

IDX="$SLURM_ARRAY_TASK_ID"
if [[ "$IDX" -lt 16 ]]; then
  ARM=gnn
  SEED=$((IDX + 1))
  INIT=models/gnn-linkmp-lgon-s8.pt                                             # Amendment A
  LR=1e-4
else
  ARM=mlp
  SEED=$((IDX - 15))
  INIT=models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt
  LR=1e-6
fi
T=0.1
[[ -f "$INIT" ]] || { echo "FAIL LOUD: missing $INIT" >&2; exit 1; }

OUT="simulation_data/p3_pilot/${ARM}_s${SEED}"
echo "=== pilot task $IDX: arm=$ARM seed=$SEED lr=$LR T=$T ==="

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
  --wandb-run-name "pilot-${ARM}-s${SEED}"

echo "=== pilot task $IDX COMPLETE (arm=$ARM seed=$SEED) ==="
