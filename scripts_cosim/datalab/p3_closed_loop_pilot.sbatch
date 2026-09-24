#!/bin/bash
# objective_pivot_v1 Phase 3 — the P1 closed-loop pilot (CL-GNN / CL-MLP).
#
# WHAT THIS IS AND IS NOT
#   This trains one arm. It does NOT produce the registered result: the kill criterion
#   reads the paired-seed CL-vs-Frozen difference from a live gate on held-out cells, and
#   that is a separate submission against the checkpoint this one writes.
#
# THE BUDGET IS NOT REGISTERED YET
#   The Increment-1 sizing rule `n >= (2*sd*2.8/MDE)^2` returned n >= 1 on the measured
#   sd (0.0029-0.0053), which is arithmetically correct and operationally useless. That
#   defect is named in docs/lineages/objective_pivot_v1.md and needs a signed amendment.
#   STEPS/EPISODES below are therefore a *shakedown* size — enough to see the loop move
#   and to measure per-step cost — not the pilot n. Do not report a result off this
#   script until the amendment fixes the budget.
#
# COST ARITHMETIC (measured locally 2026-09-02, 30k-event inner-loop trace)
#   ~72 s per episode, 1 greedy + S sampled per cell per step.
#   3 cells x (1 + 4) = 15 episodes/step; at 8 workers that is ~2 steps per 10 min.
#   STEPS=20 => ~300 episodes => ~6 CPU-h of episode time, ~50 min wall at 8 workers.
#
#SBATCH --job-name=p3-closed-loop
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-closed-loop-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-closed-loop-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
# Never `pipenv run` under sbatch: it resolves its own venv and shells straight past the
# micromamba activation above, which is how a third undeclared environment came to run
# every gate on this cluster.
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
# One thread per episode process; the parallelism is --episode-workers, not BLAS.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

ARM="${ARM:-gnn}"
case "$ARM" in
  gnn) INIT="${INIT:-models/gnn-linkmp-lgon-s8.pt}" ;;   # Amendment A
  mlp) INIT="${INIT:?set INIT to the promoted MLP checkpoint}" ;;
  *)   echo "FAIL LOUD: ARM must be gnn or mlp, got $ARM" >&2; exit 1 ;;
esac
[[ -f "$INIT" ]] || { echo "FAIL LOUD: missing $INIT" >&2; exit 1; }
if [[ "$ARM" == "gnn" && ! -f "${INIT%.pt}.contract.json" ]]; then
  echo "FAIL LOUD: no sidecar for $INIT — architecture flags invisible in the weights" >&2
  exit 1
fi

WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-150-100-30k.json}"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD (gitignored; rsync it)" >&2; exit 1; }

STEPS="${STEPS:-20}"
EPISODES="${EPISODES:-4}"
TEMPERATURE="${TEMPERATURE:-0.1}"   # Increment 1: 17.3% exploration for +0.05% RTT
SEED="${SEED:-1}"
RESERVOIR_K="${RESERVOIR_K:-64}"

python3 scripts_cosim/closed_loop/train_closed_loop.py \
  --arm "$ARM" \
  --init "$INIT" \
  --sweep-dir simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core4_bw0p5 \
  --cells cell01_p25_s9001 cell02_p35_s9002 cell04_p50_s9004 \
  --workload "$WORKLOAD" \
  --out-dir "simulation_data/p3_closed_loop/${ARM}_T${TEMPERATURE}_s${SEED}" \
  --steps "$STEPS" \
  --episodes-per-cell "$EPISODES" \
  --temperature "$TEMPERATURE" \
  --reservoir-k "$RESERVOIR_K" \
  --seed "$SEED" \
  --episode-workers 8

echo "=== p3 closed-loop training COMPLETE (arm=$ARM seed=$SEED) ==="
echo "Next: live-gate the final checkpoint against Frozen-GNN / Frozen-MLP / Knative."
