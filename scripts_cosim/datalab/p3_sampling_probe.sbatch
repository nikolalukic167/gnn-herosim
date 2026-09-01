#!/bin/bash
# objective_pivot_v1 Phase 3, Increment 1 — the registered sampling-feasibility probe.
#
# Reading fixed 2026-09-01 BEFORE this ran (see the Phase 3 registration in
# docs/lineages/objective_pivot_v1.md):
#   GO             some T has d(T) <= 0.10 AND sd <= 0.05
#   NO-GO          every T has d(T) > 0.25, OR sd > 0.15
#   INDETERMINATE  anything else — returned to the user with the numbers, never
#                  resolved by choosing a T after seeing the outcome.
#
# 3 backbone cells × (1 argmax + 3 temperatures × 5 seeds) = 48 full episodes on the
# 30k-event inner-loop trace. Local smoke: ~90 s/episode, so ~75 min single-threaded;
# episodes are run sequentially on purpose so each gets an uncontended core and the
# per-episode noise measured is the POLICY's, not the scheduler's.
#
#SBATCH --job-name=p3-sampling-probe
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-sampling-probe-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-sampling-probe-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

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

MODEL="models/gnn-linkmp-lgon-s8.pt"
[[ -f "$MODEL" ]] || { echo "FAIL LOUD: missing $MODEL" >&2; exit 1; }
[[ -f "${MODEL%.pt}.contract.json" ]] || { echo "FAIL LOUD: no sidecar for $MODEL" >&2; exit 1; }

python3 scripts_cosim/closed_loop/run_sampling_probe.py \
  --sweep-dir simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core4_bw0p5 \
  --cells cell01_p25_s9001 cell02_p35_s9002 cell04_p50_s9004 \
  --model "$MODEL" \
  --workload data/nofs-ids/traces/workload-150-100-30k.json \
  --temperatures 0.1 0.3 1.0 \
  --noise-seeds 5 \
  --out-dir simulation_data/p3_sampling_probe

echo "=== p3 sampling probe COMPLETE ==="
