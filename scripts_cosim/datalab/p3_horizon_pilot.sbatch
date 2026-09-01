#!/bin/bash
# objective_pivot_v1 Phase 2 — the P3 horizon pilot array, h REGISTERED at 10 s.
#
# Registration chain: thresholds fixed 2026-08-24 (program_verdict_v1 P3 entry:
# co-primaries median additive_choice_regret_rel > 0.02 OR frac(regret > 2%) >= 15%,
# each firing only with node-count repair < 0.5 AND link repair < 0.5 on the affected
# stratum; n >= 300 snapshots, >= 2 backbone cells, K=4 => 256 combos). h fixed by the
# registered calibration rule 2026-09-01 (job 732918): largest affordable value —
# measured 16.2–28.2 CPU-s/combo at h=10 => ~340–600 CPU-h for the array, inside the
# registered anchor; h >= 5 axis-terminality bound satisfied. Truncated-window
# snapshots near trace end are KEPT (dropping them would be an unregistered filter);
# n_horizon_events in each snapshot_meta.json records the truncation.
#
# 10 tasks: 2 backbone fabrics x 5 cells, 30 snapshots each = 300 snapshots,
# 76,800 mini co-sims. Memory: h=10 at 16 workers peaked 71 GB in calibration.
#
#SBATCH --job-name=p3-horizon-pilot
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-horizon-pilot-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-horizon-pilot-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --array=0-9

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

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
if (( I / 5 == 0 )); then COND=bb_core4_bw0p5; else COND=bb_core8_bw1p5; fi

CAP="simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_${COND}"
SNAPS="${CAP}/snapshots/bbrob_${COND}_${CELL}.jsonl"
CONFIG="${CAP}/configs/${CELL}.json"
INFRA="${CAP}/cell_infrastructure/${CELL}/infrastructure.json"
TRACE="data/nofs-ids/traces/workload-150-100.json"
OUT="simulation_data/snapshot_sweeps_p3_h10/bbrob_${COND}_${CELL}"

[[ -s "$SNAPS" ]] || { echo "FAIL LOUD: no snapshots at $SNAPS" >&2; exit 1; }
head -1 "$SNAPS" | grep -q '"replicas_by_type"' || {
  echo "FAIL LOUD: $SNAPS is a pre-P3 capture (no replicas_by_type)" >&2; exit 1; }

echo "=== task=${I} cond=${COND} cell=${CELL} h=10s snapshots=30 top_k=4 ==="
python3 scripts_cosim/snapshot_separability_sweep.py \
  --snapshots "$SNAPS" \
  --config "$CONFIG" \
  --cell-infra "$INFRA" \
  --out-root "$OUT" \
  --trace "$TRACE" \
  --horizon-seconds 10 \
  --top-k 4 --max-snapshots 30 --workers 16

echo "=== task=${I} ${COND}/${CELL} COMPLETE ==="
