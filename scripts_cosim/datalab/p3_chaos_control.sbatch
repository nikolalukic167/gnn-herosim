#!/bin/bash
# P3 CHAOS CONTROL (exploratory, post-hoc — NOT part of the registered verdict).
#
# The registered co-primary (b) fired at 21.7% of snapshots above 2% regret. Before
# those horizon labels are used as Phase 3 DAgger targets, one alternative explanation
# has to be ruled out: in a deterministic discrete-event simulator, a tiny placement
# difference can reshuffle queue orderings seconds later and produce large,
# deterministic-but-arbitrary RTT differences that encode nothing about placement
# quality. That would look exactly like "regret" while being unlearnable.
#
# The test: sweep the SAME snapshots at h = 2 and 5 and compare the placement RANKING
# to h = 10. Genuine dispersal value is horizon-stable (a good decision is good at
# every horizon => high rank correlation). Chaotic amplification is not (rankings
# reshuffle => rho ~ 0).
#
# A first pass on the calibration snapshots gave rho ~ 0.03-0.18 — but those sit in
# cell01, the LOWEST-signal cell in the pilot (0/30 snapshots above the bar), where
# every difference is tiny and near-zero correlation proves nothing. This runs the
# control where the signal actually is: the highest-regret snapshots of the pilot,
# on the p50 cells that carry most of the firing mass.
#
#SBATCH --job-name=p3-chaos-ctrl
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-chaos-ctrl-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-chaos-ctrl-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (cond, cell, horizon) — the two p50 cells that hold the top regrets, at h=2 and h=5.
case "$SLURM_ARRAY_TASK_ID" in
  0) COND=bb_core4_bw0p5; CELL=cell04_p50_s9004; H=2 ;;
  1) COND=bb_core4_bw0p5; CELL=cell04_p50_s9004; H=5 ;;
  2) COND=bb_core8_bw1p5; CELL=cell04_p50_s9004; H=2 ;;
  3) COND=bb_core8_bw1p5; CELL=cell04_p50_s9004; H=5 ;;
  *) echo "FAIL LOUD: bad array id" >&2; exit 1 ;;
esac

CAP="simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_${COND}"
python3 scripts_cosim/snapshot_separability_sweep.py \
  --snapshots "${CAP}/snapshots/bbrob_${COND}_${CELL}.jsonl" \
  --config "${CAP}/configs/${CELL}.json" \
  --cell-infra "${CAP}/cell_infrastructure/${CELL}/infrastructure.json" \
  --out-root "simulation_data/snapshot_sweeps_p3_chaos/h${H}/bbrob_${COND}_${CELL}" \
  --trace data/nofs-ids/traces/workload-150-100.json \
  --horizon-seconds "$H" \
  --top-k 4 --max-snapshots 30 --workers 16

echo "=== chaos control ${COND}/${CELL} h=${H} COMPLETE ==="
