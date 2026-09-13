#!/bin/bash
# objective_pivot_v1 Phase 2 (P3): the registered in-harness horizon calibration.
#
# Pre-registered 2026-08-24 in program_verdict_v1 (P3 entry): before the n>=300 array
# is queued, 3 snapshots at h in {2, 5, 10} must (a) confirm combos genuinely run
# in-process — if each combo pays the ~5.1 s process startup the budget quadruples and
# the pilot is re-scoped before queueing — and (b) measure the per-combo cost per h, so
# h is fixed at the largest value the stated budget affords and registered before the
# array. h >= 5 s is required for axis-level terminality of a null.
#
# Runs on the slowest backbone fabric (bbrob bb_core4_bw0p5 cell01) — worst-case cost.
# Consumes the post-P3 re-capture (job 732878 task 25), which records replicas_by_type.
#
#SBATCH --job-name=p3-horizon-calib
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-horizon-calib-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-horizon-calib-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00

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

CAP="simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core4_bw0p5"
SNAPS="${CAP}/snapshots/bbrob_bb_core4_bw0p5_cell01_p25_s9001.jsonl"
CONFIG="${CAP}/configs/cell01_p25_s9001.json"
INFRA="${CAP}/cell_infrastructure/cell01_p25_s9001/infrastructure.json"
TRACE="data/nofs-ids/traces/workload-150-100.json"
OUT_ROOT="scratch/p3_calibration"

[[ -s "$SNAPS" ]] || { echo "FAIL LOUD: no snapshots at $SNAPS (capture not done?)" >&2; exit 1; }
# The calibration is meaningless on pre-P3 snapshots (no replicas_by_type — the sweep
# would fail loud anyway, but say why up front).
head -1 "$SNAPS" | grep -q '"replicas_by_type"' || {
  echo "FAIL LOUD: $SNAPS has no replicas_by_type — this is a pre-P3 capture" >&2; exit 1; }

for H in 2 5 10; do
  echo "=== calibration h=${H}s (3 snapshots, top-k 4 => 256 combos each) ==="
  python3 scripts_cosim/snapshot_separability_sweep.py \
    --snapshots "$SNAPS" \
    --config "$CONFIG" \
    --cell-infra "$INFRA" \
    --out-root "${OUT_ROOT}/h${H}" \
    --trace "$TRACE" \
    --horizon-seconds "$H" \
    --top-k 4 --max-snapshots 3 --workers 16
done

echo "=== calibration summary ==="
python3 - << 'EOF'
import json
from pathlib import Path

print(f"{'h':>4} {'snapshot':>10} {'t_snap':>8} {'combos':>7} {'arrivals':>9} "
      f"{'ms/combo':>9} {'elapsed_s':>10}")
for h in (2, 5, 10):
    root = Path(f"scratch/p3_calibration/h{h}")
    metas = sorted(root.glob("ds_snap_*/snapshot_meta.json"))
    for mp in metas:
        m = json.loads(mp.read_text())
        print(f"{h:>4} {mp.parent.name:>10} {m['time']:>8.1f} {m['swept_combos']:>7} "
              f"{m['n_horizon_events']:>9} {m['per_combo_ms']:>9.0f} {m['elapsed_s']:>10.1f}")
EOF
echo "=== p3 calibration COMPLETE ==="
