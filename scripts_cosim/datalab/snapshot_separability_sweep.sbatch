#!/bin/bash
# WS3 (co-sim deep dive): bounded placement sweeps over the live-audit snapshots of one
# gate cell per array task, via scripts_cosim/snapshot_separability_sweep.py.
#
# Depends on job J1 (live_audit_capture_all_gates.sbatch) having produced
#   simulation_data/normal_sim_sweeps/live_audit_capture_<gate>_<cond>/snapshots/
#       <gate>_<cond>_<cell>.jsonl
# Block mapping is identical to the capture sbatch, so index i sweeps what index i
# captured. Output pseudo-datasets go under simulation_data/snapshot_sweeps/, one dir
# per (gate, cond, cell), consumable by separability_diagnostic.py unmodified.
#
# Sizing knobs (env overridable at submit): TOP_K=6 -> <=1296 combos/snapshot,
# MAX_SNAPSHOTS=50 stratified by time. Calibrated per J2 before full submit.
#SBATCH --job-name=snap-sep-sweep
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/snap-sep-sweep-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/snap-sep-sweep-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-29

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1

python3 -c "import sys;print('[INTERP]',sys.executable)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
BLOCK=$(( I / 5 ))

case "$BLOCK" in
  0) PREFIX=drawgate;  COND=backbone ;;
  1) PREFIX=drawgate;  COND=nobackbone ;;
  2) PREFIX=promo175;  COND=backbone ;;
  3) PREFIX=promo175;  COND=nobackbone ;;
  4) PREFIX=bbrob;     COND=bb_core8_bw1p5 ;;
  5) PREFIX=bbrob;     COND=bb_core4_bw0p5 ;;
  *) echo "FAIL LOUD: bad block $BLOCK (array must be 0-29)" >&2; exit 1 ;;
esac

CAP_SWEEP="simulation_data/normal_sim_sweeps/live_audit_capture_${PREFIX}_${COND}"
SNAP_PATH="${CAP_SWEEP}/snapshots/${PREFIX}_${COND}_${CELL}.jsonl"
CFG="${CAP_SWEEP}/configs/${CELL}.json"
INFRA="${CAP_SWEEP}/cell_infrastructure/${CELL}/infrastructure.json"
OUT_ROOT="simulation_data/snapshot_sweeps/${PREFIX}_${COND}_${CELL}"

[[ -s "$SNAP_PATH" ]] || { echo "FAIL LOUD: no snapshots at $SNAP_PATH (run J1 first)" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: missing cell config $CFG" >&2; exit 1; }
[[ -f "$INFRA" ]] || { echo "FAIL LOUD: missing cell infrastructure $INFRA" >&2; exit 1; }
mkdir -p logs

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} snapshots=$(wc -l < "$SNAP_PATH") ==="
python3 scripts_cosim/snapshot_separability_sweep.py \
  --snapshots "$SNAP_PATH" \
  --config "$CFG" \
  --cell-infra "$INFRA" \
  --sim-input data/nofs-ids \
  --out-root "$OUT_ROOT" \
  --top-k "${TOP_K:-6}" \
  --horizon 4 \
  --max-snapshots "${MAX_SNAPSHOTS:-50}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  ${CALIBRATE:+--calibrate}
echo "=== task=${I} COMPLETE ==="
