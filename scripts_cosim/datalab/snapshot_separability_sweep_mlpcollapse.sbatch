#!/bin/bash
# WS3 (co-sim deep dive): bounded placement sweeps over the MLP collapse-moment
# snapshots (captured by live_audit_capture_mlp_collapse.sbatch), one array task per
# collapse event. Same driver and sizing as snapshot_separability_sweep.sbatch.
#SBATCH --job-name=snap-sep-mlpcollapse
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/snap-sep-mlpcollapse-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/snap-sep-mlpcollapse-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-13

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1

python3 -c "import sys;print('[INTERP]',sys.executable)"

# Same event mapping as live_audit_capture_mlp_collapse.sbatch.
case "$SLURM_ARRAY_TASK_ID" in
  0)  PREFIX=bbrob;    COND=bb_core4_bw0p5; CELL=cell03_p15_s9003; ARM=mlp ;;
  1)  PREFIX=bbrob;    COND=bb_core4_bw0p5; CELL=cell05_p20_s9005; ARM=mlp ;;
  2)  PREFIX=bbrob;    COND=bb_core4_bw0p5; CELL=cell05_p20_s9005; ARM=mlptempfix ;;
  3)  PREFIX=bbrob;    COND=bb_core8_bw1p5; CELL=cell03_p15_s9003; ARM=mlp ;;
  4)  PREFIX=bbrob;    COND=bb_core8_bw1p5; CELL=cell05_p20_s9005; ARM=mlp ;;
  5)  PREFIX=bbrob;    COND=bb_core8_bw1p5; CELL=cell05_p20_s9005; ARM=mlptempfix ;;
  6)  PREFIX=drawgate; COND=backbone;       CELL=cell05_p20_s9005; ARM=mlp ;;
  7)  PREFIX=drawgate; COND=backbone;       CELL=cell05_p20_s9005; ARM=mlptempfix ;;
  8)  PREFIX=drawgate; COND=nobackbone;     CELL=cell03_p15_s9003; ARM=mlptempfix ;;
  9)  PREFIX=drawgate; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlp ;;
  10) PREFIX=drawgate; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlptempfix ;;
  11) PREFIX=promo175; COND=backbone;       CELL=cell05_p20_s9005; ARM=mlptempfix ;;
  12) PREFIX=promo175; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlp ;;
  13) PREFIX=promo175; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlptempfix ;;
  *) echo "FAIL LOUD: bad array index $SLURM_ARRAY_TASK_ID (must be 0-13)" >&2; exit 1 ;;
esac

CAP_SWEEP="simulation_data/normal_sim_sweeps/live_audit_mlpcollapse_${PREFIX}_${COND}_${ARM}"
SNAP_PATH="${CAP_SWEEP}/snapshots/${PREFIX}_${COND}_${CELL}_${ARM}.jsonl"
CFG="${CAP_SWEEP}/configs/${CELL}.json"
INFRA="${CAP_SWEEP}/cell_infrastructure/${CELL}/infrastructure.json"
OUT_ROOT="simulation_data/snapshot_sweeps_mlpcollapse/${PREFIX}_${COND}_${CELL}_${ARM}"

[[ -s "$SNAP_PATH" ]] || { echo "FAIL LOUD: no snapshots at $SNAP_PATH (run capture first)" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: missing cell config $CFG" >&2; exit 1; }
[[ -f "$INFRA" ]] || { echo "FAIL LOUD: missing cell infrastructure $INFRA" >&2; exit 1; }
mkdir -p logs

echo "=== task=${SLURM_ARRAY_TASK_ID} gate=${PREFIX} cond=${COND} cell=${CELL} arm=${ARM} snapshots=$(wc -l < "$SNAP_PATH") ==="
python3 scripts_cosim/snapshot_separability_sweep.py \
  --snapshots "$SNAP_PATH" \
  --config "$CFG" \
  --cell-infra "$INFRA" \
  --sim-input data/nofs-ids \
  --out-root "$OUT_ROOT" \
  --top-k "${TOP_K:-6}" \
  --horizon 4 \
  --max-snapshots "${MAX_SNAPSHOTS:-100}" \
  --workers "${SLURM_CPUS_PER_TASK}"
echo "=== task=${SLURM_ARRAY_TASK_ID} COMPLETE ==="
