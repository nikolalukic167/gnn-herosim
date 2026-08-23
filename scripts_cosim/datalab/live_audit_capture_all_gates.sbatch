#!/bin/bash
# WS2 (co-sim deep dive): capture live-regime oracle-audit snapshots on all 30 gate cells.
#
# Runs the knative_network_batch arm (same shortest-queue rule the gates' knative arm
# uses, but batched so batch>=4 states qualify for capture) on every (gate, condition,
# cell) of the drawgate / promo175 / bbrob matrix, with LIVE_AUDIT_SNAPSHOT_PATH set.
# The product is NOT the result JSON (which stays on the cluster, ~80MB) but the
# per-cell snapshot JSONL (~KBs-MBs): batch states with per-task candidate lists, to be
# swept by the live-regime separability experiment (WS3). Capture filters follow the
# 4-candidate audit pipeline: batch >= 4 tasks, every task >= 4 candidates.
#
# Stride is 1 and MAX_SNAPSHOTS generous (500): qualifying batches may be rare, so we
# capture everything and subsample stratified-by-time locally rather than re-running.
#
# The block -> (cells, workload, PARITY_EXTRA_ARGS) mapping is byte-matched to
# mlp_tempfix_arm_all_gates.sbatch so the captured states come from exactly the gate
# cells the GNN/MLP verdicts were measured on.
#
# 30 tasks: 3 gates x 2 conditions x 5 cells.
#SBATCH --job-name=live-audit-capture
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/live-audit-capture-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/live-audit-capture-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-29

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
BLOCK=$(( I / 5 ))

BACKBONE_CELLS="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
FLAT_CELLS="simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"

case "$BLOCK" in
  0) PREFIX=drawgate;  COND=backbone;         SRC_SWEEP="$BACKBONE_CELLS"; WL=150-100
     export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence" ;;
  1) PREFIX=drawgate;  COND=nobackbone;       SRC_SWEEP="$FLAT_CELLS";     WL=150-100
     export PARITY_EXTRA_ARGS="" ;;
  2) PREFIX=promo175;  COND=backbone;         SRC_SWEEP="$BACKBONE_CELLS"; WL=175-100
     export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence" ;;
  3) PREFIX=promo175;  COND=nobackbone;       SRC_SWEEP="$FLAT_CELLS";     WL=175-100
     export PARITY_EXTRA_ARGS="" ;;
  4) PREFIX=bbrob;     COND=bb_core8_bw1p5;   SRC_SWEEP="simulation_data/normal_sim_sweeps/bb_core8_bw1p5"; WL=150-100
     export PARITY_EXTRA_ARGS="" ;;
  5) PREFIX=bbrob;     COND=bb_core4_bw0p5;   SRC_SWEEP="simulation_data/normal_sim_sweeps/bb_core4_bw0p5"; WL=150-100
     export PARITY_EXTRA_ARGS="" ;;
  *) echo "FAIL LOUD: bad block $BLOCK (array must be 0-29)" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/live_audit_capture_${PREFIX}_${COND}"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"

SNAP_DIR="${SWEEP_DIR}/snapshots"
SNAP_PATH="${SNAP_DIR}/${PREFIX}_${COND}_${CELL}.jsonl"

# Idempotence keys on the CAPTURE artifact, not the result JSON.
if [[ -s "$SNAP_PATH" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "SKIP (snapshots exist): $SNAP_PATH ($(wc -l < "$SNAP_PATH") lines)"
  exit 0
fi
rm -f "$SNAP_PATH"   # capture appends; never mix two runs in one JSONL

# Snapshots are the product: if a stale result JSON exists without its snapshot file, the
# runner's SKIP-on-result branch would silently skip the capture too. Force the sim.
export FORCE_RERUN=1

export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP_PATH"
export LIVE_AUDIT_MAX_SNAPSHOTS=500
export LIVE_AUDIT_STRIDE=1
export LIVE_AUDIT_MIN_BATCH_SIZE=4
export LIVE_AUDIT_MIN_CANDIDATES=4
export KNATIVE_BATCH_SIZE=4
export GNN_CAPTURE_DATASET_STATE=0

# The gate runner asserts these exist even for knative arms; never loaded on this path.
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"
export MLP_MODEL="models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt"

[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# mkdir -p + rsync, never `[[ -d ]] || cp -r` (not atomic; killed 3 tasks of job 710315).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" "$SNAP_DIR" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} trace=${WL} arm=knative_batch capture=on ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh knative_batch "$CELL"

SNAP_COUNT="$(wc -l < "$SNAP_PATH" 2>/dev/null || echo 0)"
echo "=== task=${I} captured ${SNAP_COUNT} qualifying snapshots -> ${SNAP_PATH} ==="
if [[ "$SNAP_COUNT" -eq 0 ]]; then
  echo "WARNING: zero qualifying snapshots on ${PREFIX}/${COND}/${CELL} — check yield (D4)" >&2
fi
echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} COMPLETE ==="
