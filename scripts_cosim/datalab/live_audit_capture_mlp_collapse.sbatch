#!/bin/bash
# WS2 extension (co-sim deep dive): capture live-audit snapshots from the MLP arms on
# their 14 collapse events, so WS3 can sweep TRUE collapse-moment states — not just the
# Knative-visited proxy states of live_audit_capture_all_gates.sbatch.
#
# The capture hook landed in the GNN scheduler family (src/placement/live_audit.py,
# d88278c), which MLPBatchScheduler inherits, so an mlp run with
# LIVE_AUDIT_SNAPSHOT_PATH set writes the same snapshot schema as knative_network_batch.
#
# The 14 events (chosen_queue_vs_min p95 > 5000, reproduced from
# simulation_data/gate_stats_summary.json on 2026-08-23) map to array indices below —
# one task per (gate, condition, cell, checkpoint) event.
#SBATCH --job-name=live-audit-mlp-collapse
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/live-audit-mlp-collapse-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/live-audit-mlp-collapse-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-13

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

BACKBONE_CELLS="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
FLAT_CELLS="simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"
BB8="simulation_data/normal_sim_sweeps/bb_core8_bw1p5"
BB4="simulation_data/normal_sim_sweeps/bb_core4_bw0p5"
MLP_STD="models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt"
MLP_TFX="models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt"

# gate cond cell arm — the 14 collapse events.
case "$SLURM_ARRAY_TASK_ID" in
  0)  PREFIX=bbrob;    COND=bb_core4_bw0p5; CELL=cell03_p15_s9003; ARM=mlp;        SRC_SWEEP="$BB4"; WL=150-100; PW="" ;;
  1)  PREFIX=bbrob;    COND=bb_core4_bw0p5; CELL=cell05_p20_s9005; ARM=mlp;        SRC_SWEEP="$BB4"; WL=150-100; PW="" ;;
  2)  PREFIX=bbrob;    COND=bb_core4_bw0p5; CELL=cell05_p20_s9005; ARM=mlptempfix; SRC_SWEEP="$BB4"; WL=150-100; PW="" ;;
  3)  PREFIX=bbrob;    COND=bb_core8_bw1p5; CELL=cell03_p15_s9003; ARM=mlp;        SRC_SWEEP="$BB8"; WL=150-100; PW="" ;;
  4)  PREFIX=bbrob;    COND=bb_core8_bw1p5; CELL=cell05_p20_s9005; ARM=mlp;        SRC_SWEEP="$BB8"; WL=150-100; PW="" ;;
  5)  PREFIX=bbrob;    COND=bb_core8_bw1p5; CELL=cell05_p20_s9005; ARM=mlptempfix; SRC_SWEEP="$BB8"; WL=150-100; PW="" ;;
  6)  PREFIX=drawgate; COND=backbone;       CELL=cell05_p20_s9005; ARM=mlp;        SRC_SWEEP="$BACKBONE_CELLS"; WL=150-100; PW="--allow-backbone-latency-divergence" ;;
  7)  PREFIX=drawgate; COND=backbone;       CELL=cell05_p20_s9005; ARM=mlptempfix; SRC_SWEEP="$BACKBONE_CELLS"; WL=150-100; PW="--allow-backbone-latency-divergence" ;;
  8)  PREFIX=drawgate; COND=nobackbone;     CELL=cell03_p15_s9003; ARM=mlptempfix; SRC_SWEEP="$FLAT_CELLS"; WL=150-100; PW="" ;;
  9)  PREFIX=drawgate; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlp;        SRC_SWEEP="$FLAT_CELLS"; WL=150-100; PW="" ;;
  10) PREFIX=drawgate; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlptempfix; SRC_SWEEP="$FLAT_CELLS"; WL=150-100; PW="" ;;
  11) PREFIX=promo175; COND=backbone;       CELL=cell05_p20_s9005; ARM=mlptempfix; SRC_SWEEP="$BACKBONE_CELLS"; WL=175-100; PW="--allow-backbone-latency-divergence" ;;
  12) PREFIX=promo175; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlp;        SRC_SWEEP="$FLAT_CELLS"; WL=175-100; PW="" ;;
  13) PREFIX=promo175; COND=nobackbone;     CELL=cell05_p20_s9005; ARM=mlptempfix; SRC_SWEEP="$FLAT_CELLS"; WL=175-100; PW="" ;;
  *) echo "FAIL LOUD: bad array index $SLURM_ARRAY_TASK_ID (must be 0-13)" >&2; exit 1 ;;
esac
export PARITY_EXTRA_ARGS="$PW"

if [[ "$ARM" == "mlptempfix" ]]; then export MLP_MODEL="$MLP_TFX"; else export MLP_MODEL="$MLP_STD"; fi
export SWEEP_DIR="simulation_data/normal_sim_sweeps/live_audit_mlpcollapse_${PREFIX}_${COND}_${ARM}"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"

SNAP_DIR="${SWEEP_DIR}/snapshots"
SNAP_PATH="${SNAP_DIR}/${PREFIX}_${COND}_${CELL}_${ARM}.jsonl"

if [[ -s "$SNAP_PATH" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "SKIP (snapshots exist): $SNAP_PATH ($(wc -l < "$SNAP_PATH") lines)"
  exit 0
fi
rm -f "$SNAP_PATH"
export FORCE_RERUN=1

export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP_PATH"
export LIVE_AUDIT_MAX_SNAPSHOTS=500
if [[ "$WL" == "175-100" ]]; then export LIVE_AUDIT_STRIDE=37; else export LIVE_AUDIT_STRIDE=31; fi
export LIVE_AUDIT_MIN_BATCH_SIZE=4
export LIVE_AUDIT_MIN_CANDIDATES=4
export GNN_BATCH_SIZE=4
export GNN_CAPTURE_DATASET_STATE=0

[[ -f "$MLP_MODEL" ]] || { echo "FAIL LOUD: missing $MLP_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# mkdir -p + rsync, never `[[ -d ]] || cp -r` (not atomic; killed 3 tasks of job 710315).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" "$SNAP_DIR" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${SLURM_ARRAY_TASK_ID} gate=${PREFIX} cond=${COND} cell=${CELL} arm=${ARM} capture=on ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh mlp "$CELL"

SNAP_COUNT="$(wc -l < "$SNAP_PATH" 2>/dev/null || echo 0)"
echo "=== task=${SLURM_ARRAY_TASK_ID} captured ${SNAP_COUNT} qualifying snapshots -> ${SNAP_PATH} ==="
if [[ "$SNAP_COUNT" -eq 0 ]]; then
  echo "WARNING: zero qualifying snapshots on ${PREFIX}/${COND}/${CELL}/${ARM} (D4)" >&2
fi
echo "=== task=${SLURM_ARRAY_TASK_ID} COMPLETE ==="
