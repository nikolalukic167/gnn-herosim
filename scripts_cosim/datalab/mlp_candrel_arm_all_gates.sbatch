#!/bin/bash
# P5b (program_verdict_v1): the candidate-relative MLP arms on all 30 backbone gate cells,
# for BOTH caches — 60 tasks.
#
# The question: is the MLP's collapse architectural, or was it just never given the
# information? A pointwise edge score cannot condition on the candidate set; message
# passing can. So hand the MLP that view directly as three engineered columns (dim25cr:
# delta-to-min, within-set rank, within-set z-score over the same normalized queue column
# it already reads) and re-run the same 30 cells.
#
#   * collapse DISAPPEARS -> the reliability separation is feature engineering. The paper
#     claim shrinks, and every P5a gate run before knowing this would have been wasted —
#     which is exactly why this control is registered FIRST.
#   * collapse SURVIVES   -> the architectural reading stands, now against the strongest
#     hand-engineered version of the missing feature rather than against its absence.
#
# Each arm is paired against the baseline trained on the SAME cache, so the pairing is an
# A/B on the feature set alone:
#     mlpcandrel   (dim14 cache)         vs mlp
#     mlpcandreltf (dim14_tempfix cache) vs mlptempfix
#
# Everything except the checkpoint, the sweep dir and INFERENCE_FEATURE_LAYOUT is
# byte-identical to mlp_tempfix_arm_all_gates.sbatch on purpose. In particular the block ->
# (cells, workload, PARITY_EXTRA_ARGS) mapping below is copied verbatim: it is matched
# per-gate to the Knative arm each run is scored against, and the bbrob blocks must keep
# PARITY_EXTRA_ARGS="".
#
# SWEEP_DIR gets its own per-arm suffix because run_full_corpus_siv1_live_gate.sh names the
# result file from the POLICY (`<cell>_s0_mlp_dim22.json`), not the checkpoint. Two MLP arms
# sharing one sweep dir would overwrite each other -- worse, the runner's "SKIP (exists)"
# branch would silently keep the first arm's numbers and report success.
#
# 60 tasks: 2 arms x 3 gates x 2 conditions x 5 cells.
#SBATCH --job-name=mlp-candrel-gate-arm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/mlp-candrel-gate-arm-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/mlp-candrel-gate-arm-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-59

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

# run_simulation.py re-spawns the simulator; since 2026-08-23 it honors HEROSIM_PY, so
# this now really is the activated env end to end. Print both to make it checkable.
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
ARM_IDX=$(( I / 30 ))
J=$(( I % 30 ))
CELL="${CELLS[$(( J % 5 ))]}"
BLOCK=$(( J / 5 ))

# Which cache the checkpoint came from decides which baseline arm it pairs against; the
# scorer keys on the arm name, so these two must not be swapped.
case "$ARM_IDX" in
  0) ARM=mlpcandrel
     MLP_MODEL="models/tabular/batch_edge_mlp_full_corpus_siv1_dim25cr_batchcache.pt" ;;
  1) ARM=mlpcandreltf
     MLP_MODEL="models/tabular/batch_edge_mlp_full_corpus_siv1_dim25cr_batchcache_tempfix.pt" ;;
  *) echo "FAIL LOUD: bad arm index $ARM_IDX (array must be 0-59)" >&2; exit 1 ;;
esac
export MLP_MODEL

BACKBONE_CELLS="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
FLAT_CELLS="simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"

# Each block names the gate whose knative arm this MLP run will be scored against, so the
# cells and the workload must match that gate's sbatch exactly. PARITY_EXTRA_ARGS follows
# the same rule: the legacy backbone cells (a1_backbone_bw1p5) predate the rng_stream fix
# and need the waiver; the bbrob cells are parity-exact by construction and must not get
# it -- if they ever need it, the fix has regressed and this should fail, not be relaxed.
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
  *) echo "FAIL LOUD: bad block $BLOCK (array must be 0-59)" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/${PREFIX}_${COND}_${ARM}"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"

# dim25cr, NOT dim22: the checkpoint declares its own layout and the loader treats a
# conflicting declaration as a hard error, so a stale dim22 here fails loudly at load
# rather than serving 22 of 25 columns under the wrong meaning.
export INFERENCE_FEATURE_LAYOUT=dim25cr

# The gate runner insists on a GNN checkpoint existing even when the policy is mlp; point
# it at the deployed one so the check is truthful. It is never loaded on this path.
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"

[[ -f "$MLP_MODEL" ]] || { echo "FAIL LOUD: missing $MLP_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# The pairing is only meaningful if this arm really loaded a candidate-relative checkpoint
# under its own sweep dir. Assert both before anything runs, since a stale SWEEP_DIR result
# would be reported as a SKIP.
[[ "$SWEEP_DIR" == *_mlpcandrel || "$SWEEP_DIR" == *_mlpcandreltf ]] || {
  echo "FAIL LOUD: SWEEP_DIR must not be shared with a baseline MLP arm: $SWEEP_DIR" >&2; exit 1; }
python3 - "$MLP_MODEL" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if ck.get("inference_feature_layout") != "dim25cr" or int(ck["input_dim"]) != 25:
    raise SystemExit(
        f"FAIL LOUD: {sys.argv[1]} is layout={ck.get('inference_feature_layout')!r} "
        f"input_dim={ck.get('input_dim')} — this arm must serve a dim25cr/25 checkpoint")
print(f"[CKPT] {sys.argv[1]} layout=dim25cr input_dim=25 "
      f"contract={ck.get('queue_feature_contract')}")
PY

# mkdir -p + rsync, never `[[ -d ]] || cp -r` (not atomic; killed 3 tasks of job 710315).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} trace=${WL} arm=${ARM} layout=dim25cr ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh mlp "$CELL"
echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} arm=${ARM} COMPLETE ==="
