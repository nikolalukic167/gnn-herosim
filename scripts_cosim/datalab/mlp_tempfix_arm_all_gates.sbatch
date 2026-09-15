#!/bin/bash
# The SECOND MLP arm: the same pointwise baseline under the corrected batch cache.
#
# mlp_arm_all_gates.sbatch ran the standard siv1 MLP and found 7 of 30 cells collapsing
# (up to +509.8% vs Knative, the averageOccupation packing failure). That checkpoint was
# trained from the tabular cache; `_tempfix` is the same architecture trained from the
# batch cache the GNN uses, so its platform features match dim22 inference. This arm
# settles which of the two explanations holds:
#
#   * collapse SURVIVES retraining -> the failure is the pointwise architecture, and the
#     "GNN is the only arm that beats Knative on every cell" claim hardens into an
#     architectural one.
#   * collapse DISAPPEARS -> it was a training-data artifact of the bad cache, and the
#     GNN's reliability advantage over the MLP has to be restated.
#
# Everything except the checkpoint and the sweep dir is byte-identical to
# mlp_arm_all_gates.sbatch on purpose: this is an A/B on the checkpoint, so the cells, the
# workloads and the parity waivers must not move. In particular the block -> (cells,
# workload, PARITY_EXTRA_ARGS) mapping below is matched per-gate to the Knative arm each
# run is scored against, and the bbrob blocks must keep PARITY_EXTRA_ARGS="".
#
# SWEEP_DIR gets its own `_mlptempfix` suffix because run_full_corpus_siv1_live_gate.sh
# names the result file from the POLICY (`<cell>_s0_mlp_dim22.json`), not the checkpoint.
# Two MLP arms sharing one sweep dir would overwrite each other -- worse, the runner's
# "SKIP (exists)" branch would silently keep the first arm's numbers and report success.
#
# 30 tasks: 3 gates x 2 conditions x 5 cells.
#SBATCH --job-name=mlp-tempfix-gate-arm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/mlp-tempfix-gate-arm-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/mlp-tempfix-gate-arm-%A_%a.err
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

# run_simulation.py re-spawns the simulator; since 2026-08-23 it honors HEROSIM_PY, so
# this now really is the activated env end to end. Print both to make it checkable.
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
BLOCK=$(( I / 5 ))

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
  *) echo "FAIL LOUD: bad block $BLOCK (array must be 0-29)" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/${PREFIX}_${COND}_mlptempfix"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"
export MLP_MODEL="models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt"
export INFERENCE_FEATURE_LAYOUT=dim22

# The gate runner insists on a GNN checkpoint existing even when the policy is mlp; point
# it at the deployed one so the check is truthful. It is never loaded on this path.
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"

[[ -f "$MLP_MODEL" ]] || { echo "FAIL LOUD: missing $MLP_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# The A/B is only meaningful if this arm really loaded the tempfix weights. The runner
# passes --mlp-model through, but a stale SWEEP_DIR result would be reported as a SKIP,
# so assert the sweep dir is not the first MLP arm's before anything runs.
[[ "$SWEEP_DIR" == *_mlptempfix ]] || {
  echo "FAIL LOUD: SWEEP_DIR must not be shared with the first MLP arm: $SWEEP_DIR" >&2; exit 1; }

# mkdir -p + rsync, never `[[ -d ]] || cp -r` (not atomic; killed 3 tasks of job 710315).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} trace=${WL} arm=mlptempfix layout=dim22 ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh mlp "$CELL"
echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} COMPLETE ==="
