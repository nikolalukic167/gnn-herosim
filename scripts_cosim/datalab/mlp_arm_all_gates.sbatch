#!/bin/bash
# The MLP verification baseline, added to all three network-contention gates.
#
# drawgate, promo175 and bbrob all compare GNN draws against Knative only. CLAUDE.md keeps
# the pointwise MLP precisely so a graph-aware win can be shown to be graph-aware and not
# just "any learned model beats Knative here", so every published cell should carry an MLP
# number next to it. This produces the missing arm; score_live_gate_matrix.py then reads it
# like any other arm (--arms ...,mlp).
#
# One MLP checkpoint, not one per GNN draw: the MLP is a baseline, not a competitor draw.
#
# INFERENCE_FEATURE_LAYOUT=dim22 matches the three gates exactly. The MLP self-declares its
# layout from inside the checkpoint (mlp_scheduler.py), so the pin AGREES with it rather
# than overriding it -- it is set so this arm records the same serving configuration as the
# arms it will be scored beside, which the scorer's consistency guard requires.
#
# 30 tasks: 3 gates x 2 conditions x 5 cells.
#SBATCH --job-name=mlp-gate-arm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/mlp-gate-arm-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/mlp-gate-arm-%A_%a.err
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

export SWEEP_DIR="simulation_data/normal_sim_sweeps/${PREFIX}_${COND}_mlp"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"
export MLP_MODEL="models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt"
export INFERENCE_FEATURE_LAYOUT=dim22

# The gate runner insists on a GNN checkpoint existing even when the policy is mlp; point
# it at the deployed one so the check is truthful. It is never loaded on this path.
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"

[[ -f "$MLP_MODEL" ]] || { echo "FAIL LOUD: missing $MLP_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# mkdir -p + rsync, never `[[ -d ]] || cp -r` (not atomic; killed 3 tasks of job 710315).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} trace=${WL} arm=mlp layout=dim22 ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh mlp "$CELL"
echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} COMPLETE ==="
