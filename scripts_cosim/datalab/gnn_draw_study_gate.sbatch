#!/bin/bash
# gnn_draw_study_v1 — gate all 8 GNN draws on the same 30 cells p5b_draw_study used.
#
# 8 checkpoints x 30 cells = 240 tasks, throttled to 60 concurrent (~6-9 min per task).
#
# The 30 cells are 5 topology cells x 6 blocks, and the block -> (cells, workload,
# PARITY_EXTRA_ARGS) mapping is copied VERBATIM from p5b_draw_study_gate.sbatch. It has to
# be: the whole point is to compare this GNN draw distribution against the MLP draw
# distribution already measured on exactly these cells, and a cell that differs is a cell
# that cannot be compared. The Knative arm each run is scored against already exists in
# simulation_data/gate_stats_summary.json for all 30 — nothing here re-runs it.
#
# Arm naming carries the design, for the same reason it did in the MLP study:
# run_full_corpus_siv1_live_gate.sh names the result file from the POLICY
# (`<cell>_s0_gnn.json`), never from the checkpoint. Two arms sharing a SWEEP_DIR would hit
# the runner's "SKIP (exists)" branch and each silently report the other's numbers.
#
#     arm = gnndraws<SEED>    e.g. gnndraws1 .. gnndraws8
#
# INFERENCE_FEATURE_LAYOUT is deliberately NOT exported. The GNN checkpoint's
# .contract.json declares it and load_gnn_model enforces it; exporting a conflicting value
# raises. (This is the one line that must NOT be copied from the MLP gate, which does set
# it.)
#SBATCH --job-name=gnn-draw-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/gnn-draw-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/gnn-draw-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-239%60

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

# Serving device is explicit as of 2026-08-25. CPU-amd has no GPU so this resolves the same
# way either way, but recording it beats inferring it: cuda is the only axis PARITY.md finds
# that moves GNN logits at all.
export HEROSIM_GNN_DEVICE=cpu

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"

CKPT_IDX=$(( I / 30 ))        # 0..7
J=$(( I % 30 ))
CELL="${CELLS[$(( J % 5 ))]}"
BLOCK=$(( J / 5 ))

SEED=$(( CKPT_IDX + 1 ))      # seeds 1..8
[[ "$SEED" -ge 1 && "$SEED" -le 8 ]] || { echo "FAIL LOUD: bad seed $SEED (array must be 0-239)" >&2; exit 1; }

ARM="gnndraws${SEED}"
export GNN_MODEL="models/gnn-draw-s${SEED}.pt"

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
  *) echo "FAIL LOUD: bad block $BLOCK" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/${PREFIX}_${COND}_${ARM}"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"

[[ -f "$GNN_MODEL" ]] || { echo "FAIL LOUD: missing $GNN_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }
[[ "$SWEEP_DIR" == *"_gnndraws"* ]] || {
  echo "FAIL LOUD: SWEEP_DIR must be draw-study private: $SWEEP_DIR" >&2; exit 1; }

# The study attributes variance to the seed, so serving the wrong draw would be silent and
# would look like a result. The filename is not evidence; the contract is.
python3 - "$GNN_MODEL" "$SEED" <<'PY'
import json, sys
from pathlib import Path
path, want_seed = sys.argv[1], int(sys.argv[2])
sidecar = Path(path).with_suffix(".contract.json")
if not sidecar.is_file():
    raise SystemExit(f"FAIL LOUD: {sidecar} missing — serving would silently adopt defaults")
c = json.load(open(sidecar))
if int(c.get("train_seed", -1)) != want_seed:
    raise SystemExit(f"FAIL LOUD: {path} contract train_seed={c.get('train_seed')} != {want_seed}")
if not c.get("deterministic_algorithms"):
    raise SystemExit(f"FAIL LOUD: {path} is not a reproducible draw")
print(f"[CKPT] {path} seed={want_seed} layout={c.get('inference_feature_layout')} "
      f"queue={c.get('queue_feature_contract')}")
PY

mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
# rsync, not `[[ -d ]] || cp -r`: the non-atomic form killed three tasks of job 710315 when
# an arm's tasks started together (datalab-pitfalls #7).
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} arm=${ARM} seed=${SEED} ==="
# No --seed is passed to the simulator anywhere in this chain: it would OVERRIDE the config's
# topology seed and turn a seed study into a topology study.
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh gnn "$CELL"
echo "=== task=${I} arm=${ARM} cell=${CELL} COMPLETE ==="
