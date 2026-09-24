#!/bin/bash
# P5b draw study — gate all 16 checkpoints on the same 30 backbone cells.
#
# 16 checkpoints x 30 cells = 480 tasks. Throttled to 60 concurrent, which is what the
# previous 60-task array ran at comfortably (~6-9 min per task).
#
# Arm naming carries the whole design, because run_full_corpus_siv1_live_gate.sh names the
# result file from the POLICY (`<cell>_s0_mlp_dim22.json`) and not from the checkpoint:
# every checkpoint therefore needs its own SWEEP_DIR or the runner's "SKIP (exists)" branch
# silently reports another arm's numbers as success.
#
#     arm = ds<TAG><LAYOUT>s<SEED>    e.g. dsdim14dim22s1, dstempfixdim25crs4
#
# The block -> (cells, workload, PARITY_EXTRA_ARGS) mapping is copied VERBATIM from
# mlp_candrel_arm_all_gates.sbatch: it is matched per-gate to the Knative arm each run is
# scored against, and the bbrob blocks must keep PARITY_EXTRA_ARGS="".
#SBATCH --job-name=p5b-draw-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p5b-draw-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p5b-draw-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-479%60

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

CKPT_IDX=$(( I / 30 ))        # 0..15
J=$(( I % 30 ))
CELL="${CELLS[$(( J % 5 ))]}"
BLOCK=$(( J / 5 ))

SEEDS=(1 2 3 4)
SEED="${SEEDS[$(( CKPT_IDX % 4 ))]}"
COMBO=$(( CKPT_IDX / 4 ))
case "$COMBO" in
  0) TAG=dim14;   LAYOUT=dim22   ;;
  1) TAG=dim14;   LAYOUT=dim25cr ;;
  2) TAG=tempfix; LAYOUT=dim22   ;;
  3) TAG=tempfix; LAYOUT=dim25cr ;;
  *) echo "FAIL LOUD: bad combo $COMBO (array must be 0-479)" >&2; exit 1 ;;
esac

ARM="ds${TAG}${LAYOUT}s${SEED}"
export MLP_MODEL="models/tabular/draw_study/mlp_${TAG}_${LAYOUT}_s${SEED}.pt"
export INFERENCE_FEATURE_LAYOUT="$LAYOUT"

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
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"

[[ -f "$MLP_MODEL" ]] || { echo "FAIL LOUD: missing $MLP_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }
[[ "$SWEEP_DIR" == *"_ds"* ]] || {
  echo "FAIL LOUD: SWEEP_DIR must be draw-study private: $SWEEP_DIR" >&2; exit 1; }

# The whole study is about attributing variance to the seed, so assert the checkpoint
# really is the seeded one this task is supposed to serve, and really is the declared
# layout, before spending the run.
python3 - "$MLP_MODEL" "$LAYOUT" "$SEED" <<'PY'
import json, sys, torch
path, want_layout, want_seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(path, map_location="cpu", weights_only=False)
meta = json.load(open(path + ".meta.json"))
if ck.get("inference_feature_layout") != want_layout:
    raise SystemExit(f"FAIL LOUD: {path} layout={ck.get('inference_feature_layout')!r} != {want_layout!r}")
if not ck.get("torch_seeded"):
    raise SystemExit(f"FAIL LOUD: {path} is not torch-seeded; its init is an unreproducible draw")
if int(meta["random_state"]) != want_seed:
    raise SystemExit(f"FAIL LOUD: {path} seed={meta['random_state']} != {want_seed}")
print(f"[CKPT] {path} layout={want_layout} seed={want_seed} input_dim={ck['input_dim']}")
PY

mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} arm=${ARM} layout=${LAYOUT} seed=${SEED} ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh mlp "$CELL"
echo "=== task=${I} arm=${ARM} cell=${CELL} COMPLETE ==="
