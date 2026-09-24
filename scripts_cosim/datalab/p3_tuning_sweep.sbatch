#!/bin/bash
# objective_pivot_v1 Phase 3, Amendment D3 — the pre-registered tuning sweep.
#
# WHY THIS EXISTS
#   The kill criterion is unforgiving: paired improvement <= 0 freezes P1 as
#   measured-negative with no re-runs. Pointed at a loop whose learning rate was never
#   examined, that turns a tuning failure into "closed-loop RL does not work here" and
#   forbids the re-run that would catch it. This buys the method one fair shot, on cells
#   that the gate never sees.
#
# THE SPLIT (Amendment D3, fixed before this ran)
#   train : bbrob_bb_core4_bw0p5  cell01, cell02, cell04
#   dev   : bbrob_bb_core4_bw0p5  cell03, cell05   <- selection ONLY, never a reported gate
#   gate  : bbrob_bb_core8_bw1p5  all 5 cells      <- unseen fabric AND unseen cells
#
#   The winner is frozen before the pilot and is NOT revisited. A second tuning pass after
#   seeing pilot or gate numbers is exactly the fishing the kill criterion prevents.
#
# GRID: lr in {1e-6, 1e-5, 1e-4} x T in {0.1, 0.3}, one seed each, 20 steps.
# Selection statistic: greedy total RTT on the two dev cells, lower is better.
#
# BOTH TRAINED ARMS GET THIS, on the same grid and the same budget. Tuning only the GNN
# and handing the MLP an inherited learning rate would bias the one comparison the whole
# program exists to make; the original Phase 3 registration says the MLP runs in the
# identical loop with the same objective, budget and seeds, and a tuned-vs-untuned pair
# is not that. Select the arm with ARM=gnn|mlp and INIT=<checkpoint>.
#
#SBATCH --job-name=p3-tune
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-tune-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-tune-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5

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

LRS=(1e-6 1e-6 1e-5 1e-5 1e-4 1e-4)
TEMPS=(0.1 0.3 0.1 0.3 0.1 0.3)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
T="${TEMPS[$SLURM_ARRAY_TASK_ID]}"

ARM="${ARM:-gnn}"
case "$ARM" in
  gnn) INIT="${INIT:-models/gnn-linkmp-lgon-s8.pt}" ;;   # Amendment A
  mlp) INIT="${INIT:-models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt}" ;;
  *)   echo "FAIL LOUD: ARM must be gnn or mlp, got $ARM" >&2; exit 1 ;;
esac
[[ -f "$INIT" ]] || { echo "FAIL LOUD: missing $INIT" >&2; exit 1; }

TAG="${ARM}_lr${LR}_T${T}"
OUT="simulation_data/p3_tuning/${TAG}"

TRAIN_SWEEP=simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core4_bw0p5
WORKLOAD=data/nofs-ids/traces/workload-150-100-30k.json

echo "=== tuning arm $SLURM_ARRAY_TASK_ID: policy=$ARM lr=$LR T=$T ==="

python3 scripts_cosim/closed_loop/train_closed_loop.py \
  --arm "$ARM" --init "$INIT" \
  --sweep-dir "$TRAIN_SWEEP" \
  --cells cell01_p25_s9001 cell02_p35_s9002 cell04_p50_s9004 \
  --workload "$WORKLOAD" \
  --out-dir "$OUT" \
  --steps 20 --episodes-per-cell 4 \
  --temperature "$T" --lr "$LR" --reservoir-k 64 --seed 1 \
  --episode-workers 8 \
  --wandb-run-name "tune-$TAG"

# Selection statistic on the DEV cells, produced by the same script that will produce the
# gate's numbers -- a configuration must not be chosen under one measurement and judged
# under another.
python3 scripts_cosim/closed_loop/evaluate_policy.py \
  --sweep-dir "$TRAIN_SWEEP" \
  --cells cell03_p15_s9003 cell05_p20_s9005 \
  --workload "$WORKLOAD" \
  --arm "${ARM}:frozen=$INIT" \
  --arm "${ARM}:tuned_${TAG}=${OUT}/checkpoints/cl_${ARM}_step020.pt" \
  --out "${OUT}/dev_eval.json" \
  --workers 4

echo "=== tuning arm $SLURM_ARRAY_TASK_ID COMPLETE ($TAG) ==="
