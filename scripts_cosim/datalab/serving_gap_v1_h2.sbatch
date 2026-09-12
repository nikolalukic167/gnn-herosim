#!/bin/bash
# serving_gap_v1 stage 1 — H2 (queue insensitivity), one array task per (corpus, arm, seed).
# Reads the graph-carrying traces written by serving_gap_v1_traces.sbatch with TRACE_SET=h2,
# re-scores each sampled graph with every platform told it is one standard deviation busier,
# and records how much the arm's preference moves. Same index map as the trace array.
#
#SBATCH --job-name=sg-h2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sg-h2-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sg-h2-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-95

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" HEROSIM_GNN_DEVICE=cpu

T=${SLURM_ARRAY_TASK_ID}
CORPORA=(x200p2 x800p2 x800p3); ARMS=(gnn mpoff)
CORPUS=${CORPORA[$(( T / 32 ))]}; ARM=${ARMS[$(( (T % 32) / 16 ))]}; SEED=$(( T % 16 + 1 ))
case $CORPUS in
  x200p2) CKPREFIX=models/peer-affinity-v1-t1b ;;
  x800p2) CKPREFIX=models/peer-affinity-v1-x800p2 ;;
  x800p3) CKPREFIX=models/peer-affinity-v1-x800p3 ;;
esac
CK="${CKPREFIX}-${ARM}-lr2e3-seed${SEED}.pt"
TRACE=simulation_data/serving_gap_v1/traces_h2/$CORPUS/${ARM}_s${SEED}.pkl
OUT_DIR=simulation_data/serving_gap_v1/h2/$CORPUS
OUT=$OUT_DIR/${ARM}_s${SEED}.json
mkdir -p "$OUT_DIR" logs
[[ -f "$OUT" ]] && { echo "[skip] $OUT exists"; exit 0; }
[[ -s "$TRACE" ]] || { echo "FAIL LOUD: trace missing: $TRACE" >&2; exit 1; }
[[ -f "$CK" ]] || { echo "FAIL LOUD: checkpoint missing: $CK" >&2; exit 1; }

# the arm environment the registered evaluator uses, so the loader's contract check passes
export PARTIAL_STATE_CONTRACT=partial_state_v2 PARTIAL_STATE_PEER_MASS=1
export GNN_DECODE_MODE=masked_topo
if [[ $ARM == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; else unset GNN_DISABLE_MESSAGE_PASSING; fi

python3 scripts_cosim/serving_gap_trace_read.py h2-one \
  --corpus "$CORPUS" --arm "$ARM" --seed "$SEED" --checkpoint "$CK" --trace "$TRACE" --output "$OUT"
echo "=== task $T done ==="
