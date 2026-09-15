#!/bin/bash
# drainable_objective_v1 -- Phase C read: the C1 behaviour control off the retained raw
# arms, then the gate read that decides C0/C1/C3/C4/C5 (C2 WITHDRAWN by Amendment 1).
#
#   sbatch scripts_cosim/datalab/drainable_objective_v1_c_read.sbatch
#
#SBATCH --job-name=dobj-cread
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dobj-cread-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dobj-cread-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
RES="${RES:-$GATE/results/dobj_f4000_pg16}"
CONTROL="${CONTROL:-$GATE/results/drain_f4000_E_pg16}"
WORKLOAD="${WORKLOAD:-$GATE/workloads/drainable_f4000_n50000.json}"
OUT_DIR=simulation_data/drainable_objective_v1
mkdir -p "$OUT_DIR"

[[ -d "$CONTROL" ]] || { echo "FAIL LOUD: control results missing: $CONTROL" >&2; exit 1; }

# --- C1: the behaviour control, off the four raw-retained arms -----------------
RAW_ARGS=()
for arm in v1_gnn_s1 v1_gnn_s2 v1_mpoff_s1 v1_mpoff_s2; do
  f="$RES/${arm}.json"
  [[ -f "$f" ]] || { echo "FAIL LOUD: raw arm missing: $f" >&2; exit 1; }
  RAW_ARGS+=(--raw "${arm}=${f}")
done

${HEROSIM_PY} scripts_cosim/drainable_debug_pertask_read.py \
  "${RAW_ARGS[@]}" \
  --workload "$WORKLOAD" \
  --reference v1_mpoff_s1 \
  --out "$OUT_DIR/c1_pertask.json"

# --- The gate read ------------------------------------------------------------
${HEROSIM_PY} scripts_cosim/drainable_objective_v1_gate_read.py \
  --results "$RES" \
  --raw-read "$OUT_DIR/c1_pertask.json" \
  --control-results "$CONTROL" \
  --out "$OUT_DIR/c_gate_read.json"

echo "=== C read done: $OUT_DIR/c_gate_read.json ==="
