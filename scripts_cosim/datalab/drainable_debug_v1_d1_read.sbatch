#!/bin/bash
# drainable_debug_v1 -- D1, step 5: decode both checkpoints on the captured states, then apply
# the registered bars. Re-simulates nothing.
#   sbatch --dependency=afterok:<cache job> scripts_cosim/datalab/drainable_debug_v1_d1_read.sbatch
#
#SBATCH --job-name=dbg-d1-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_GNN_DEVICE=cpu
export DAG_PRIMARY_ALPHA_KEY=2.5 PARTIAL_STATE_CONTRACT=partial_state_v2 PARTIAL_STATE_PEER_MASS=1
export INFERENCE_FEATURE_LAYOUT=dim22 QUEUE_FEATURE_CONTRACT=legacy_v0
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CACHE="${CACHE:-simulation_data/graphs_cache_drainable_debug_v1_d1}"
OUTDIR=simulation_data/drainable_debug_v1
DECODED="$OUTDIR/d1_decoded.json"
mkdir -p "$OUTDIR" logs
[[ -d "$CACHE" ]] || { echo "FAIL LOUD: cache missing: $CACHE" >&2; exit 1; }

echo "=== [1/2] decode both checkpoints on every captured state"
python3 -u scripts_cosim/drainable_debug_d1_decode.py \
  --cache-dir "$CACHE" --arms gnn mpoff --seed "${GATE_SEED:-1}" --out "$DECODED"

echo "=== [2/2] registered read"
# One invocation per source corpus: the read walks ds_* dirs, and each source is its own corpus.
for S in knb gnn mpoff; do
  C="simulation_data/gnn_datasets_drainable_debug_v1_d1_${S}"
  [[ -d "$C" ]] || { echo "FAIL LOUD: corpus missing: $C" >&2; exit 1; }
done
python3 -u scripts_cosim/drainable_debug_one_step_read.py \
  --corpus simulation_data/gnn_datasets_drainable_debug_v1_d1_knb \
  --decoded "$DECODED" --arm gnn --out "$OUTDIR/d1_read_knb.json"
python3 -u scripts_cosim/drainable_debug_one_step_read.py \
  --corpus simulation_data/gnn_datasets_drainable_debug_v1_d1_gnn \
  --decoded "$DECODED" --arm gnn --out "$OUTDIR/d1_read_gnn.json"
python3 -u scripts_cosim/drainable_debug_one_step_read.py \
  --corpus simulation_data/gnn_datasets_drainable_debug_v1_d1_mpoff \
  --decoded "$DECODED" --arm gnn --out "$OUTDIR/d1_read_mpoff.json"
echo "=== D1 read done ==="
