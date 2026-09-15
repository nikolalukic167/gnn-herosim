#!/bin/bash
# offline_live_transfer_v1 -- R2 step 1: decode every registered checkpoint on ONE fixed
# state set, so the surrogate candidates are measured with the same yardstick for all 96.
#
# The yardstick is drainable_debug_v1's D1 cache -- 151 states captured while real arms were
# being served, brute-forced into datasets with their full enumerated sweep. Using one common
# state set (rather than each family's own validation split) is deliberate: a surrogate is
# meant to be a property OF THE CHECKPOINT, comparable across families.
#
# 48 array tasks = 3 families x 16 seeds; each decodes both arms.
#
#   sbatch --array=0 scripts_cosim/datalab/offline_live_transfer_v1_r2_decode.sbatch   # smoke
#   sbatch --array=0-47%8 scripts_cosim/datalab/offline_live_transfer_v1_r2_decode.sbatch
#
#SBATCH --job-name=olt-r2dec
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/olt-r2dec-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/olt-r2dec-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-47

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CACHE="${CACHE:-simulation_data/graphs_cache_drainable_debug_v1_d1}"
OUT_DIR="${OUT_DIR:-simulation_data/offline_live_transfer_v1/r2_decoded}"
[[ -d "$CACHE" ]] || { echo "FAIL LOUD: D1 cache missing: $CACHE" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# The three REGISTERED families, in the order the node lists them. The two burned families
# (peer-affinity-v1-t1b, drainable-objective-v1) are absent on purpose.
TAGS=(peer-affinity-v1-x800p2 peer-affinity-v1-x800p3 peer-affinity-v1-warm)
for t in "${TAGS[@]}"; do
  case "$t" in
    *t1b*|*drainable-objective*)
      echo "FAIL LOUD: $t is a BURNED family and must not be scored here" >&2; exit 1 ;;
  esac
done

T=${SLURM_ARRAY_TASK_ID}
FAM=$(( T / 16 )); SEED=$(( T % 16 + 1 ))
TAG="${TAGS[$FAM]}"
OUT="$OUT_DIR/${TAG}-seed${SEED}.json"
echo "[TASK $T] tag=$TAG seed=$SEED -> $OUT"
if [[ -f "$OUT" ]]; then echo "[TASK $T] already decoded, skipping"; exit 0; fi

for arm in gnn mpoff; do
  CK="models/${TAG}-${arm}-lr2e3-seed${SEED}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  [[ -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
done

${HEROSIM_PY} scripts_cosim/drainable_debug_d1_decode.py \
  --cache-dir "$CACHE" \
  --tag "$TAG" --lr lr2e3 --arms gnn mpoff --seed "$SEED" \
  --out "$OUT.partial"
mv "$OUT.partial" "$OUT"
echo "=== task $T done: $OUT ==="
