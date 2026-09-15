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
# Feature environment copied VERBATIM from drainable_debug_v1_d1_read.sbatch:23-25, the run
# whose decode these plans must be comparable to. Written from memory instead of copied, the
# first smoke died on the partial-state contract guard -- which is the guard doing its job,
# since the default resolves to v1 and every family here is v2.
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_GNN_DEVICE=cpu
export DAG_PRIMARY_ALPHA_KEY=2.5 PARTIAL_STATE_CONTRACT=partial_state_v2 PARTIAL_STATE_PEER_MASS=1
export INFERENCE_FEATURE_LAYOUT=dim22 QUEUE_FEATURE_CONTRACT=legacy_v0

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CACHE="${CACHE:-simulation_data/graphs_cache_drainable_debug_v1_d1}"
# QUEUE_SCALE=1.0 is the unscaled pass. The second pass uses R2_QUEUE_SCALE from
# offline_live_transfer_v1_surrogate_read.py and writes to its own OUT_DIR, so the two
# decodes never overwrite each other and range_sensitivity compares like with like.
QUEUE_SCALE="${QUEUE_SCALE:-1.0}"
if [[ "$QUEUE_SCALE" == "1.0" ]]; then
  OUT_DIR="${OUT_DIR:-simulation_data/offline_live_transfer_v1/r2_decoded}"
else
  OUT_DIR="${OUT_DIR:-simulation_data/offline_live_transfer_v1/r2_decoded_scaled}"
fi
echo "[SCALE] queue depth x $QUEUE_SCALE -> $OUT_DIR"
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
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
  # Do not trust the exported contract: check it against what this checkpoint was trained
  # under, and against the architecture its arm name claims.
  python3 - "$SC" "$arm" <<'GUARD'
import json, os, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
for key, env in (("partial_state_contract", "PARTIAL_STATE_CONTRACT"),
                 ("queue_feature_contract", "QUEUE_FEATURE_CONTRACT")):
    want, got = os.environ.get(env), sc.get(key)
    if got is not None and want != got:
        raise SystemExit(f"FAIL LOUD: {env}={want!r} but the checkpoint was trained under "
                         f"{key}={got!r}")
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing="
                     f"{sc.get('disable_message_passing')!r} for arm {arm}")
GUARD
done

${HEROSIM_PY} scripts_cosim/drainable_debug_d1_decode.py \
  --cache-dir "$CACHE" \
  --tag "$TAG" --lr lr2e3 --arms gnn mpoff --seed "$SEED" \
  --queue-scale "$QUEUE_SCALE" \
  --out "$OUT.partial"
mv "$OUT.partial" "$OUT"
echo "=== task $T done: $OUT ==="
